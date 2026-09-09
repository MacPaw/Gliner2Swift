#!/usr/bin/env python3
"""Generate the end-to-end prediction-parity corpus (IMPLEMENTATION_PLAN.md §0.4).

This is the master gate for the whole perf effort: every case records what Python
GLiNER2 predicts for a (text, schema, threshold) triple, and the Swift side must
reproduce it exactly. Perf refactors are only safe while this stays green.

Design notes
------------
* Python is always called with ``format_results=True, include_confidence=True,
  include_spans=True``. The default formatted output collapses entities to bare
  strings, which would hide both span offsets and confidences from the comparator.
* Schemas are described declaratively (see ``SCHEMA SPEC`` below) so that the Swift
  test can rebuild a byte-identical schema from the same JSON.
* Every decision's ``|confidence - threshold|`` margin is recorded. Phase 3's
  "predictions stay exact" gate is vacuous unless some cases sit near a decision
  boundary, so the summary reports the margin distribution and flags borderline
  (< 0.02) decisions for the tolerance policy in §9.
* Corpus inputs only use API surface the Swift port already has: no prompt /
  examples / label_descriptions (§6.5) and no max_len (§6.1).

SCHEMA SPEC
-----------
    {
      "entities":       ["person", ...]  |  {"person": "description", ...},
      "classification": {"task", "labels", "multi_label", "cls_threshold"},
      "structures":     [{"name", "fields": [{"name", "dtype", "choices",
                                              "description", "threshold"}]}],
      "relations":      ["works_at", ...]
    }

Usage
-----
    # v1 span checkpoint (default, unchanged)
    /path/to/GLiNER2/.venv/bin/python scripts/generate_prediction_corpus.py

    # GLiNER 2.5 boundary checkpoint (needs the gliner2 2.0.0 venv, scripts/requirements-gliner25.txt)
    /Users/tmwstw/Documents/mnemos/gliner2-ref/.venv/bin/python \
        scripts/generate_prediction_corpus.py --architecture boundary \
        --model fastino/gliner2.5-small-v1

Output
------
    ../Tests/GLiNER2SwiftTests/Fixtures/prediction_corpus/corpus.json                 (span)
    ../Tests/GLiNER2SwiftTests/Fixtures/prediction_corpus_boundary/<small|base|multi>/corpus.json

Boundary mode (``--architecture boundary``)
-------------------------------------------
The SAME 58 texts/schemas (plus the derived borderline cases) are run through
``AutoExtractor.from_pretrained`` in the identical JSON format (``expected`` /
``margins`` / ``tags``), so ``PredictionParityTests`` can consume either corpus. What the
boundary runtime does differently, for the reader of the expected values:

* Structures are declared with ``schema.structure(name)`` and NO ``mode`` -> no
  ``record_metadata`` in the schema dict -> the runtime takes the LEGACY structure path
  (``_decode_legacy_structures``): exactly one instance per structure holding every
  list-valued field, the first surviving span for ``dtype="str"`` fields, and choice
  fields scored through ``score_explicit_spans`` at their prompt-prefix positions.
  ``extract_json`` (record mode via the record head) is not exercised here.
* v1's count-slot multi-instance structures therefore collapse to one instance
  (``struct_multi_instance`` and friends), and there is no ``max_width``: any span
  length is representable.
* Default overlap policy is ``flat`` (max-total-confidence non-overlapping set), a
  per-query abstention gate can empty a label (``sigmoid(null_logit) > 0.5``), and
  relations come from the sparse typed pair generator + ``relation_scorer`` with the
  boundary-specific edge deduplication.
* Everything else (prompt layout, ``"."`` suffixing, word offsets, ``include_spans``
  character offsets, classification decode) is shared with v1.
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

FIXTURES_DIR = Path(__file__).parent.parent / "Tests" / "GLiNER2SwiftTests" / "Fixtures"
DEFAULT_MODELS = {
    "span": os.environ.get("GLINER2_CORPUS_MODEL", "fastino/gliner2-base-v1"),
    "boundary": os.environ.get("GLINER2_CORPUS_MODEL", "fastino/gliner2.5-small-v1"),
}


def model_size_name(model_id: str) -> str:
    """'fastino/gliner2.5-small-v1' -> 'small' (falls back to a sanitised basename)."""
    base = os.path.basename(model_id.rstrip("/")).lower()
    for size in ("small", "base", "multi", "large"):
        if f"-{size}-" in base or base.endswith(f"-{size}") or f"-{size}_" in base:
            return size
    return "".join(ch if ch.isalnum() else "_" for ch in base)


def default_output_dir(architecture: str, model_id: str) -> Path:
    if architecture == "boundary":
        return FIXTURES_DIR / "prediction_corpus_boundary" / model_size_name(model_id)
    return FIXTURES_DIR / "prediction_corpus"


def load_model(architecture: str, model_id: str):
    """Span checkpoints load through the v1 class; boundary ones through AutoExtractor.
    Both expose the same ``create_schema`` / ``extract`` runtime API."""
    if architecture == "boundary":
        from gliner2 import AutoExtractor
        model = AutoExtractor.from_pretrained(model_id)
        if getattr(model, "architecture", None) != "boundary":
            raise SystemExit(f"{model_id} is not a boundary checkpoint")
        return model
    from gliner2 import GLiNER2
    return GLiNER2.from_pretrained(model_id)

# --------------------------------------------------------------------------------------
# Corpus definition
#
# tags drive the Swift side's `expectedFailures` list; each tag names the plan phase that
# makes the case pass. Untagged cases must pass from Phase 0 onward.
# --------------------------------------------------------------------------------------

LONG_TEXT = (
    "The quarterly review meeting was held in San Francisco on March 15, where Tim Cook "
    "and Satya Nadella discussed a joint venture between Apple and Microsoft. "
) * 12

CASES = [
    # ---- plain NER -------------------------------------------------------------------
    dict(name="ner_basic", text="Tim Cook is the CEO of Apple.", threshold=0.5,
         schema={"entities": ["person", "company"]}),
    dict(name="ner_many_labels",
         text="Dr. Jane Smith met Barack Obama in Berlin on Tuesday to discuss Tesla and Google.",
         threshold=0.5,
         schema={"entities": ["person", "company", "location", "date", "title",
                              "organization", "product", "event"]}),
    dict(name="ner_no_match", text="The weather is quite pleasant today.", threshold=0.5,
         schema={"entities": ["person", "company"]}),
    dict(name="ner_repeated_mention",
         text="Apple released a phone. Apple also released a laptop. Apple is busy.",
         threshold=0.5, schema={"entities": ["company"]},
         tags=["dedup"]),
    dict(name="ner_threshold_030", text="Tim Cook is the CEO of Apple.", threshold=0.3,
         schema={"entities": ["person", "company", "location"]}),
    dict(name="ner_threshold_070", text="Tim Cook is the CEO of Apple.", threshold=0.7,
         schema={"entities": ["person", "company", "location"]}),
    dict(name="ner_descriptions_unordered",
         text="Tim Cook visited Berlin with Satya Nadella.", threshold=0.5,
         schema={"entities": {"person": "a human being",
                              "company": "a business organisation",
                              "location": "a geographic place"}},
         tags=["entity_order"]),
    dict(name="ner_long_text", text=LONG_TEXT, threshold=0.5,
         schema={"entities": ["person", "company", "location"]}),
    dict(name="ner_empty_text", text="", threshold=0.5,
         schema={"entities": ["person"]}),
    dict(name="ner_punctuation_only", text="!!! ???", threshold=0.5,
         schema={"entities": ["person"]}),

    # ---- non-ASCII (Phase 1.2 charsmap) ----------------------------------------------
    dict(name="nonascii_nfd_accents",
         text="Zoé Dupont works at Café René in Montréal.",
         threshold=0.5, schema={"entities": ["person", "company", "location"]},
         tags=["non_ascii"]),
    dict(name="nonascii_precomposed",
         text="Zoé Dupont works at Café René in Montréal.",
         threshold=0.5, schema={"entities": ["person", "company", "location"]},
         tags=["non_ascii"]),
    dict(name="nonascii_ligature",
         text="The oﬃce of Grifﬁn Ltd is in Shefﬁeld.", threshold=0.5,
         schema={"entities": ["company", "location"]}, tags=["non_ascii"]),
    dict(name="nonascii_fullwidth",
         text="ＡＢＣ Corp hired Ｊｏｈｎ Ｓｍｉｔｈ in Ｔｏｋｙｏ.", threshold=0.5,
         schema={"entities": ["person", "company", "location"]}, tags=["non_ascii"]),
    dict(name="nonascii_micro_sign",
         text="The µm sensor from Micron measures 5 μm precisely.", threshold=0.5,
         schema={"entities": ["company", "product"]}, tags=["non_ascii"]),
    dict(name="nonascii_cjk",
         text="田中太郎は東京のソニーで働いています。", threshold=0.5,
         schema={"entities": ["person", "company", "location"]}, tags=["non_ascii"]),
    dict(name="nonascii_nbsp_zwsp",
         text="Tim Cook joined Apple​Inc in Cupertino.", threshold=0.5,
         schema={"entities": ["person", "company", "location"]}, tags=["non_ascii"]),
    dict(name="nonascii_cyrillic",
         text="Иван Петров работает в Яндекс в Москве.", threshold=0.5,
         schema={"entities": ["person", "company", "location"]}, tags=["non_ascii"]),

    # ---- classification --------------------------------------------------------------
    dict(name="cls_single_positive", text="I love this product! It's amazing.",
         threshold=0.5,
         schema={"classification": {"task": "sentiment",
                                    "labels": ["positive", "negative"]}}),
    dict(name="cls_single_negative", text="This is terrible and broke immediately.",
         threshold=0.5,
         schema={"classification": {"task": "sentiment",
                                    "labels": ["positive", "negative", "neutral"]}}),
    dict(name="cls_multi_label",
         text="The team scored a goal while the new phone launched at the tech expo.",
         threshold=0.5,
         schema={"classification": {"task": "topic",
                                    "labels": ["sports", "technology", "politics", "food"],
                                    "multi_label": True}}),
    # Python emits the argmax label even when nothing crosses the threshold
    # (engine.py multi-label fallback). Swift must do the same.
    dict(name="cls_multi_all_below",
         text="asdf qwer zxcv.", threshold=0.5,
         schema={"classification": {"task": "topic",
                                    "labels": ["sports", "technology", "politics"],
                                    "multi_label": True, "cls_threshold": 0.99}}),
    dict(name="cls_ambiguous",
         text="It was fine, I guess. Not great, not awful.", threshold=0.5,
         schema={"classification": {"task": "sentiment",
                                    "labels": ["positive", "negative", "neutral"]}}),
    dict(name="cls_custom_threshold",
         text="Absolutely fantastic experience.", threshold=0.5,
         schema={"classification": {"task": "sentiment",
                                    "labels": ["positive", "negative"],
                                    "cls_threshold": 0.9}}),

    # ---- structures ------------------------------------------------------------------
    dict(name="struct_contact",
         text="Contact john@email.com or call 555-1234.", threshold=0.5,
         schema={"structures": [{"name": "contact",
                                 "fields": [{"name": "email"}, {"name": "phone"}]}]}),
    dict(name="struct_dtype_str",
         text="Contact john@email.com or call 555-1234.", threshold=0.5,
         schema={"structures": [{"name": "contact",
                                 "fields": [{"name": "email", "dtype": "str"},
                                            {"name": "phone", "dtype": "str"}]}]},
         tags=["field_dtype"]),
    # dtype=str takes the positionally-first span (row-major start/width order), which is
    # not necessarily the highest-confidence one — an argmax-by-confidence port diverges.
    dict(name="struct_dtype_str_first_not_max",
         text="Alice met Bob and then Carol and afterwards Dave joined them.",
         threshold=0.3,
         schema={"structures": [{"name": "meeting",
                                 "fields": [{"name": "person", "dtype": "str"}]}]},
         tags=["field_dtype"]),
    dict(name="struct_multi_instance",
         text=("Alice Johnson works at Acme Corp. Bob Smith works at Globex. "
               "Carol White works at Initech."),
         threshold=0.5,
         schema={"structures": [{"name": "employment",
                                 "fields": [{"name": "employee"}, {"name": "employer"}]}]}),
    dict(name="struct_descriptions",
         text="The summit runs March 15 in Berlin.", threshold=0.5,
         schema={"structures": [{"name": "event",
                                 "fields": [{"name": "date", "description": "when it happens"},
                                            {"name": "place", "description": "where it happens"}]}]}),
    dict(name="struct_field_threshold",
         text="Contact john@email.com or call 555-1234.", threshold=0.5,
         schema={"structures": [{"name": "contact",
                                 "fields": [{"name": "email", "threshold": 0.9},
                                            {"name": "phone", "threshold": 0.1}]}]},
         tags=["field_threshold"]),
    dict(name="struct_all_below_threshold",
         text="Nothing relevant appears in this sentence at all.", threshold=0.99,
         schema={"structures": [{"name": "contact",
                                 "fields": [{"name": "email"}, {"name": "phone"}]}]}),

    # ---- choice fields (Phase 1.1) ---------------------------------------------------
    dict(name="choices_basic",
         text="The laptop arrived damaged and I want a refund.", threshold=0.5,
         schema={"structures": [{"name": "ticket",
                                 "fields": [{"name": "item"},
                                            {"name": "issue", "dtype": "str",
                                             "choices": ["damaged", "missing", "late"]}]}]},
         tags=["choices"]),
    dict(name="choices_list_dtype",
         text="The laptop arrived damaged and late.", threshold=0.5,
         schema={"structures": [{"name": "ticket",
                                 "fields": [{"name": "issue", "dtype": "list",
                                             "choices": ["damaged", "missing", "late"]}]}]},
         tags=["choices"]),
    # Overlapping labels: Python's _find_choice_idx is a single ==-or-contains pass, so
    # "positive" resolves to the "very positive" element. A two-pass exact-first port
    # picks a different score cell.
    dict(name="choices_overlapping_labels",
         text="This product is absolutely wonderful in every way.", threshold=0.5,
         schema={"structures": [{"name": "review",
                                 "fields": [{"name": "rating", "dtype": "str",
                                             "choices": ["very positive", "positive",
                                                         "negative"]}]}]},
         tags=["choices"]),
    dict(name="choices_multiword",
         text="Please ship it with next day air to the Berlin office.", threshold=0.5,
         schema={"structures": [{"name": "order",
                                 "fields": [{"name": "shipping", "dtype": "str",
                                             "choices": ["next day air", "ground",
                                                         "two day"]}]}]},
         tags=["choices"]),
    dict(name="choices_below_threshold_keeps_null",
         text="Completely unrelated sentence about gardening.", threshold=0.5,
         schema={"structures": [{"name": "ticket",
                                 "fields": [{"name": "issue", "dtype": "str",
                                             "choices": ["damaged", "missing", "late"]}]}]},
         tags=["choices"]),
    dict(name="choices_with_span_field_kept",
         text="The laptop arrived damaged.", threshold=0.5,
         schema={"structures": [{"name": "ticket",
                                 "fields": [{"name": "item"},
                                            {"name": "issue", "dtype": "str",
                                             "choices": ["shattered", "vanished"]}]}]},
         tags=["choices"]),

    # ---- relations -------------------------------------------------------------------
    dict(name="rel_basic", text="Tim Cook works at Apple.", threshold=0.5,
         schema={"relations": ["works_at"]}, tags=["relation_grouping"]),
    dict(name="rel_multi_instance",
         text="Tim Cook works at Apple. Satya Nadella works at Microsoft.",
         threshold=0.5, schema={"relations": ["works_at"]}, tags=["relation_grouping"]),
    dict(name="rel_no_match", text="The sky is blue today.", threshold=0.5,
         schema={"relations": ["works_at"]}, tags=["relation_grouping"]),
    dict(name="rel_head_without_tail", text="Tim Cook is a person.", threshold=0.8,
         schema={"relations": ["works_at"]}, tags=["relation_grouping"]),

    # ---- combined multi-task schema --------------------------------------------------
    dict(name="combined_all_tasks",
         text=("Tim Cook, the CEO of Apple, announced a fantastic new product in "
               "Cupertino on March 15. Contact press@apple.com for details."),
         threshold=0.5,
         schema={"entities": ["person", "company", "location", "date"],
                 "classification": {"task": "sentiment",
                                    "labels": ["positive", "negative"]},
                 "structures": [{"name": "contact", "fields": [{"name": "email"}]}],
                 "relations": ["works_at"]},
         tags=["relation_grouping"]),
    dict(name="combined_entities_and_classification",
         text="Apple's new phone is disappointing and overpriced.", threshold=0.5,
         schema={"entities": ["company", "product"],
                 "classification": {"task": "sentiment",
                                    "labels": ["positive", "negative"]}}),
    dict(name="combined_two_structures",
         text="Alice Johnson (alice@acme.com) attended the Berlin summit on May 2.",
         threshold=0.5,
         schema={"structures": [{"name": "contact",
                                 "fields": [{"name": "name"}, {"name": "email"}]},
                                {"name": "event",
                                 "fields": [{"name": "place"}, {"name": "date"}]}]}),
]


# --------------------------------------------------------------------------------------
# Schema construction (mirrored by PredictionParityTests.swift)
# --------------------------------------------------------------------------------------

# GLiNER 2.5 record mode (Instance Formation head). Only meaningful for boundary
# checkpoints: `mode` / `anchor` on a structure, `cardinality` / `exclusive` on a field.
# `records_json_api` mirrors what `extract_json` builds on a 2.5 model (natural mode, first
# field anchor, str fields required_one, others zero_or_more, everything exclusive).
BOUNDARY_ONLY_CASES = [
    dict(name="records_json_api",
         text=("Alice Johnson works at Acme Corp as an engineer. "
               "Bob Smith works at Globex as a manager."),
         threshold=0.5,
         schema={"structures": [{"name": "employment", "mode": "natural",
                                 "fields": [{"name": "employee", "dtype": "str",
                                             "cardinality": "required_one", "exclusive": True},
                                            {"name": "employer", "dtype": "str",
                                             "cardinality": "required_one", "exclusive": True},
                                            {"name": "role", "cardinality": "zero_or_more",
                                             "exclusive": True}]}]},
         tags=["records"]),
    dict(name="records_natural_defaults",
         text=("Alice Johnson works at Acme Corp. Bob Smith works at Globex. "
               "Carol White works at Initech."),
         threshold=0.5,
         schema={"structures": [{"name": "employment", "mode": "natural",
                                 "fields": [{"name": "employee"}, {"name": "employer"}]}]},
         tags=["records"]),
    dict(name="records_explicit_anchor",
         text="Acme Corp hired Alice Johnson. Globex hired Bob Smith.",
         threshold=0.5,
         schema={"structures": [{"name": "hiring", "mode": "natural", "anchor": "company",
                                 "fields": [{"name": "person", "dtype": "str"},
                                            {"name": "company", "dtype": "str"}]}]},
         tags=["records"]),
    dict(name="records_choices_literal",
         text="The laptop arrived damaged and I want a refund. The charger works fine.",
         threshold=0.5,
         schema={"structures": [{"name": "ticket", "mode": "natural",
                                 "fields": [{"name": "product", "dtype": "str",
                                             "cardinality": "required_one", "exclusive": True},
                                            {"name": "resolution", "dtype": "str",
                                             "choices": ["refund", "exchange", "repair"],
                                             "cardinality": "required_one", "exclusive": True}]}]},
         tags=["records", "choices"]),
    dict(name="records_choices_scored",
         text="My headphones stopped working after two days.",
         threshold=0.5,
         schema={"structures": [{"name": "ticket", "mode": "natural",
                                 "fields": [{"name": "product", "dtype": "str",
                                             "cardinality": "required_one", "exclusive": True},
                                            {"name": "issue",
                                             "choices": ["damaged", "defective", "late"],
                                             "cardinality": "zero_or_more", "exclusive": True}]}]},
         tags=["records", "choices"]),
    dict(name="records_latent",
         text="Tim Cook leads Apple. Satya Nadella leads Microsoft.",
         threshold=0.5,
         schema={"structures": [{"name": "leadership", "mode": "latent",
                                 "fields": [{"name": "person", "dtype": "str"},
                                            {"name": "company", "dtype": "str"}]}]},
         tags=["records"]),
    dict(name="records_anchorless",
         text="Tim Cook leads Apple. Satya Nadella leads Microsoft.",
         threshold=0.5,
         schema={"structures": [{"name": "leadership", "mode": "anchorless",
                                 "fields": [{"name": "person", "dtype": "str"},
                                            {"name": "company", "dtype": "str"}]}]},
         tags=["records"]),
    dict(name="records_with_entities",
         text="Jane Doe joined Initech in Austin in 2019.",
         threshold=0.5,
         schema={"entities": ["location", "date"],
                 "structures": [{"name": "employment", "mode": "natural",
                                 "fields": [{"name": "employee", "dtype": "str",
                                             "cardinality": "required_one", "exclusive": True},
                                            {"name": "employer", "dtype": "str",
                                             "cardinality": "required_one", "exclusive": True}]}]},
         tags=["records"]),
]


def build_schema(model, spec):
    """Build a gliner2 Schema from the declarative spec."""
    schema = model.create_schema()

    if "entities" in spec:
        schema = schema.entities(spec["entities"])

    for struct in spec.get("structures", []):
        record_kwargs = {key: struct[key] for key in ("mode", "anchor", "occurrence_policy")
                         if struct.get(key) is not None}
        builder = schema.structure(struct["name"], **record_kwargs)
        for field in struct["fields"]:
            field_kwargs = {}
            if field.get("cardinality") is not None:
                field_kwargs["cardinality"] = field["cardinality"]
            if field.get("exclusive"):
                field_kwargs["exclusive"] = True
            builder = builder.field(
                field["name"],
                dtype=field.get("dtype", "list"),
                choices=field.get("choices"),
                description=field.get("description"),
                threshold=field.get("threshold"),
                **field_kwargs,
            )
        schema = builder.done() if hasattr(builder, "done") else builder

    if "relations" in spec:
        schema = schema.relations(spec["relations"])

    if "classification" in spec:
        cls = spec["classification"]
        schema = schema.classification(
            task=cls["task"],
            labels=cls["labels"],
            multi_label=cls.get("multi_label", False),
            cls_threshold=cls.get("cls_threshold", 0.5),
        )

    return schema


# --------------------------------------------------------------------------------------
# Margin analysis
# --------------------------------------------------------------------------------------

def collect_margins(result, threshold, path="", out=None):
    """Record |confidence - threshold| for every decision in a result tree."""
    if out is None:
        out = []
    if isinstance(result, dict):
        if "confidence" in result and isinstance(result["confidence"], (int, float)):
            conf = float(result["confidence"])
            out.append({"path": path or "<root>", "confidence": conf,
                        "margin": abs(conf - threshold)})
        for key, value in result.items():
            if key == "confidence":
                continue
            collect_margins(value, threshold, f"{path}.{key}" if path else key, out)
    elif isinstance(result, list):
        for index, item in enumerate(result):
            collect_margins(item, threshold, f"{path}[{index}]", out)
    return out


def run_case(model, case):
    """Run one case and package it with its decision margins."""
    schema = build_schema(model, case["schema"])
    result = model.extract(
        case["text"],
        schema,
        threshold=case["threshold"],
        format_results=True,
        include_confidence=True,
        include_spans=True,
    )
    return {
        "name": case["name"],
        "text": case["text"],
        "threshold": case["threshold"],
        "schema": {
            **case["schema"],
            # JSON objects do not carry declaration order; the Swift harness reads it back
            # from here so dictionary-form entities keep Python's prompt order.
            **({"entity_order": list(case["schema"]["entities"].keys())}
               if isinstance(case["schema"].get("entities"), dict) else {}),
        },
        "tags": case.get("tags", []),
        "expected": result,
        "margins": collect_margins(result, case["threshold"]),
    }


def build_borderline_cases(model, base_cases, delta=0.01):
    """Derive cases whose threshold sits `delta` away from a real confidence.

    GLiNER2's span confidences are heavily saturated (≈1.0 or ≈0), so hunting for
    naturally mid-confidence text yields almost nothing. Pinning the *threshold* just
    below and just above an observed confidence is the equivalent — and strictly better
    — boundary test: it asks whether a numerically tiny change flips the decision, which
    is exactly what the Phase 3 reassociation and Phase 4 fp16 gates must not do.
    """
    derived = []
    for case in base_cases:
        if not case["margins"]:
            continue
        # Pivot on the least-confident decision: moving the threshold across it changes
        # that decision without disturbing the more confident ones.
        pivot = min(m["confidence"] for m in case["margins"])
        for sign, kind in ((-1.0, "accept"), (1.0, "reject")):
            threshold = round(pivot + sign * delta, 6)
            if not (0.0 < threshold < 1.0):
                continue
            derived.append(dict(
                name=f"borderline_{kind}_{case['name']}",
                text=case["text"],
                threshold=threshold,
                schema=case["schema"],
                tags=list(case["tags"]),
            ))
    return derived


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--architecture", choices=("span", "boundary"), default="span")
    parser.add_argument("--model", default=None, help="HF id or local checkpoint directory")
    parser.add_argument("--output", default=None, help="Output directory (default per architecture)")
    args = parser.parse_args()

    MODEL_ID = args.model or DEFAULT_MODELS[args.architecture]
    OUTPUT_DIR = Path(args.output) if args.output else default_output_dir(args.architecture, MODEL_ID)

    print(f"Loading {MODEL_ID} ({args.architecture}) ...")
    model = load_model(args.architecture, MODEL_ID)
    if hasattr(model, "eval"):
        model.eval()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cases_out = []

    cases = list(CASES) + (BOUNDARY_ONLY_CASES if args.architecture == "boundary" else [])
    for case in cases:
        record = run_case(model, case)
        cases_out.append(record)
        tight = [m for m in record["margins"] if m["margin"] < 0.05]
        flag = f"  <-- {len(tight)} borderline" if tight else ""
        print(f"  {case['name']:38s} decisions={len(record['margins']):3d}{flag}")

    # Boundary coverage: pin thresholds either side of real confidences.
    print("\nDeriving borderline cases ...")
    borderline_specs = build_borderline_cases(model, cases_out)
    # Keep the corpus a sane size: a spread across task types is enough.
    keep = [spec for spec in borderline_specs
            if spec["name"].split("_", 2)[2] in {
                "ner_basic", "ner_many_labels", "ner_threshold_030", "ner_threshold_070",
                "struct_contact", "struct_multi_instance", "cls_single_positive",
                "cls_multi_label", "rel_basic", "combined_all_tasks",
                "combined_two_structures", "struct_dtype_str_first_not_max",
            }]
    for spec in keep:
        record = run_case(model, spec)
        cases_out.append(record)
        tight = [m for m in record["margins"] if m["margin"] < 0.05]
        print(f"  {spec['name']:48s} thr={spec['threshold']:.4f} "
              f"decisions={len(record['margins']):3d}  borderline={len(tight)}")

    corpus = {
        "model": MODEL_ID,
        "architecture": args.architecture,
        "generator": "scripts/generate_prediction_corpus.py",
        "note": ("Python ground truth. Swift must reproduce `expected` exactly for "
                 "text/start/end/label; confidences within the tier-C tolerance."
                 + (" Boundary (GLiNER 2.5) runtime: structures without a `mode` use the "
                    "legacy (non-record) decoder; `records_*` cases exercise the record "
                    "head; overlap policy 'flat', abstention gate."
                    if args.architecture == "boundary" else "")),
        "cases": cases_out,
    }
    path = OUTPUT_DIR / "corpus.json"
    with open(path, "w") as handle:
        json.dump(corpus, handle, indent=2, ensure_ascii=False, allow_nan=False)

    # ---- summary -------------------------------------------------------------------
    all_margins = [m for case in cases_out for m in case["margins"]]
    by_threshold = {}
    for case in cases_out:
        bucket = by_threshold.setdefault(case["threshold"], [])
        bucket.extend(case["margins"])

    print(f"\nWrote {len(cases_out)} cases to {path}")
    print(f"Total scored decisions: {len(all_margins)}")

    tight_all = [m for m in all_margins if m["margin"] < 0.05]
    very_tight_all = [m for m in all_margins if m["margin"] < 0.02]
    status = "OK " if len(tight_all) >= 5 else "LOW"
    print(f"\nBorderline coverage (plan §0.4 wants >= 5 decisions with margin < 0.05):")
    print(f"  [{status}] {len(tight_all)} < 0.05, {len(very_tight_all)} < 0.02, "
          f"of {len(all_margins)} decisions across {len(cases_out)} cases")
    print("  Natural thresholds (borderline counts are expected to be ~0 here — GLiNER2 "
          "confidences saturate;\n  the derived `borderline_*` cases supply the boundary "
          "coverage):")
    for thr in (0.3, 0.5, 0.7):
        margins = by_threshold.get(thr, [])
        if not margins:
            continue
        tight = [m for m in margins if m["margin"] < 0.05]
        print(f"    threshold={thr}: {len(tight)} < 0.05 of {len(margins)} decisions")

    very_tight = sorted((m for m in all_margins if m["margin"] < 0.02),
                        key=lambda m: m["margin"])
    if very_tight:
        print(f"\nBorderline decisions (< 0.02) — seed the §9 tolerance list with these:")
        for m in very_tight[:20]:
            print(f"    margin={m['margin']:.4f} conf={m['confidence']:.4f} {m['path']}")

    tags = {}
    for case in cases_out:
        for tag in case["tags"]:
            tags[tag] = tags.get(tag, 0) + 1
    print(f"\nExpected-failure tags: {tags or '(none)'}")


if __name__ == "__main__":
    main()
