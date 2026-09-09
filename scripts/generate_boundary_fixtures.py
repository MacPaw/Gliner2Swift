#!/usr/bin/env python3
# Copyright 2026 MacPaw Way Ltd.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
#
"""Generate GLiNER 2.5 ("boundary" architecture) parity fixtures for the Swift port.

For every case the script runs the Python reference (``gliner2`` 2.0.0,
``AutoExtractor``) on CPU in fp32 with ``torch.manual_seed(0)``, ``model.eval()``
and ``torch.inference_mode()``, and dumps -- WITH the batch dimension kept --

  <case>.safetensors   processor routing tensors, the ``_encode_core`` text/query
                        states, every intermediate of ``BoundaryHead.forward``
                        (captured with ``register_forward_hook`` on the head and
                        its sub-modules), classification logits, explicit-span
                        logits (``score_explicit_spans``) and, for relation
                        schemas, the relation pair batch + ``relation_scorer`` logits
  <case>.json          sidecar: texts, schemas (``Schema.build()``), threshold,
                        word offsets / char mappings, per-sample ``QueryLayout``,
                        per-query thresholds, the ``boundary_head`` settings, and
                        the runtime's ``batch_extract`` results, raw and formatted

Tensor dtypes: floats fp32, integers int32, booleans uint8; every tensor is
``.contiguous()``. Tensors with zero elements are skipped and listed in the
sidecar (``skipped_empty_tensors``).

The head intermediates are the FIRST capture of each hook: the choice-field
decoder re-enters ``boundary_encoder``/``boundary_query_head``/``pair_scorer``
with 1x1 slices afterwards, and those calls are recorded separately as
``runtime_explicit_<k>_*`` (see ``runtime_explicit_calls`` in the sidecar).

Usage (run under the 2.5 reference venv, see scripts/requirements-gliner25.txt):

    /Users/tmwstw/Documents/mnemos/gliner2-ref/.venv/bin/python \
        scripts/generate_boundary_fixtures.py \
        --model fastino/gliner2.5-small-v1 \
        --output Tests/GLiNER2SwiftTests/Fixtures/boundary/small \
        [--cases all|name,...] [--omit-tensors a,b --omit-cases x,y]

``--cases all`` runs every non-multilingual case; the ``multi_*`` cases are
included automatically when the model id contains "multi" and can always be
requested by name.
"""

import argparse
import dataclasses
import json
import os
import platform
import sys
import unicodedata
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
from safetensors.torch import save_file

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "fastino/gliner2.5-small-v1"
DEFAULT_OUTPUT = REPO_ROOT / "Tests" / "GLiNER2SwiftTests" / "Fixtures" / "boundary" / "small"
EXPLICIT_SPAN_COUNT = 16      # pool spans re-scored through score_explicit_spans
CHUNK_SIZE, CHUNK_OVERLAP = 384, 64

# --------------------------------------------------------------------------------------
# Texts
# --------------------------------------------------------------------------------------

# PerfBenchmarkTests.swift `denseText` (the perf plan's "mixed schema" scenario text).
DENSE_TEXT = (
    "Tim Cook is the CEO of Apple in Cupertino. Satya Nadella leads Microsoft in Redmond, "
    "while Sundar Pichai runs Google in Mountain View. Jensen Huang founded NVIDIA in Santa "
    "Clara, and Lisa Su is CEO of AMD in Austin. Elon Musk runs Tesla and SpaceX."
)

DOC_450 = (
    "On Monday morning, the European Commission announced in Brussels that it had reached a "
    "preliminary agreement with the government of Norway on a new framework for cross-border "
    "energy trading. Commission President Ursula von der Leyen told reporters that the deal, "
    "which had been negotiated over eighteen months by teams led by Kadri Simson and Norwegian "
    "minister Terje Aasland, would take effect on January 1, 2027. The announcement came two "
    "days after the International Energy Agency published its annual outlook in Paris, warning "
    "that demand for electricity in Northern Europe could rise by twelve percent before 2030. "
    "Analysts at Goldman Sachs and Morgan Stanley said the agreement would benefit Equinor and "
    "Statkraft, while executives at Siemens Energy in Munich welcomed the news. In Oslo, Prime "
    "Minister Jonas Gahr Støre described the framework as a milestone for the North Sea region. "
    "He was joined at the press conference by Anniken Huitfeldt and by Espen Barth Eide, who had "
    "returned from talks at the United Nations in New York on Friday. Meanwhile, in Washington, "
    "Senator Lisa Murkowski said the United States Department of Energy would study the European "
    "model during hearings scheduled for March 14. Reporters from Reuters, Bloomberg and the "
    "Financial Times noted that the agreement still required approval from the European "
    "Parliament in Strasbourg, where the vote is expected in late April. Opposition to the deal "
    "has been voiced by Marine Le Pen in France and by members of the Alternative für Deutschland "
    "in Berlin, who argue that the framework favours Norwegian exporters. Supporters, including "
    "German Chancellor Friedrich Merz and Dutch Prime Minister Dick Schoof, countered that the "
    "deal would lower prices for consumers in Amsterdam, Hamburg and Copenhagen. The World Bank, "
    "in a statement issued from its headquarters on Tuesday, said that similar arrangements could "
    "be replicated in Southeast Asia, pointing to ongoing discussions between Singapore and "
    "Malaysia. Shares of Ørsted rose four percent on the Copenhagen exchange after the "
    "announcement, and Vestas Wind Systems gained three percent. In London, Ofgem chief "
    "executive Jonathan Brearley said the United Kingdom would open its own consultation on "
    "May 2, noting that National Grid had already begun feasibility studies with TenneT in the "
    "Netherlands. The Norwegian Sovereign Wealth Fund, managed by Nicolai Tangen, declined to "
    "comment. Later in the week, Greenpeace and Friends of the Earth organised demonstrations "
    "outside the Berlaymont building, demanding stronger protections for fisheries in the "
    "Skagerrak. A spokesperson for the Commission, Arianna Podestà, confirmed that an "
    "environmental review led by scientists from the University of Bergen and ETH Zurich would "
    "be completed by September 30, 2026, before any interconnector construction begins near "
    "Stavanger or Aberdeen."
)

DOC_PARA_2 = (
    "The following Thursday, Apple chief executive Tim Cook opened the company's annual developer "
    "conference in Cupertino with a keynote that lasted nearly two hours. Joined on stage by Craig "
    "Federighi and Johny Srouji, Cook unveiled a new generation of laptops built around chips "
    "manufactured by TSMC in Hsinchu and assembled by Foxconn in Zhengzhou. Analysts from Morgan "
    "Stanley and Wedbush Securities, who had travelled from New York for the event, said the "
    "announcement exceeded expectations. Microsoft responded the same afternoon: Satya Nadella "
    "posted from Redmond that Windows would receive a comparable update before the Consumer "
    "Electronics Show in Las Vegas in January. In Mountain View, Sundar Pichai told employees at "
    "Google that Android engineers led by Hiroshi Lockheimer had been working with Samsung in "
    "Suwon on similar features since February 2026. Shares of NVIDIA, whose founder Jensen Huang "
    "had appeared at a rival event in Taipei on Wednesday, slipped two percent, while AMD, led by "
    "Lisa Su in Santa Clara, was largely unchanged. Later that evening, the Wall Street Journal "
    "reported that the Federal Trade Commission had opened an inquiry into app-store fees, citing "
    "complaints filed by Spotify in Stockholm and by Epic Games in Cary, North Carolina. Lina "
    "Khan's successor at the agency, Andrew Ferguson, declined to confirm the report during a "
    "hearing in Washington on Friday, though Senator Amy Klobuchar of Minnesota pressed him "
    "repeatedly. Meanwhile in Europe, Margrethe Vestager's former office at the European "
    "Commission published guidance under the Digital Markets Act that would apply to Meta, Amazon "
    "and ByteDance from July 1. Mark Zuckerberg, speaking from Menlo Park, called the guidance "
    "workable, and Amazon's Andy Jassy said from Seattle that the company would comply. Reporters "
    "from TechCrunch, The Verge and Wired noted that Tesla and SpaceX were absent from the list, a "
    "point Elon Musk celebrated on X from Austin. By Sunday, the Nikkei in Tokyo had reported that "
    "Sony and Nintendo were considering their own joint response, and the Financial Times said "
    "that Arm Holdings in Cambridge would brief investors on Monday, March 30."
)

DOC_PARA_3 = (
    "Across the Atlantic, the weekend belonged to sport. On Saturday afternoon at Wembley Stadium "
    "in London, Arsenal defeated Manchester City two goals to one in the FA Cup final, with Bukayo "
    "Saka scoring the winner in the eighty-seventh minute. Manager Mikel Arteta praised goalkeeper "
    "David Raya, while Pep Guardiola conceded that his side had been second best. Earlier in the "
    "day, in Paris, Novak Djokovic and Carlos Alcaraz advanced to the semi-finals at Roland "
    "Garros, setting up a match on Friday, June 5. In Madrid, Real Madrid confirmed that Kylian "
    "Mbappé would miss the Champions League final in Munich on May 31 after a training injury at "
    "Valdebebas. Across the ocean, the Boston Celtics beat the Denver Nuggets in Game Six at TD "
    "Garden, and Jayson Tatum was named the most valuable player of the series ahead of the finals "
    "against the Oklahoma City Thunder. In baseball, Shohei Ohtani hit his twentieth home run of "
    "the season for the Los Angeles Dodgers at Dodger Stadium, a game attended by Mayor Karen Bass "
    "and by actor Tom Hanks. The Olympic Committee, meeting in Lausanne, announced that Brisbane "
    "would host a test event in October, and IOC President Kirsty Coventry said that talks with "
    "Riyadh over the 2036 Games would continue in Doha. Meanwhile, the Cannes Film Festival closed "
    "on Sunday with the Palme d'Or awarded to a film by Céline Sciamma, presented by jury president "
    "Cate Blanchett. Netflix and Warner Bros. Discovery were reported by Variety to be bidding for "
    "distribution rights, and Amazon MGM Studios in Culver City was said to be preparing an offer. "
    "On Monday morning, the Metropolitan Museum of Art in New York opened an exhibition of works "
    "loaned by the Louvre and by the Uffizi in Florence, with curator Max Hollein thanking donors "
    "including the Ford Foundation and Bloomberg Philanthropies. Attendance in the first week, "
    "according to the New York Times, exceeded one hundred thousand visitors, the strongest "
    "opening since 2019."
)

DOC_1200 = DOC_450 + " " + DOC_PARA_2 + " " + DOC_PARA_3

BATCH3_TEXT_2 = (
    "On March 3, Satya Nadella and Sundar Pichai met in Seattle to discuss a Microsoft and "
    "Google partnership; investors were thrilled."
)
BATCH3_TEXT_3 = (
    "Acme Robotics announced its new support line today. Customers can reach the Boston office "
    "at support@acme-robotics.com or call 555-0142 between nine and five. The company, founded "
    "in 2011 by Dana Whitfield, also opened a second facility in Austin where the hardware team "
    "led by Priya Natarajan builds the next generation of warehouse robots for clients such as "
    "Globex and Initech."
)

# PR #16's description only quotes the FR verification output (personnage: Charles, Emma;
# lieu: Rouen -- Madame Bovary) and no DE sentence, so the multilingual sentences below are
# written to match those entities/labels rather than copied.
MULTI_FR = "Charles Bovary a rencontré Emma à Rouen avant de rejoindre l'hôpital de Yonville."
MULTI_DE = "Angela Merkel traf sich mit Olaf Scholz in Berlin, um über Siemens zu sprechen."
MULTI_DECOMPOSED = unicodedata.normalize(
    "NFD", "Zoé Dupont travaille chez Café René à Montréal."
)


# --------------------------------------------------------------------------------------
# Case definitions (schemas are built with the Schema DSL, one fresh object per sample)
# --------------------------------------------------------------------------------------

def _schema():
    from gliner2.inference.schema import Schema
    return Schema()


def ents(labels):
    return lambda: _schema().entities(labels)


def mixed_schema():
    return (
        _schema()
        .entities(["person", "company", "location"])
        .classification("sentiment", ["positive", "negative"])
        .structure("product")
        .field("name", dtype="str")
        .field("price", dtype="str")
        .field("category", choices=["hardware", "software", "vehicle"])
    )


def batch3_schema_2():
    return (
        _schema()
        .entities(["person", "company", "location", "date"])
        .classification("sentiment", ["positive", "negative", "neutral"])
    )


def batch3_schema_3():
    return (
        _schema()
        .entities(["organization"])
        .structure("contact")
        .field("email")
        .field("phone")
    )


def classification_only_schema():
    return (
        _schema()
        .classification("sentiment", ["positive", "negative", "neutral"], multi_label=False)
        .classification("topics", ["film", "food", "sports"], multi_label=True, cls_threshold=0.3)
    )


def relations_schema():
    return (
        _schema()
        .entities(["person", "company", "city"])
        .relations(["founded", "located_in"])
    )


def structure_legacy_schema():
    # mode=None keeps the legacy aggregate decoder (no record_metadata emitted).
    return (
        _schema()
        .structure("person", mode=None)
        .field("name")
        .field("age")
        .field("role", choices=["engineer", "manager", "designer"])
    )


def case_specs() -> "OrderedDict[str, Dict[str, Any]]":
    cases = OrderedDict()

    def add(name, texts, schemas, threshold=0.5, kind="standard", multi_only=False, **extra):
        if callable(schemas):
            schemas = [schemas] * len(texts)
        cases[name] = dict(texts=list(texts), schemas=list(schemas), threshold=threshold,
                           kind=kind, multi_only=multi_only, **extra)

    add("one_word", ["Paris"], ents(["city"]))
    add("five_words", ["Tim Cook visited Paris today"], ents(["person", "city"]))
    add("mixed_schema", [DENSE_TEXT], mixed_schema)
    add("long_entity",
        ["Maria Lopez joined the International Committee for the Advancement of Renewable "
         "Energy Research and Development in Southern Europe last spring."],
        ents(["organization", "person"]))
    add("abstain", ["The weather was pleasant and the streets were quiet."],
        ents(["person", "company", "date"]))
    # On gliner2.5-small the case above never crosses the abstention gate (null probs
    # 0.24/0.12/0.46); this one does: sigmoid(null_logit) = 0.91 for "city".
    add("abstain_gate", ["The recipe calls for two eggs, some flour and a pinch of salt."],
        ents(["person", "company", "city"]))
    add("overlap_two_labels", ["New York City Council met in New York on Monday."],
        ents(["city", "organization", "location"]))
    add("doc_450", [DOC_450], ents(["person", "location", "organization", "date"]))
    add("doc_1200_chunked", [DOC_1200], ents(["person", "location", "organization"]),
        kind="long", entity_types=["person", "location", "organization"])
    add("inner_whitespace", ["Contact  Jane\tDoe at Acme  Corp."], ents(["person", "company"]))
    add("batch3",
        ["Tim Cook visited Paris today", BATCH3_TEXT_2, BATCH3_TEXT_3],
        [ents(["person", "city"]), batch3_schema_2, batch3_schema_3])
    add("classification_only", ["I loved this movie, absolutely brilliant!"],
        classification_only_schema)
    add("relations", ["Steve Jobs founded Apple in Cupertino."], relations_schema)
    add("structure_legacy", ["Alice Smith, 34, is the lead engineer on the project."],
        structure_legacy_schema)
    add("multi_fr", [MULTI_FR], ents(["personnage", "lieu"]), multi_only=True)
    add("multi_de", [MULTI_DE], ents(["person", "ort", "organisation"]), multi_only=True)
    add("multi_decomposed", [MULTI_DECOMPOSED], ents(["personne", "entreprise", "lieu"]),
        multi_only=True)
    return cases


# --------------------------------------------------------------------------------------
# Hook-based capture of the boundary head
# --------------------------------------------------------------------------------------

class HeadCapture:
    """Forward hooks on ``boundary_head`` + sub-modules, ``relation_scorer`` and a
    wrapper around ``_encode_core``. The first call of every hook is kept (the
    runtime's main forward); later calls (choice-field re-scoring) go to
    ``explicit_calls``; every ``relation_scorer`` call goes to ``relation_calls``."""

    def __init__(self, model):
        self.model = model
        self.first: Dict[str, Any] = {}
        self.counts: Dict[str, int] = {}
        self.explicit_calls: List[Dict[str, Any]] = []
        self.relation_calls: List[Dict[str, Any]] = []
        self.batch = None
        self.core = None
        self._handles = []

    def _record(self, name, value):
        self.counts[name] = self.counts.get(name, 0) + 1
        if name not in self.first:
            self.first[name] = value

    def _hook(self, name):
        return lambda module, inputs, output: self._record(name, output)

    def _explicit_hook(self, module, inputs, output):
        # SparseBoundaryPairScorer.forward(boundary_states, query_states, proposals, ...)
        proposals = inputs[2]
        self.counts["pair_scorer"] = self.counts.get("pair_scorer", 0) + 1
        self.explicit_calls.append(dict(
            query_states=inputs[1].detach().clone(),
            indices=proposals.indices.detach().clone(),
            valid=proposals.valid_mask.detach().clone(),
            compat=(proposals.compat_logits.detach().clone()
                    if proposals.compat_logits is not None else None),
            logits=output.detach().clone(),
        ))

    def _relation_hook(self, module, inputs, output):
        # SparseRelationScorer.forward(text_states_i, relation_query_states, candidates, pairs)
        self.counts["relation_scorer"] = self.counts.get("relation_scorer", 0) + 1
        self.relation_calls.append(dict(
            query_states=inputs[1].detach().clone(),
            pairs=inputs[3],
            logits=output.detach().clone(),
        ))

    def __enter__(self):
        head = self.model.boundary_head
        for name, module in (
            ("boundary_encoder", head.boundary_encoder),
            ("boundary_query_head", head.boundary_query_head),
            ("shared_pool_builder", head.shared_pool_builder),
            ("shared_pool_scorer", head.shared_pool_scorer),
            ("boundary_head", head),
        ):
            self._handles.append(module.register_forward_hook(self._hook(name)))
        self._handles.append(head.pair_scorer.register_forward_hook(self._explicit_hook))
        if getattr(self.model, "relation_scorer", None) is not None:
            self._handles.append(
                self.model.relation_scorer.register_forward_hook(self._relation_hook)
            )

        original = self.model._encode_core

        def wrapped_encode_core(batch):
            core = original(batch)
            self.counts["_encode_core"] = self.counts.get("_encode_core", 0) + 1
            if self.batch is None:
                self.batch, self.core = batch, core
            return core

        self.model._encode_core = wrapped_encode_core   # instance attribute shadows the method
        return self

    def __exit__(self, *exc):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        if "_encode_core" in self.model.__dict__:
            del self.model._encode_core
        return False


# --------------------------------------------------------------------------------------
# Serialisation helpers
# --------------------------------------------------------------------------------------

def to_saveable(t: torch.Tensor) -> torch.Tensor:
    t = t.detach().cpu()
    if t.dtype == torch.bool:
        t = t.to(torch.uint8)
    elif t.is_floating_point():
        t = t.to(torch.float32)
    elif t.dtype != torch.int32:
        t = t.to(torch.int32)
    # clone: views such as pair_logits = scorer_score.transpose(1, 2) would otherwise share
    # storage, which safetensors refuses to serialise.
    return t.contiguous().clone()


def json_default(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    if dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)
    return str(obj)


def sanitize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """The runtime metadata dict, minus RegexValidator objects (repr'd instead)."""
    out = {}
    for key, value in metadata.items():
        if key in ("field_metadata", "entity_metadata"):
            cleaned = {}
            for name, entry in value.items():
                entry = dict(entry)
                if entry.get("validators"):
                    entry["validators"] = [repr(v) for v in entry["validators"]]
                cleaned[name] = entry
            out[key] = cleaned
        else:
            out[key] = value
    return out


def versions() -> Dict[str, str]:
    import gliner2, safetensors, transformers
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "gliner2": gliner2.__version__,
        "safetensors": safetensors.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
    }


def resolve_model_files(model_id: str):
    """Return (snapshot_dir, revision, config_dict, encoder_config_dict)."""
    if os.path.isdir(model_id):
        snapshot, revision = Path(model_id), None
    else:
        from huggingface_hub import snapshot_download
        try:
            snapshot = Path(snapshot_download(model_id, local_files_only=True))
        except Exception:
            snapshot = Path(snapshot_download(model_id))
        revision = snapshot.name
    config = json.load(open(snapshot / "config.json"))
    encoder_config_path = snapshot / "encoder_config" / "config.json"
    encoder_config = json.load(open(encoder_config_path)) if encoder_config_path.exists() else None
    return snapshot, revision, config, encoder_config


# --------------------------------------------------------------------------------------
# Case runners
# --------------------------------------------------------------------------------------

def build_schemas(case):
    return [builder() for builder in case["schemas"]]


def run_standard_case(model, name: str, case: Dict[str, Any], env: Dict[str, Any]):
    texts, threshold = case["texts"], case["threshold"]
    schemas = build_schemas(case)
    settings = model.boundary_settings
    device = next(model.parameters()).device

    # 1. The runtime's own raw pass, with hooks attached.
    with HeadCapture(model) as cap:
        expected_raw = model.batch_extract(
            texts, schemas, threshold=threshold, format_results=False,
            include_confidence=True, include_spans=True,
        )
    batch, core = cap.batch, cap.core
    if batch is None:
        raise RuntimeError(f"{name}: _encode_core was never called")

    # 2. Formatted results (a second, hook-free pass; eval + fp32 CPU is deterministic).
    expected_formatted = model.batch_extract(
        texts, schemas, threshold=threshold, format_results=True,
        include_confidence=True, include_spans=True,
    )
    expected_plain = model.batch_extract(texts, schemas, threshold=threshold)

    schema_dicts, metadata_list = model._build_schema_dicts_and_metadata(schemas)
    B = len(batch)
    Q = core["query_states"].shape[1]

    tensors: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    notes: Dict[str, str] = {}

    # ---- processor routing tensors ---------------------------------------------------
    tensors["input_ids"] = batch.input_ids
    tensors["attention_mask"] = batch.attention_mask
    tensors["text_word_indices"] = batch.text_word_indices
    tensors["text_word_mask"] = batch.text_word_mask
    tensors["query_marker_indices"] = batch.query_marker_indices
    tensors["query_marker_mask"] = batch.query_marker_mask
    tensors["query_group_index"] = batch.query_group_index
    tensors["cls_marker_indices"] = batch.cls_marker_indices
    tensors["cls_marker_mask"] = batch.cls_marker_mask
    tensors["cls_group_index"] = batch.cls_group_index

    # ---- _encode_core -------------------------------------------------------------------
    tensors["text_states"] = core["text_states"]
    tensors["query_states"] = core["query_states"]
    tensors["text_lengths"] = core["text_lengths"]
    notes["text_states"] = "_encode_core: last_hidden_state gathered at text_word_indices, zeroed by text_word_mask"
    notes["query_states"] = "_encode_core: last_hidden_state gathered at query_marker_indices, zeroed by query_marker_mask"

    # ---- boundary head intermediates (first hook capture = runtime main forward) -------
    enc = cap.first.get("boundary_encoder")
    if enc is not None:
        tensors["boundary_states"] = enc.states
        tensors["boundary_mask"] = enc.mask
    marg = cap.first.get("boundary_query_head")
    if marg is not None:
        tensors["start_logits"] = marg.start_logits
        tensors["end_logits"] = marg.end_logits
        tensors["inside_logits"] = marg.inside_logits
        tensors["inside_prefix"] = marg.inside_prefix
        tensors["inside_prefix_mean"] = marg.inside_prefix_mean
    pooled = cap.first.get("shared_pool_builder")
    if pooled is not None:
        tensors["pool_indices"] = pooled.indices
        tensors["pool_mask"] = pooled.mask
        tensors["pool_proposal_logits"] = pooled.proposal_logits
        tensors["pool_compat_logits"] = pooled.compat_logits
        notes["pool_indices"] = "DocumentCandidatePool output [B,P,2] (start,end) half-open word boundaries; (0,0) where pool_mask is 0"
    scored = cap.first.get("shared_pool_scorer")
    if scored is not None:
        tensors["scorer_score"] = scored[0]
        tensors["scorer_candidate"] = scored[1]
        notes["scorer_score"] = "SharedPoolScorer score [B,P,Q] (candidate-major); pair_logits = transpose(1,2)"
    head_out = cap.first.get("boundary_head")
    pair_logits = None
    if head_out is not None and head_out.candidates is not None:
        cands = head_out.candidates
        pair_logits = cands.pair_logits
        tensors["pair_logits"] = pair_logits
        if head_out.null_logits is not None:
            tensors["null_logits"] = head_out.null_logits
        if head_out.count_log_rates is not None:
            tensors["count_log_rates"] = head_out.count_log_rates
        if cands.candidate_states is not None:
            tensors["candidate_states"] = cands.candidate_states[:, 0]
            notes["candidate_states"] = ("candidate_encoder output masked to 0 where pool_mask is 0, "
                                         "stored as [B,P,H]; the runtime's CandidateTensorBatch."
                                         "candidate_states is this expanded to [B,Q,P,H]")
        tensors["probs"] = torch.sigmoid(pair_logits / settings.pair_temperature)
        notes["probs"] = f"sigmoid(pair_logits / pair_temperature={settings.pair_temperature})"

    # ---- per-query thresholds ------------------------------------------------------------
    query_thresholds = model._query_thresholds(core["ext_specs"], metadata_list, threshold, device)

    # ---- classification logits -----------------------------------------------------------
    cls_entries = []
    for i in range(B):
        for cls in core["cls_specs"][i]:
            logits = model.classifier(cls["group_embs"][1:]).squeeze(-1)
            key = (f"cls_logits_{cls['task_name']}" if B == 1
                   else f"cls_logits_s{i}_{cls['task_name']}")
            tensors[key] = logits
            labels = next((c["labels"] for c in schema_dicts[i].get("classifications", [])
                           if c["task"] == cls["task_name"]), None)
            cls_entries.append(dict(key=key, sample=i, group_index=cls["group_index"],
                                    task_name=cls["task_name"], labels=labels,
                                    schema_tokens=cls["schema_tokens"]))
    if cls_entries:
        notes["cls_logits_*"] = ("classifier(group_embs[1:]).squeeze(-1) over the [L] marker states; "
                                 "probs = sigmoid (multi_label) or softmax, / classification_temperature")

    # ---- explicit spans through score_explicit_spans (M2) --------------------------------
    explicit_info = None
    if Q > 0 and pooled is not None:
        from gliner2.models.base import EncodedBatch
        C = EXPLICIT_SPAN_COUNT
        idx = torch.zeros((B, C, 2), dtype=torch.long)
        valid = torch.zeros((B, C), dtype=torch.bool)
        for i in range(B):
            keep = pooled.mask[i].nonzero(as_tuple=True)[0][:C]
            idx[i, : len(keep)] = pooled.indices[i, keep]
            valid[i, : len(keep)] = True
        indices = idx.unsqueeze(1).expand(B, Q, C, 2).contiguous()
        valid_mask = (valid.unsqueeze(1).expand(B, Q, C) & core["query_mask"].unsqueeze(-1)).contiguous()
        encoded = EncodedBatch(
            text_states=core["text_states"], text_mask=core["text_mask"],
            text_lengths=core["text_lengths"], query_states=core["query_states"],
            query_mask=core["query_mask"], query_layouts=tuple(batch.query_layouts),
        )
        explicit_logits = model.score_explicit_spans(encoded, indices, valid_mask)
        tensors["explicit_indices"] = indices
        tensors["explicit_mask"] = valid_mask
        tensors["explicit_logits"] = explicit_logits
        explicit_info = dict(
            count=C,
            note=("the pool's first min(16,P) valid spans (pool order) broadcast over Q, scored with "
                  "model.score_explicit_spans(EncodedBatch from _encode_core, indices, valid_mask); "
                  "logits are masked to -1e4 where explicit_mask is 0 or the span is illegal"),
        )

    # ---- runtime's own score_explicit_spans calls (choice fields) -------------------------
    runtime_explicit = []
    for k, call in enumerate(cap.explicit_calls):
        located = None
        qs = call["query_states"]
        if qs.shape[0] == 1 and qs.shape[1] == 1:
            for i in range(B):
                for q in range(Q):
                    if torch.equal(core["query_states"][i, q], qs[0, 0]):
                        located = (i, q)
                        break
                if located:
                    break
        prefix = f"runtime_explicit_{k}"
        tensors[f"{prefix}_indices"] = call["indices"]
        tensors[f"{prefix}_valid"] = call["valid"]
        if call["compat"] is not None:
            tensors[f"{prefix}_compat"] = call["compat"]
        tensors[f"{prefix}_logits"] = call["logits"]
        runtime_explicit.append(dict(
            tensor_prefix=prefix, call_index=k,
            sample=located[0] if located else None,
            query_id=located[1] if located else None,
            field=(core["ext_specs"][located[0]][located[1]] if located else None),
            note="pair_scorer forward hook during the runtime's choice-field decode "
                 "(_decode_choice_field -> BoundaryHead.score_explicit_spans on a 1x1 slice)",
        ))

    # ---- relations --------------------------------------------------------------------------
    relation_info = []
    for k, call in enumerate(cap.relation_calls):
        prefix = "relation" if len(cap.relation_calls) == 1 else f"relation_s{k}"
        pairs = call["pairs"]
        tensors[f"{prefix}_query_states"] = call["query_states"]
        for field_name in ("batch_index", "relation_index", "head_start", "head_end",
                           "tail_start", "tail_end", "head_prob", "tail_prob"):
            tensors[f"{prefix}_pairs_{field_name}"] = getattr(pairs, field_name)
        if pairs.pair_mask is not None:
            tensors[f"{prefix}_pairs_pair_mask"] = pairs.pair_mask
        tensors[f"{prefix}_logits"] = call["logits"]
        tensors[f"{prefix}_probs"] = torch.sigmoid(call["logits"] / settings.relation_temperature)
        relation_info.append(dict(
            tensor_prefix=prefix, call_index=k,
            relation_types=list(pairs.relation_types),
            head_keys=[list(x) for x in pairs.head_keys],
            tail_keys=[list(x) for x in pairs.tail_keys],
            note=("relation_scorer forward hook: inputs (text_states[i:i+1], relation query states "
                  "[1,R,2H] = cat(head,tail) query states, single-sample candidates, RelationPairBatch "
                  "from relation_pair_generator.generate); logits [P]; probs = sigmoid(logits / "
                  f"relation_temperature={settings.relation_temperature})"),
        ))

    # ---- write ------------------------------------------------------------------------------
    omit = set(env["omit_tensors"]) if name in env["omit_cases"] else set()
    saved: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    skipped_empty, omitted = [], []
    for key, value in tensors.items():
        if value is None:
            continue
        if key in omit:
            omitted.append(key)
            continue
        t = to_saveable(value)
        if t.numel() == 0:
            skipped_empty.append(dict(name=key, shape=list(t.shape)))
            continue
        saved[key] = t

    out_dir = env["output"]
    save_file(saved, str(out_dir / f"{name}.safetensors"), metadata={"case": name, "model": env["model"]})

    rel_specs_json = [
        [dict(group_index=r["group_index"], relation_type=r["relation_type"],
              head_query_ids=list(r["spec"].head_query_ids),
              tail_query_ids=list(r["spec"].tail_query_ids),
              allow_self=r["spec"].allow_self)
         for r in core["rel_specs"][i]]
        for i in range(B)
    ]
    cls_specs_json = [
        [dict(group_index=c["group_index"], task_name=c["task_name"], schema_tokens=c["schema_tokens"])
         for c in core["cls_specs"][i]]
        for i in range(B)
    ]
    layouts_json = [
        dict(queries=[dataclasses.asdict(q) for q in layout.queries],
             classification_query_ids=list(layout.classification_query_ids),
             extractive_query_ids=list(layout.extractive_query_ids))
        for layout in batch.query_layouts
    ]
    built = [s.build() for s in schemas]

    sidecar = OrderedDict(
        case=name,
        kind="standard",
        model=env["model"],
        revision=env["revision"],
        generator="scripts/generate_boundary_fixtures.py",
        versions=env["versions"],
        seed=0,
        dtype="float32",
        device="cpu",
        batch_size=B,
        text=texts[0] if B == 1 else None,
        texts=texts,
        original_texts=list(batch.original_texts),
        model_texts=list(getattr(batch, "model_texts", ()) or ()),
        schema=built[0] if B == 1 else built,
        schemas=built,
        schema_dicts_passed_to_collator=list(batch.original_schemas),
        threshold=threshold,
        word_offsets=list(core["word_offsets"]),
        text_word_counts=list(batch.text_word_counts),
        text_lengths=core["text_lengths"].tolist(),
        start_mappings=batch.start_mappings,
        end_mappings=batch.end_mappings,
        text_tokens=batch.text_tokens,
        schema_tokens=batch.schema_tokens_list,
        task_types=batch.task_types,
        schema_special_indices=batch.schema_special_indices,
        query_layout=[layout["queries"] for layout in layouts_json],
        query_layouts=layouts_json,
        ext_specs=core["ext_specs"],
        cls_specs=cls_specs_json,
        rel_specs=rel_specs_json,
        has_record_specs=any(bool(r) for r in (getattr(batch, "record_specs", ()) or ())),
        query_thresholds=query_thresholds.tolist(),
        metadata=[sanitize_metadata(m) for m in metadata_list],
        boundary_head_settings=dataclasses.asdict(model.boundary_settings),
        config=env["config"],
        encoder_config=env["encoder_config"],
        classification_logits=cls_entries,
        explicit_spans=explicit_info,
        runtime_explicit_calls=runtime_explicit,
        relations=relation_info,
        tensors={k: list(v.shape) for k, v in saved.items()},
        tensor_dtypes={k: str(v.dtype).replace("torch.", "") for k, v in saved.items()},
        tensor_notes=notes,
        omitted_tensors=omitted,
        skipped_empty_tensors=skipped_empty,
        hook_call_counts=cap.counts,
        expected_raw=expected_raw,
        expected_formatted=expected_formatted,
        expected_plain=expected_plain,
    )
    with open(out_dir / f"{name}.json", "w") as handle:
        json.dump(sidecar, handle, indent=2, ensure_ascii=False, default=json_default)

    size = (out_dir / f"{name}.safetensors").stat().st_size
    print(f"  {name:22s} B={B} Q={Q} L={core['text_states'].shape[1]:4d} "
          f"tensors={len(saved):3d} ({size / 1024:.0f} KB) formatted={json.dumps(expected_plain, ensure_ascii=False)[:100]}")


def run_long_case(model, name: str, case: Dict[str, Any], env: Dict[str, Any]):
    """Sidecar-only: extract_entities_long + the chunk layout the runtime used."""
    from gliner2.inference.chunking import split_text_into_chunks
    from gliner2.processing.word_splitter import word_splitter_from

    text, threshold = case["texts"][0], case["threshold"]
    entity_types = case["entity_types"]
    chunks = split_text_into_chunks(text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
                                    word_splitter=word_splitter_from(model))
    expected_formatted = model.extract_entities_long(
        text, entity_types, threshold=threshold, chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP, include_confidence=True, include_spans=True,
    )
    expected_plain = model.extract_entities_long(
        text, entity_types, threshold=threshold, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
    )
    schema = build_schemas(case)[0]
    chunk_results = model.batch_extract(
        [c.text for c in chunks], schema, threshold=threshold, format_results=True,
        include_confidence=True, include_spans=True, max_len=CHUNK_SIZE,
    )
    word_count = len(list(word_splitter_from(model)(text, lower=False)))
    sidecar = OrderedDict(
        case=name,
        kind="long",
        model=env["model"],
        revision=env["revision"],
        generator="scripts/generate_boundary_fixtures.py",
        versions=env["versions"],
        text=text,
        word_count=word_count,
        entity_types=entity_types,
        schema=schema.build(),
        threshold=threshold,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        chunks=[dataclasses.asdict(c) for c in chunks],
        api="extract_entities_long(text, entity_types, threshold, chunk_size=384, chunk_overlap=64, "
            "include_confidence=True, include_spans=True) = batch_extract_long -> "
            "split_text_into_chunks + batch_extract(max_len=384) + merge_chunk_results",
        expected_chunk_results=chunk_results,
        expected_formatted=expected_formatted,
        expected_plain=expected_plain,
        boundary_head_settings=dataclasses.asdict(model.boundary_settings),
    )
    with open(env["output"] / f"{name}.json", "w") as handle:
        json.dump(sidecar, handle, indent=2, ensure_ascii=False, default=json_default)
    print(f"  {name:22s} words={word_count} chunks={len(chunks)} "
          f"{[(c.start_word, c.end_word) for c in chunks]} "
          f"entities={ {k: len(v) for k, v in expected_plain.get('entities', {}).items()} }")


# --------------------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate GLiNER 2.5 boundary-architecture parity fixtures.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HF id or local checkpoint dir")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Fixture directory")
    parser.add_argument("--cases", default="all", help="'all' or a comma-separated list of case names")
    parser.add_argument("--omit-tensors", default="",
                        help="Comma-separated tensor names to leave out of the cases in --omit-cases")
    parser.add_argument("--omit-cases", default="", help="Cases the --omit-tensors rule applies to")
    parser.add_argument("--list", action="store_true", help="List case names and exit")
    args = parser.parse_args()

    cases = case_specs()
    if args.list:
        for name, case in cases.items():
            print(f"{name:22s} kind={case['kind']:8s} texts={len(case['texts'])}"
                  f"{'  (multi only)' if case['multi_only'] else ''}")
        return

    is_multi = "multi" in os.path.basename(args.model.rstrip("/")).lower()
    if args.cases == "all":
        selected = [n for n, c in cases.items() if not c["multi_only"] or is_multi]
    else:
        selected = [n.strip() for n in args.cases.split(",") if n.strip()]
        unknown = [n for n in selected if n not in cases]
        if unknown:
            parser.error(f"unknown case(s): {unknown}; known: {list(cases)}")

    torch.manual_seed(0)
    torch.set_grad_enabled(False)

    from gliner2 import AutoExtractor

    print(f"Loading {args.model} ...")
    model = AutoExtractor.from_pretrained(args.model)
    model.eval()
    model = model.to("cpu").float()
    assert getattr(model, "architecture", None) == "boundary", "not a boundary checkpoint"

    snapshot, revision, config, encoder_config = resolve_model_files(args.model)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    env = dict(
        model=args.model, revision=revision, snapshot=str(snapshot), config=config,
        encoder_config=encoder_config, versions=versions(), output=output,
        omit_tensors=[t for t in args.omit_tensors.split(",") if t],
        omit_cases=set(c for c in args.omit_cases.split(",") if c),
    )
    print(f"  revision={revision} hidden={encoder_config.get('hidden_size') if encoder_config else '?'} "
          f"candidate_pool={model.boundary_settings.candidate_pool} "
          f"pool_size={model.boundary_settings.pool_size}")
    print(f"Writing {len(selected)} case(s) to {output}")

    manifest_cases = OrderedDict()
    with torch.inference_mode():
        for name in selected:
            case = cases[name]
            if case["kind"] == "long":
                run_long_case(model, name, case, env)
            else:
                run_standard_case(model, name, case, env)
            files = OrderedDict()
            for suffix in (".safetensors", ".json"):
                path = output / f"{name}{suffix}"
                if path.exists():
                    files[path.name] = path.stat().st_size
            manifest_cases[name] = dict(kind=case["kind"], files=files)

    manifest_path = output / "manifest.json"
    if manifest_path.exists():   # merge: partial re-runs (--cases x) keep the other cases
        previous = json.load(open(manifest_path), object_pairs_hook=OrderedDict)
        merged = OrderedDict(previous.get("cases", {}))
        merged.update(manifest_cases)
        manifest_cases = merged
    manifest = OrderedDict(
        model=args.model, revision=revision, generator="scripts/generate_boundary_fixtures.py",
        versions=env["versions"], cases=manifest_cases,
    )
    with open(output / "manifest.json", "w") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    total = sum(size for c in manifest_cases.values() for size in c["files"].values())
    print(f"Done: {len(manifest_cases)} cases, {total / (1024 * 1024):.1f} MB in {output}")


if __name__ == "__main__":
    main()
