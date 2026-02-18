# Contribution Guidelines

Make your Pull Requests clear and obvious to anyone viewing them.
Set `main` as your target branch.

Use **Conventional Commits** principles in naming PRs and branches:

- `Feat: ...` for new features and new functionality implementations.
- `Bug: ...` for bug fixes.
- `Fix: ...` for minor issues fixing, like typos or inaccuracies in code.
- `Chore: ...` for boring stuff like code polishing, refactoring, deprecation fixing etc.

**PR naming example:** `Feat: Add Threads API handling` or `Bug: Fix message result duplication`

**Branch naming example:** `feat/add-threads-API-handling` or `bug/fix-message-result-duplication`

Write description to pull requests in following format:

> **What**
>
> ...
>
> **Why**
>
> ...
>
> **Affected Areas**
>
> ...
>
> **More Info**
>
> ...

We'll appreciate you including tests to your code if it is needed and possible.

## General Guidelines

- Keep changes focused — one feature or fix per PR.
- Follow existing code style and conventions in the project.
- Document public API changes (update README if applicable).
- Make sure the project builds cleanly before submitting (`swift build`).
- If your change affects inference output, include before/after examples in the PR description.

## Parity with the Python Implementation

GLiNER2Swift is a direct port of the [Python GLiNER2](https://github.com/fastino-ai/gliner2) repository. Maintaining numerical parity with the reference implementation is critical.

**If you use AI code generation tools** (Copilot, Claude, ChatGPT, etc.) to contribute to this repo, you **must** verify your changes against the Python implementation by running parity tests:

1. Set up the Python GLiNER2 environment following its README.
2. Run the same input through both the Python and Swift implementations.
3. Compare outputs — entity spans, labels, scores, and classification results should match within floating-point tolerance.
4. Include parity test results (or a summary) in your PR description.

AI-generated code can introduce subtle numerical divergences (e.g., different attention mask handling, softmax precision, tensor layout assumptions). Manual verification against the Python repo is the only way to catch these issues.

## Currently Supported Scope

Before contributing a new feature, check the [Work in Progress](README.md#work-in-progress) section in the README. The following areas are not yet implemented and are good candidates for contribution:

- Training loop
- Relation extraction
- LoRA adapters
- Additional GLiNER model variants (currently only `deberta-v3-base`)

If you'd like to tackle one of these, open an issue first to discuss the approach.