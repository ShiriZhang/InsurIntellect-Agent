# Batch 6 review — add-bilingual-document-qa

Scope: tasks 6.1–6.5, compared with `0cd4f7e`.

An independent sub-agent reviewed the implementation against the OpenSpec design,
delta specs, and tasks. Final re-review: **no remaining issues**.

## Implementation and resolved review findings

- Resolved answer-language priority and shared bilingual terminal templates, with
  system-controlled English/Chinese headings and source text preserved verbatim.
- Preserved legal insurance questions mentioning sports/weather; advice and
  guarantee boundaries take priority over ambiguous mode detection. Product
  selection wording does not misclassify clause-navigation questions.
- Recognized explicit language instructions including English explanations and
  Chinese instructions to explain or answer in English.
- Serialized session scope epochs in SQLite. A change and return creates a new
  epoch; late results retain their accepted epoch. Follow-ups use prior user
  questions only, preserve comparison intent, and clarify ambiguous multi-topic
  pronouns. Every factual turn retrieves fresh evidence.
- Shared generation prompts treat source text and history as untrusted data;
  assistant answers never supply current facts. Comparison evidence and citation
  order remain unchanged between transports.
- Unified history provenance and terminal saving, including refusal, failures,
  source invalidation and both stream/non-stream cancellation. A cancellation
  event also guards late saves. Deleted-source results cannot restore excerpts.
- English metadata-only covers are filtered after stripping label lines; short
  source clauses following those lines remain valid evidence.

## Validation

- Full offline regression: **451 passed**.
- Final bilingual/session targeted regression, including additional boundary
  cases added after that full run: **51 passed**.
- `ruff check . --select E9,F63,F7,F82`: passed.
- `openspec validate add-bilingual-document-qa --strict`: passed.
- `git diff --check`: passed.
- An earlier full run hit the existing intermittent Chroma HNSW segment-reader
  error in its temporary profile-isolation test. The isolated rerun and the final
  full run both passed. No persistence changes were made to hide this failure.

Tests use temporary libraries, explicit encoder/generator stand-ins, and prompt
inspection. They do not establish real model quality. Real local-model/API
acceptance remains batch 10; deletion orchestration and UI remain batches 7–8.
