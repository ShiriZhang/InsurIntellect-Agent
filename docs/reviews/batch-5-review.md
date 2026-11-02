# Batch 5 review — add-bilingual-document-qa

Scope: tasks 5.1–5.6, compared with `d6ed51928a14ca06e5e84ff41cc1d70576f22fb9`.

An independent review sub-agent checked API contracts, exact document/attempt
scope isolation, comparison coverage, citation order, concurrent invalidation,
SSE resource handling and repository conventions against the OpenSpec artifacts.
Its final re-review reported **no remaining issues**.

## Resolved findings

- Restored the existing explicit off-topic refusal boundary in the new scoped
  query entry point, with no public citations for refusals. The old heuristic
  that rejected short English clause questions was not carried forward.
- Narrowed automatic comparison detection: isolated `each`, `comparison`, and
  `which` do not turn ordinary clause or terminology questions into document
  comparisons. Concept explanations such as the difference between a deductible
  and co-payment stay ordinary QA; explicit per-document comparisons still take
  priority. Manual qa/compare remains authoritative.

The same reviewer independently verified both fixes and the concept-explanation
variant before approving the batch.

## Validation

- Final offline regression: **405 passed**.
- `ruff check . --select E9,F63,F7,F82`: passed.
- `openspec validate add-bilingual-document-qa --strict`: passed.
- `git diff --check`: passed.
- Temporary SQLite/PDF tests cover upload status, duplicate and failed retry,
  same-name originals, invalid/oversized files, capacity warning, orphan files,
  mandatory/deduplicated scopes, mixed invalid scopes, pre-stream comparison
  limits, safe retrieval fallback, over 50 out-of-scope distractor chunks,
  per-document comparison/partial missing evidence, shared question vectors,
  context-budget clarification, invalidation before generation, reversed refetch,
  short English clauses, distinct conditions on one page, and identical final
  generation/API/SSE/history citation order.
- An intermediate full rerun encountered one transient existing Chroma HNSW
  segment-reader error in `test_actual_local_chroma_profile_and_directory_isolation`.
  The isolated rerun passed; the final full run is recorded above. No unrelated
  persistence code was changed to mask that failure.

The new query tests use explicit local encoder/generation stand-ins; they do not
establish real embedding or generation quality. Real-model acceptance remains
batch 10. Full answer-language templates, follow-up epochs and deletion orchestration
remain the subsequent planned batches. This batch does not mark them complete.
