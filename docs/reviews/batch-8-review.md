# Batch 8 review — add-bilingual-document-qa

Scope: tasks 8.1–8.4, compared with `811d804`.

An independent sub-agent reviewed the browser implementation, scope preparation
API, documentation and regression tests against the OpenSpec artifacts. Final
re-review: **no remaining issues**.

## Resolved findings and regressions

- Bound every browser ask to the server epoch returned by its own frozen scope
  preparation. A delayed HTTP request now stays in its historical epoch and cannot
  populate a newly selected A after A → B → A. Invalid/mismatched epochs return 422.
- Rechecked PDF render identity after asynchronous destruction of an old document;
  obsolete loads cannot restore a deleted source, roll back render tokens or reopen
  the fallback viewer. Repeated citation opens cancel earlier canvas rendering.
- Added document IDs to citation cards, evidence, viewer titles and PDF options,
  making same-name/same-page sources distinguishable.
- Persisted normalized selection even when memory was already empty, and explicitly
  synchronized empty clears. A temporarily unready saved ID cannot silently return.
- Invalidated the library render cache on selection changes so an immediate clear
  cannot leave a visually checked box inconsistent with the submitted scope.
- Refreshed PDF options from current live IDs while preserving the independently
  previewed document; deleting A does not leave A selectable while B remains open.

The reviewer independently confirmed these fixes. Its restricted-environment pytest
attempt blocked on local sockets and was stopped; that attempt is not counted as
validation. The passing browser runs below were executed by the primary agent.

## Validation

- Final full offline regression: **490 passed**, including 12 browser interactions and 6 selection/epoch API cases.
- An earlier complete run passed 486 tests before review fixes.
- Browser interactions run in actual Chromium/Edge against an isolated temporary
  FastAPI/SQLite library. Encoder/generator substitutes are explicit in regression
  fixtures and do not establish real model quality.
- Independent real-service operation: [batch-8-operation.md](batch-8-operation.md).
  Public synthetic PDF, real local BGE-M3 on CPU, actual generation API, cited
  answer, rendered original page, then deletion with a 410 original URL and scrubbed
  old excerpt while retaining the answer. [Screenshot](../screenshots/document-library.png).
- `node --check static/js/app.js`: passed.
- `ruff check . --select E9,F63,F7,F82`: passed.
- `openspec validate add-bilingual-document-qa --strict`: passed.
- `git diff --check`: passed.

The real-service walkthrough is a single operation check, not the independent
real-document quality/capacity acceptance required by batch 10. User guides describe
local cache/device behavior, API data boundaries, explicit selection, comparison,
follow-up reset, scan rejection, deletion and retained-answer boundaries.
