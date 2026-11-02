# Batch 4 review — add-bilingual-document-qa

Scope: tasks 4.1–4.4, compared with `f50afe18546a791ba0f1b9d75297b4cbf21e5b11`.

## Standards

An independent sub-agent reviewed repository conventions, lifecycle and concurrency. No documented-standards violations or actionable smell findings remain. Its correctness findings were fixed and independently rechecked.

## Spec and correctness

A separate independent sub-agent checked the implementation against the OpenSpec artifacts. Final remaining findings: **0**. Resolved findings and regressions include:

- Post-commit collection/snapshot retirement failure cannot mark a committed source failed or erase its SQLite chunks. Cleanup is retried separately; valid evidence and unrelated sources remain available.
- Reader leases include the library namespace, so unrelated databases sharing configured directories cannot pin each other's retired versions.
- Migration reserves a SQLite write lock before version/column inspection, preventing concurrent application/CLI v2→v3 duplicate ALTERs.
- Retired-resource and orphan-resource cleanup have separate state. A deterministic interleaving test first reproduced an incorrect cleanup-success result; after the fix ordinary request GC cannot erase an unresolved orphan error. Only an actual successful orphan sweep clears it.

The final reviewers independently rechecked these fixes and reported no remaining blockers.

## Validation

- Full offline regression: **345 passed** (42 additional tests versus batch 3).
- `ruff check . --select E9,F63,F7,F82`: passed.
- `openspec validate add-bilingual-document-qa --strict`: passed.
- `git diff --check`: passed.
- Real local Chroma tests cover profile/directory separation, hard process exit before and after SQLite commit, and an external publisher while an accepted old reader remains alive. Immutable per-publication collections avoid mutating an existing process's cached index.
- Temporary SQLite/synthetic PDF tests cover failures, restoration, completed deletion tombstones, missing keyword resources, exact scoped views, native late results, final-save guards, library isolation and concurrent upgrades.

Embedding/generation model quality and GPU performance are not established by these tests: the test encoders are explicit substitutes, while Chroma persistence is exercised locally. API explicit-scope validation remains task 5; complete history/SSE/deletion orchestration remains tasks 6–7. This batch exposes the corresponding cancellation and transactional write-check hooks and never marks overall source deletion complete.

Final review status: **Standards 0 remaining; Spec 0 remaining**.
