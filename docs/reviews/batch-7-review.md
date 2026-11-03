# Batch 7 review — add-bilingual-document-qa

Scope: tasks 7.1–7.4, compared with `313db7e24e43336bee30826c3266187e2df146ac`.

An independent sub-agent reviewed API behavior, persistent tombstones,
cancellation and late native/API results, source/history erasure, process-reader
barriers, index garbage collection, and startup/rollback recovery against the
OpenSpec artifacts. Final re-review: **no remaining issues**.

## Resolved review finding

Deletion initially cleared historical source excerpts only in a later worker
stage. A blocked native reader or snapshot cleanup could leave those excerpts
readable while the document was already deleting. `begin` now scrubs affected
history and invalidates session epochs in the same SQLite transaction as the
durable tombstone. A regression reads history immediately after DELETE, before
any worker cleanup, and verifies that no deleted-source content/metadata remains.

The reviewer also confirmed the final original-path safeguard: if the registered
original does not belong to the configured identity directory, deletion remains
incomplete and never reports successful erasure or removes an external file.

## Validation

- Final full offline regression: **472 passed**.
- Earlier deletion/API regressions: 14 passed; expanded deletion regressions:
  15 passed; combined deletion/bilingual/scope regressions: 117 passed.
- `ruff check . --select E9,F63,F7,F82`: passed.
- `openspec validate add-bilingual-document-qa --strict`: passed.
- `git diff --check`: passed.

Temporary-library tests cover idempotent 202/200/404/410 behavior, immediate scope
and history invalidation, deletion during blocked ordinary/SSE comparison, upstream
closure, unrelated B requests, late parser and encoder results, failure after every
cleanup stage, persistent restart recovery, same-content new identity, old pointer/
snapshot/vector restoration, owned temporary files, cross-process reader markers,
and external-path protection. Cleanup does not load an encoder or call generation.

The tests use explicit encoder/generator substitutes and local temporary resources;
they do not establish real model quality or complete the batch 10 acceptance run.
Deletion UI remains batch 8. Retained answer bodies and external/user-owned backups
are outside managed source erasure, as specified by ADR-0003.
