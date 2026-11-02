# Batch 3 review — add-bilingual-document-qa

Scope: tasks 3.1–3.4, compared with `a6d61ec554918ed52dcfd07950668e749991df9a`.

## Standards

An independent sub-agent reviewed the repository conventions and implementation. No actionable standards findings remain.

## Spec and correctness

A separate independent sub-agent checked PDF completeness, source provenance, budgets, identity, queue/recovery, cancellation and CLI parity against the OpenSpec artifacts. Initial review found two defects, both fixed and re-reviewed:

1. Tiled scans could bypass a per-image size threshold and be skipped as blank pages. Quality checks now aggregate the union of unreadable raster rectangles; overlapping decorations count once. Regression tests cover tiled mixed scans, readable headers with unreadable raster/vector bodies, and repeated overlapping logos.
2. An application starting while a CLI held the worker lock would never retry ownership. A standby supervisor now retries the OS lock and automatically takes over after the owner exits. Recovery and processing only run while the lock is held. A regression verifies takeover without restarting the application.

Both reviewers independently ran the PDF, worker and CLI suite: **39 passed**. Final remaining findings: **0** on each review axis.

## Validation

- Full offline regression: **303 passed** (45 additional tests versus batch 2).
- `ruff check . --select E9,F63,F7,F82`: passed.
- `openspec validate add-bilingual-document-qa --strict`: passed.
- `git diff --check`: passed.
- Fault, migration and deletion/cancellation tests use disposable SQLite databases and synthetic PDFs. Real model quality and GPU performance are not established by the injected local encoder tests.

The batch prepares unpublished SQLite candidates (`staged` / `indexing`). Chroma/BM25 publication remains task 4; HTTP upload and full source deletion remain later batches. Neither CLI nor the shared submission service claims that a prepared candidate is ready for questions.
