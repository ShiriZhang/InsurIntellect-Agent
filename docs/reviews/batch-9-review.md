# Batch 9 review and validation

Scope: OpenSpec add-bilingual-document-qa tasks 9.1–9.3. Completed 2026-11-10.

## Delivery

- Shared Settings-based Windows/Bash launcher preserves environment > dotenv precedence,
  with no forced Chinese model, score threshold, proxy removal or normal hash fallback.
- Explicit full model dependency group, torch CPU/CUDA installation paths, cached-only
  runtime and explicit download/preflight instructions.
- Passive liveness/readiness/model diagnostics separate database, committed index/profile,
  loaded local model/cache/actual device and recently checked generation availability.
  Explicit local preflight reuses the real query model and lock. Generation preflight is
  opt-in and its observation expires after 60 seconds.
- Lightweight default Docker and optional full CPU/CUDA target; data/models and journals
  excluded from image context; CI checks alive versus not-ready.
- Explicit ready document IDs in demo smoke, temporary empty-library 422 matrix for
  ask/SSE/batch, and new/legacy-library migration, rebuild and safe rollback guide.

## Independent review

Two independent sub-agents reviewed requirements and code standards. Both identified
an existing-library restart readiness gap: diagnostics observed only the ingestion
model while queries use a separate instance. Resolved with the explicit local preflight
on the actual query processor and passive observation of its initialized model.
Regression verifies 503 before preflight and 200 after it without uploading new files.
Both reviewers confirmed no remaining blockers. Documentation suggestions were applied.
The spec reviewer also approved the final asynchronous SQLAlchemy dependency correction.

## Verification

Passed targeted offline tests in these files:

- test_local_model_configuration.py, test_model_preparation.py, test_project_hygiene.py
- test_runtime_diagnostics.py, test_schema_migrations.py, test_index_recovery.py
- test_document_deletion.py, test_scoped_query_api.py, test_index_resources.py,
  test_ingestion_cli.py

These use temporary libraries and explicit substitutes, covering config isolation,
cache-only loads, missing weights, new/legacy idempotent migration, deleted-source
replay, preservation of unrelated/new sources and rejection of unprofiled resources.
The updated empty-index smoke passed in a separate process with temporary paths:
empty corpus, alive but not ready, omitted/null/empty/unknown scope all 422 for both
ask transports and batch. No live user library was used for these failure drills.

- Ruff critical rules E9,F63,F7,F82: passed.
- OpenSpec strict validation: passed.
- git diff --check: passed.
- Fresh lightweight Docker build: passed. Initial cold start exposed missing greenlet
  with SQLAlchemy 2.1; corrected both dependency lists to sqlalchemy[asyncio] and added
  hygiene regression. Rebuilt image started successfully.
- Docker smoke used --network none, no host mounts, no real credentials, and a deliberately
  unreachable configured generation endpoint. /health/live=200; /health/=degraded;
  /health/ready=503; generation=not_checked; local model/cache/device=not_checked/null.
  No torch/SentenceTransformers installed and no model cache files created.

Restricted-environment test attempts stalled on local communication and were cancelled;
passing runs used ordinary local permissions. This is not a full batch-10 regression,
real-model quality, actual generation, GPU, capacity or latency acceptance. Batch 10
checkboxes remain open. Full CPU/CUDA image paths are documented; only the lightweight
image was built and run in this batch.
