# Runtime, installation and safe migration / 运行与迁移

## Install / 安装

Use Python 3.11+ and a dedicated environment. Windows: `python -m venv .venv`,
then `.venv\Scripts\activate`; Linux/macOS: `source .venv/bin/activate`.
Install one appropriate torch wheel first, then the application:

```sh
# CPU (Windows or Linux)
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
# NVIDIA alternative: choose a CUDA wheel compatible with the installed driver,
# e.g. cu128; replace the CPU install above, do not install both profiles.
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

`requirements-docker.txt` is the lightweight application group;
`requirements-models.txt` adds HF, SentenceTransformers and torch;
`requirements.txt` installs both. `requirements-dev.txt` additionally installs test tools.
An installed torch version satisfying the requirements is retained. Select the CUDA
wheel/driver using the PyTorch install selector for your hardware; CUDA availability
must be checked on the actual machine. CPU works without NVIDIA hardware.

复制 `.env.example` 为 `.env`，填写固定模型 revision 和独立的生成配置。
Copy `.env.example` to `.env`; set `LOCAL_EMBEDDING_REVISION` to the verified model
commit, cache directory and device (`auto`, `cpu`, `cuda:N`). `BAAI/bge-m3` is a candidate,
not an accepted quality result. Provision enough RAM/disk for the selected model.

```sh
python -m scripts.prepare_models --download
python -m scripts.prepare_models
# Optional explicit generation call, potentially billable:
python -m scripts.prepare_models --check-generation
```

首次只有显式 `--download` 可下载权重/tokenizer。常规启动和入库仅加载本地缓存；
缓存完整可断开下载站点运行，缺失则明确不可用，不切换 hash、其他模型或远程向量化。
GPU OOM 有界缩小 batch 后回退同模型 CPU，报告实际设备及速度变化。
See [model configuration](model-configuration.md) for precedence, fingerprint and errors.
Model/normalization/tokenizer/chunking changes require rebuilding; generation model
changes alone do not. Real hardware/quality acceptance remains batch 10 work.

Start with `scripts\run_demo.bat` on Windows or `bash scripts/run_demo.sh` elsewhere.
Both use `scripts.launch` and Settings: constructor > environment > `.env` > defaults.
They do not source `.env` as commands or override model, threshold, proxies or keys.
新 `GENERATION_*` 优先于旧别名，显式空值禁用回退，见模型配置说明。

## Diagnostics / 诊断

- `GET /api/v1/health/live`: process alive only; no database/model/provider probe.
- `GET /api/v1/health/`: passive DB/index/generation booleans, `degraded` unless all ready.
- `GET /api/v1/health/ready`: 503 until DB, committed index with matching loaded model
  profile, successful local preflight and recent generation check are all ready.
  Empty library is alive but not ready. `local_embedding` shows cache state, revision,
  actual device, batch size and warnings after model load; before load these are
  honestly `not_checked`/null. Missing revision is `unavailable`.
- `GET /api/v1/health/model`: passive local-model and generation observations; configured
  credentials do not mean connected. These GET requests never load/download models
  or make generation requests.
- `POST /api/v1/health/check-local`: explicit cached local preflight, no download or provider
  call. Use after restarting an existing library; it prepares the same model used by queries.
- `POST /api/v1/health/check-generation`: explicit fixed public probe, may cost tokens;
  records availability/error code for 60 seconds. It sends no library content.
  CLI preflight runs in a separate process and does not set the server observation.

模型预检可随实际入库或显式 `/check-local` 进行；未加载不代表缓存不存在。诊断只报告已观测事实，不声称完整质量通过。

## Docker

```sh
# Lightweight liveness only, no HF/SentenceTransformers/torch runtime
# Copy/configure .env first; compose maps GENERATION_* explicitly (no legacy aliases).
docker compose up --build -d
# Full CPU runtime, same application and data layout
docker build --target full -t insurintellect-agent:cpu .
# Full GPU runtime; select wheel index for the host driver
docker build --target full --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128 -t insurintellect-agent:gpu .
# Mount data/models cache; explicit preparation, never part of health checks
docker run --rm --env-file .env -e LOCAL_EMBEDDING_CACHE_DIR=/app/data/models -v "${PWD}/data:/app/data" insurintellect-agent:cpu python -m scripts.prepare_models --download
docker run --rm --env-file .env -e HOST=0.0.0.0 -e LOCAL_EMBEDDING_CACHE_DIR=/app/data/models -p 8766:8766 -v "${PWD}/data:/app/data" insurintellect-agent:cpu
# GPU: use the gpu image and add --gpus all to both docker run commands.
```

GPU requires the NVIDIA container runtime and host driver; Docker Desktop must be
running. Never bake credentials, runtime databases, deletion journals or model caches
into images. Mount the whole data tree persistently; configure all custom paths inside
that volume. The default image/compose healthcheck uses `/health/live`, not QA readiness.
轻量镜像缺模型依赖时仍可存活，但不能回答或成功入库；完整路径还需要准备缓存与生成 API。

## Empty library and explicit scope / 空库与显式范围

```sh
python -m scripts.generate_sample_corpus
python -m scripts.simple_ingest --pdf-dir samples
# Use the actual ready IDs printed by ingestion or GET /api/v1/documents:
python -m scripts.demo_smoke --document-ids 1 2
python -m scripts.empty_index_smoke
```

The empty smoke uses a temporary library, no model/provider and expects **422** for
missing/null/empty/unknown scope, for both ask transports and batch. No evidence
refusal is not a replacement for scope validation. Uploaded files become queryable
only after committed ready status. New uploads do not automatically expand selection.

```sh
curl -X POST http://127.0.0.1:8766/api/v1/queries/batch -H "Content-Type: application/json" -d '{"questions":["What is the waiting period?","责任免除有哪些？"],"document_ids":[1,2],"answer_language":"en","query_mode":"qa"}'
```

IDs above are examples: replace with your explicitly chosen ready IDs. Batch no
longer accepts a bare question array. A scope mixing valid and invalid IDs is rejected
as a whole before generation/SSE; never fallback to all documents or a valid subset.

## Existing library, rebuild and rollback / 旧库、重建与回退

1. Stop application, CLI and all writers. Record actual SQLite, PDF, Chroma, processed,
   model and revision paths. Take a consistent SQLite backup (backup API, or closed
   connections plus WAL/SHM), and copy the matching files and **latest deletion journal**.
2. On the working copy run twice to verify idempotence:
   `python -c "import asyncio; from app.core.database import init_db; asyncio.run(init_db())"`.
   Migration retains IDs, source/chunk text and answers. Legacy active rows stay pending;
   unknown-scope history stays archived and cannot become follow-up evidence.
3. Never reuse unprofiled Chinese/hash vectors. Preserve old files for audit and rebuild
   only still-active, non-deleted PDFs through `simple_ingest`/upload. Legacy uncontrolled
   paths are not served as ready sources. Record old/new IDs, never reuse deleted IDs.
   There is no directory-reset or in-place model migration command. For a new embedding
   profile, build a separate complete library (new DB/PDF/vector/processed roots), retain
   a map and the latest deletion identities, then validate before switching. Export only
   valid PDFs after deletion replay; never import a stale backup directory wholesale.
4. Startup replays durable deletion work **before** committed-index recovery and ingestion.
   Check failed/deleting tasks to completion before reopening queries. Do not restore from
   a manifest or orphan disk files. Test a copy with [recovery tests](../tests/test_index_recovery.py)
   and [deletion faults](../tests/test_document_deletion.py).
5. Rollback the application only to a schema/profile-compatible release. Keep the latest
   working SQLite, PDFs, new documents, tombstones and deletion task journal together;
   rebuild derived indices if needed. Do not replace the current DB with an older backup:
   that loses new uploads and revives deleted sources. If only an incompatible old release
   is available, keep the service stopped and recover forward on a copy. Automatic downgrade
   and cross-database deletion-ID remapping are not provided.

回退前逐项核对新增来源仍在、已删来源不可读、旧引用无摘录、未知历史不用于追问，
再在临时库演练重启和删除失败恢复。外部备份由用户管理；恢复前也必须核对最新删除记录。
