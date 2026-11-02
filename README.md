# InsurIntellect: An Open-Source Insurance-Clause AI Agent for Local Policy PDF Applications using Large Language Models

[English](README.md) | [中文](README.zh-CN.md)

<div align="center">

<img src="docs/screenshots/logo-192.png" width="160" alt="InsurIntellect icon — citation card with evidence chip"/>

**A clause-grounded AI agent for local insurance policy PDFs**
Not a generic PDF chatbot.

`Clauses are retrieval-grounded` · `Narration is LLM-assisted` · `Every output is citation-tracked or refused`

[![CI](https://img.shields.io/github/actions/workflow/status/Phoenix0531-sudo/InsurIntellect-Agent/ci.yml?branch=master&label=CI)](https://github.com/Phoenix0531-sudo/InsurIntellect-Agent/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/github/license/Phoenix0531-sudo/InsurIntellect-Agent)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux-lightgrey.svg)](#quickstart)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![Chroma](https://img.shields.io/badge/vector-Chroma-FF6F61.svg)](https://www.trychroma.com/)
[![BM25](https://img.shields.io/badge/lexical-BM25%2Bjieba-6C63FF.svg)](#architecture)
[![RAG](https://img.shields.io/badge/RAG-clause--grounded-1D9CFF.svg)](#concept-of-clause-grounded-rag)
[![Not advice](https://img.shields.io/badge/Not%20regulated%20insurance%20advice-critical.svg)](#disclaimer)

**InsurIntellect** turns a natural-language insurance question into a **cited answer or an honest refusal**: local clause PDFs &rarr; hybrid retrieval &rarr; evidence gate &rarr; grounded narration.

</div>

> Not regulated insurance advice. For demonstration and research only.

---

**InsurIntellect** is an AI agent system tailored for insurance policy and clause PDFs. It builds a local clause corpus, retrieves evidence with vector + BM25 search, and uses an OpenAI-compatible large language model only to narrate answers that can be traced back to document, page, and excerpt.

**Concept of Clause-Grounded RAG:** clauses are the environment, hybrid retrieval is the tool layer, and the language model is the narrator. Unlike a general insurance chatbot, InsurIntellect does not answer from model memory and does not act as a licensed insurance advisor. It answers when indexed wording supports the response, and refuses when evidence or regulatory boundary is missing.

This repository keeps the product intentionally compact: a lightweight RAGFlow-style path for insurance clauses, not a multi-tenant SaaS, not a generic ChatGPT advisor, and not an agent canvas platform.

---

## Table of contents

- [What is InsurIntellect?](#what-is-insurintellect)
- [Concept of clause-grounded RAG](#concept-of-clause-grounded-rag)
- [Architecture](#architecture)
- [Retrieved clauses, LLM narration](#retrieved-clauses-llm-narration)
- [Codebase snapshot](#codebase-snapshot)
- [InsurIntellect ecosystem](#insurintellect-ecosystem)
- [InsurIntellect: Agent workflow](#insurintellect-agent-workflow)
- [InsurIntellect: Smart retrieval schedule](#insurintellect-smart-retrieval-schedule)
- [Preview](#preview)
- [Core capabilities](#core-capabilities)
- [Quickstart](#quickstart)
- [Demo questions](#demo-questions)
- [API](#api)
- [Project layout](#project-layout-main-path)
- [Tests and CI](#tests--ci)
- [Scope](#scope)
- [Provider notes](#provider-notes)
- [Disclaimer](#disclaimer)
- [License](#license)
- [Related reading](#related-reading-structure-inspiration-only)

---

## What is InsurIntellect?

InsurIntellect is a clause-grounded AI agent for local insurance documents. It focuses on one repeatable workflow:

```text
Local clause PDFs → hybrid retrieval → evidence gate → cited answer or refusal
```

The project is built for portfolio review: the sample corpus stays synthetic and public, the API stays small, the UI surfaces evidence as first-class content, and CI runs without a live LLM key.

## Concept of clause-grounded RAG

Insurance wording is high-stakes. Waiting periods, exclusions, deductibles, and free-look terms should be read from the policy text, not guessed by a chatbot. InsurIntellect therefore treats the indexed corpus as the source of truth and uses the LLM only after retrieval succeeds.

1. **Clause corpus first.** The answer must come from indexed PDFs, not model memory.
2. **Provenance over fluency.** Every grounded answer exposes document / page / excerpt citations.
3. **Boundary as a product feature.** Purchase advice, guaranteed payout claims, and off-topic questions are refused instead of softened into generic chat.
4. **Reproducible demo.** Public synthetic samples, forced BGE + threshold in `run_demo`, fixed smoke cases (Q1/Q2/Q3/weather), CI without live LLM keys.

---

## Architecture

<div align="center">
  <img align="center" src="docs/screenshots/architecture.svg" width="94%" alt="InsurIntellect architecture"/>
</div>

InsurIntellect keeps one main path in front of the user:

```text
samples/*.pdf
  → simple_ingest.py
  → Chroma vectors + BM25/jieba + corpus_manifest
  → POST /api/v1/queries/ask
  → answer_kind + retrieved_chunks[]
  → dual-pane evidence UI
```

Default path is **`SIMPLE_RAG_MODE=true`**: retrieve → generate.

Off by default (advanced / optional code only): query rewriting, SQL routing, knowledge-graph injection, regulatory multi-agent chains.

## Retrieved clauses, LLM narration

A core design principle of InsurIntellect is the strict separation between **retrieved clause evidence** and **LLM-based narration**.

Policy facts are retrieved from the local corpus through deterministic code paths: PDF text extraction, chunking, embeddings, BM25, vector search, reciprocal-rank fusion, and a score gate. The LLM is used for synthesis, wording, and structure only after those evidence paths return usable clauses.

In short:

```text
Clauses are retrieval-grounded.
Narration is LLM-assisted.
Every output is citation-tracked or refused.
```

### Answer kinds

| `answer_kind` | Meaning | Public `retrieved_chunks` |
|---------------|---------|---------------------------|
| `answer` | Grounded clause Q&A | Real scored citations |
| `refusal` | Off-topic / insufficient evidence | Empty |
| `advice` | Purchase / guarantee / “should I buy” style | Empty |
| `llm_unavailable` | No key or LLM failure; retrieval may still run | Policy-dependent honest degrade |
| `degraded` | Timeout / partial path | Best-effort, honest copy |

---

## Codebase snapshot

| Layer | What it includes |
|-------|------------------|
| **Clause corpus** | Synthetic public sample PDFs plus generator; no real customer policies |
| **Ingest pipeline** | `scripts/generate_sample_corpus.py`, `scripts/simple_ingest.py`, Chroma vectors, BM25/jieba, corpus manifest |
| **Retrieval runtime** | `QueryService`, `rag_workflow`, RRF fusion, BGE threshold gate, public citation curation |
| **Answer protocol** | `answer_kind`, `retrieved_chunks[]`, `public_citations`, structured conclusion / basis / boundary copy |
| **LLM adaptor** | OpenAI-compatible client with no-key and timeout degrade paths |
| **Product UI** | Static ChatPDF-style dual pane: corpus/cited PDF on the left, answer/citations on the right |
| **Verification** | `demo_smoke.py`, `empty_index_smoke.py`, pytest, ruff critical rules, GitHub Actions |

---

## InsurIntellect ecosystem

<div align="center">
  <img align="center" src="docs/screenshots/ecosystem.svg" width="94%" alt="InsurIntellect ecosystem"/>
</div>

The overall framework is organized into four layers:

1. **Application layer:** dual-pane UI, REST API, smoke scripts, and the README demo path.
2. **Clause agent layer:** query guard, evidence curator, answer protocol, and refusal/advice boundary.
3. **Retrieval and DataOps layer:** PDF extraction, chunk map, Chroma vectors, BM25/jieba, RRF fusion, and corpus manifest.
4. **Foundation and governance layer:** OpenAI-compatible LLM, local BGE embeddings, pytest/ruff CI, and the not-advice disclaimer.

## InsurIntellect: Agent workflow

<div align="center">
  <img align="center" src="docs/screenshots/workflow.svg" width="94%" alt="InsurIntellect agent workflow"/>
</div>

1. **Perception:** receive the user question and classify empty, off-topic, purchase-advice, or guarantee-style requests.
2. **Retrieval:** search the local clause corpus with vector + BM25 retrieval and reciprocal-rank fusion.
3. **Evidence gate:** curate citations, deduplicate document/page hits, and stop if scores are too weak.
4. **Narration:** use the LLM only to organize conclusion, clause basis, and uncertainty/boundary text.
5. **Action:** return a stable API response and render citation cards in the UI.

## InsurIntellect: Smart retrieval schedule

<div align="center">
  <img align="center" src="docs/screenshots/schedule.svg" width="94%" alt="InsurIntellect smart retrieval schedule"/>
</div>

The schedule is intentionally simple. At ingest-time, documents are converted into page-aware chunks, vectors, lexical BM25 indexes, and a corpus manifest. At ask-time, the service validates the question, retrieves top-k evidence, curates citations, and either narrates with the LLM or refuses honestly.

This is the insurance-clause analogue of a smart scheduler: it selects between retrieval, refusal, and LLM narration based on evidence quality instead of routing into a heavy multi-agent platform.

---

## Preview

<table align="center">
  <tr>
    <td align="center"><a href="docs/screenshots/preview.png"><img src="docs/screenshots/preview.png" width="100%" alt="Main two-pane UI"/></a><br/><sub><b>Main two-pane UI</b><br/>Corpus &middot; disclaimer &middot; cited answer</sub></td>
    <td align="center"><a href="docs/screenshots/citations.png"><img src="docs/screenshots/citations.png" width="100%" alt="Citation cards"/></a><br/><sub><b>Citation cards</b><br/>Document &middot; page &middot; excerpt evidence</sub></td>
  </tr>
  <tr>
    <td align="center"><a href="docs/screenshots/refuse_advice.png"><img src="docs/screenshots/refuse_advice.png" width="100%" alt="Refuse / advice boundary"/></a><br/><sub><b>Refuse / advice boundary</b><br/>Off-topic &middot; purchase questions &rarr; honest refusal</sub></td>
    <td align="center"><a href="docs/screenshots/preview_empty.png"><img src="docs/screenshots/preview_empty.png" width="100%" alt="Empty / cold start"/></a><br/><sub><b>Empty / cold start</b><br/>No corpus indexed &middot; clear next step</sub></td>
  </tr>
</table>

Left pane: product name, indexed documents, demo prompts, disclaimer.<br>
Right pane: dialogue, structured answer, citation cards (document / page / excerpt).<br>
UI shell is static HTML/CSS/JS (ChatPDF-like dual pane, accent `#1D9CFF`); product story is financial clause RAG, not generic PDF chat.

---

## Core capabilities

Key capabilities include:

| Capability | Detail |
|------------|--------|
| Clause corpus first | Public fake sample PDFs only; no real customer policies in-repo |
| Hybrid retrieval | Chroma + BM25/jieba; default embed `hf:BAAI/bge-small-zh-v1.5` |
| Score gate | `SIMILARITY_THRESHOLD=0.32` for BGE; demo scripts force ingest/query consistency |
| Cited answers | Conclusion / clause basis / boundary + disclaimer line |
| Honest refuse | Weather, purchase advice, “guaranteed payout” → boundary, not free chat |
| Public citation policy | Answer keeps real scores; refuse/advice → `[]` |
| Evidence UI | Dual pane, citation cards, clickable `[n]`, status pill |
| Local-first | `uv` + **[IP]:8766**; OpenAI-compatible gateway |

---

## Quickstart

### 1. Clone and environment

```bash
git clone https://github.com/Phoenix0531-sudo/InsurIntellect-Agent.git
cd InsurIntellect-Agent

uv venv .venv --python 3.11
# Windows: .venv\Scripts\activate
source .venv/bin/activate
uv pip install -r requirements-dev.txt

cp .env.example .env
# set OPENAI_BASE_URL / OPENAI_API_KEY for your OpenAI-compatible gateway
# preferred embedding (must match ingest + query):
# OPENAI_EMBEDDING_MODEL=hf:BAAI/bge-small-zh-v1.5
# SIMILARITY_THRESHOLD=0.32
# offline fallback: OPENAI_EMBEDDING_MODEL=local:hash  (re-ingest after switch)
```

### 2. Sample corpus + ingest

```bash
uv run python scripts/generate_sample_corpus.py --copy-to-data
uv run python -m scripts.simple_ingest --pdf-dir samples
```

Samples are **public synthetic clauses** (waiting period, exclusions, free-look, etc.), not real policies.

The shared ingestion service currently prepares durable candidates (`staged`), while index publication is scheduled for OpenSpec batch 4. Preparation does not yet make a new PDF queryable. See [ingestion and CLI states](docs/document-ingestion.md); `--reset` is rejected to protect other documents.

### 3. One-command demo server

Shell env can pollute `.env`. Prefer the demo launcher, which **re-forces** BGE + threshold:

```bash
bash scripts/run_demo.sh
# Windows: scripts\run_demo.bat
```

Or manually:

```bash
export HOST=127.0.0.1 PORT=8766
export OPENAI_EMBEDDING_MODEL=hf:BAAI/bge-small-zh-v1.5
export SIMILARITY_THRESHOLD=0.32
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export SIMPLE_RAG_MODE=true
uv run uvicorn app.main:app --host 127.0.0.1 --port 8766
```

Open **http://127.0.0.1:8766/**  
API docs (when `DEBUG=true`): **http://127.0.0.1:8766/docs**

If the LLM key is missing, the API still returns an honest **LLM unavailable** path instead of inventing coverage.

### 4. Optional Docker service

Use Compose when you want a persistent local service with a predictable container name:

```bash
docker compose up -d --build
docker compose ps
# container: insurintellect-agent
# URL: http://[IP]:8766/
```

The Compose file intentionally sets `container_name: insurintellect-agent` to avoid anonymous or ambiguous Docker Desktop entries. The Docker image uses `requirements-docker.txt` and `OPENAI_EMBEDDING_MODEL=local:hash` by default so the container can build and expose health quickly without downloading torch/HuggingFace weights; use the uv path above for the full BGE demo.

### 5. Fixed smoke (server must be up)

```bash
# Linux/macOS/git-bash
.venv/bin/python scripts/demo_smoke.py
# Windows
.venv\Scripts\python.exe scripts\demo_smoke.py
```

Covers Q1 / Q2 / Q3 / off-topic weather and asserts `answer_kind` + citation honesty.

Optional empty-index honesty matrix (isolated Windows temp dirs, no live corpus):

```bash
.venv/Scripts/python.exe scripts/empty_index_smoke.py
```

---

## Demo questions

| ID | Question | Expectation |
|----|----------|-------------|
| Q1 | 等待期是多久？ | Cited answer from sample clauses (`answer`, ≥1 citation) |
| Q2 | 责任免除包括哪些情形？ | Multi-point exclusions with citations |
| Q3 | 这份保单保证我一定能获赔吗？ | `advice` boundary; **public citations empty** |
| WX | 今天北京天气怎么样？ | `refusal`; **public citations empty** |

---

## API

| Method | Path | Notes |
|--------|------|--------|
| `GET` | `/api/v1/health/` | DB / vector / LLM status |
| `GET` | `/api/v1/corpus` | Indexed document list for the left pane |
| `POST` | `/api/v1/queries/ask` | Body: `{ "question": "...", "stream": false }` |

Stable response fields include `question`, `answer`, `answer_kind`, `retrieved_chunks[]` (`document_name`, `page_number`, `content`, `similarity_score`), `chunks_used`, `confidence_score`, `response_time`.

Streaming uses the same `POST` with `"stream": true` (SSE); final event carries the same citation policy as non-stream.

Example:

```bash
curl -s http://127.0.0.1:8766/api/v1/queries/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"等待期是多久？","stream":false}'
```

---

## Project layout (main path)

```text
InsurIntellect-Agent
├── app/
│   ├── main.py                 # FastAPI entry, static mount
│   ├── api/                    # /health /queries /corpus
│   ├── core/                   # config, fusion, rag_workflow (backend retrieve)
│   ├── models/                 # schemas + lightweight DB models
│   ├── services/
│   │   ├── query_service.py    # orchestration wrapper around the main ask path
│   │   ├── citation_policy.py  # citation curation, score gate, public evidence
│   │   ├── refusal_policy.py   # advice/off-topic refusal boundary
│   │   ├── response_shaping.py # stable retrieved_chunks schema
│   │   ├── query_history_service.py
│   │   ├── embedding_service.py
│   │   └── llm_service.py
│   └── prompts.py
├── static/                     # dual-pane UI (single JS path: js/app.js)
├── scripts/                    # main path only
│   ├── generate_sample_corpus.py
│   ├── simple_ingest.py
│   ├── run_demo.sh / run_demo.bat
│   ├── demo_smoke.py
│   ├── capture_ui_states.py
│   └── empty_index_smoke.py    # optional honesty matrix (empty chroma)
├── samples/                    # public fake PDFs (source)
├── tools/insurance_ontology.json  # optional advanced rewrite data
├── tests/                      # CI without live LLM keys
├── docs/screenshots/
├── .env.example
├── requirements.txt           # lightweight main path
├── requirements-dev.txt
├── requirements-docker.txt    # quick Docker health/demo profile, no torch download
├── docker-compose.yml         # fixed container_name: insurintellect-agent
└── pyproject.toml             # project metadata + ruff / pytest tool config
```

---

## Tests & CI

```bash
uv run ruff check . --select E9,F63,F7,F82
uv run pytest -q tests
```

GitHub Actions runs the same critical ruff select set + `pytest tests/` **without** requiring a live LLM or embedding API key.

Workflow: [`.github/workflows/ci.yml`](.github/workflows/ci.yml)

---

## Scope

**In**

- Local insurance-like PDF corpus
- Hybrid retrieve + cited answers
- Refuse / advice / LLM-degrade honesty
- Static evidence UI on port 8766

**Out**

- Regulated sales / claim guarantees
- Multi-tenant auth / SaaS upload product
- Full PDF.js reader as a product requirement (assets may exist for highlight demos)
- Productized text-to-SQL / KG / multi-agent orchestration UI

Docker is optional but now follows the same **8766** service port. Compose fixes the readable container name as `insurintellect-agent`, so it is easy to identify in Docker Desktop. The container defaults to `local:hash` embeddings for fast health/demo startup; the uv path remains the recommended route for the full BGE retrieval demo.

---

## Provider notes

- **LLM**: any OpenAI-compatible base URL (portfolio default: local new-api, e.g. `http://127.0.0.1:31876/v1`). Do not commit keys.
- **Embeddings**: keep ingest and query on the same local model/profile. After a model change, follow the [profile rebuild and ingestion guide](docs/document-ingestion.md); directory-wide `--reset` is no longer supported.
- **Hash fallback** `local:hash` is for offline demos only; ranking is weaker — lower threshold only if you knowingly use hash.

---

## Disclaimer

The software and sample documents in this repository are released under the **MIT** license for **demonstration and research**. They must **not** be construed as insurance sales, underwriting, claims handling, or regulated financial advice. Always consult qualified professionals and the official policy wording before any insurance decision.

**Not regulated insurance advice.**

---

## License

MIT. See [LICENSE](LICENSE).

---

## Related reading (structure inspiration only)

- [FinRobot](https://github.com/AI4Finance-Foundation/FinRobot) — financial AI agent platform; provenance / agent product narrative (primary README skeleton)
- [FinGPT](https://github.com/AI4Finance-Foundation/FinGPT) — open financial LLM ecosystem; Why + disclaimer style
- [RAGFlow](https://github.com/infiniflow/ragflow) — general RAG engine; knowledge-base + citation product shape (not insurance-specific)
