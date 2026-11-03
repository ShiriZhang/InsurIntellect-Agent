# Using the document library

[中文](user-guide.zh-CN.md) · [API contract](document-query-api.md)

## Prepare and start

1. Install the full Python dependencies from `requirements.txt`. Copy `.env.example`
   to `.env` and configure `LOCAL_EMBEDDING_MODEL`, a fixed
   `LOCAL_EMBEDDING_REVISION`, and `LOCAL_EMBEDDING_CACHE_DIR`. Configure generation
   independently with `GENERATION_API_BASE_URL`, `GENERATION_API_KEY`, and
   `GENERATION_MODEL`. Never put credentials in a PDF or the browser.
2. Run `python -m scripts.prepare_models --download` once to download/cache public
   model assets. Subsequent runs use that cache locally; `python -m
   scripts.prepare_models` checks the local path without downloading. See the
   [model preparation guide](model-configuration.md) for CPU/GPU dependencies and
   diagnostics. `auto` prefers a usable GPU; memory pressure lowers batch size and
   can fall back to the same model on CPU. CPU may be slower; no timing guarantee
   is implied. A missing model never silently switches to hash or a remote model.
3. Start `python -m uvicorn app.main:app --host 127.0.0.1 --port 8766`, then open
   [the local app](http://127.0.0.1:8766/). Application liveness alone does not prove
   the local model, index and generation API are all ready. Use the full Python
   runtime for this workflow; deployment profiles are documented separately.

Document parsing, embeddings, PDFs, SQLite, vectors and keyword retrieval stay on
the local machine. Model downloads contain no document text or questions. Only
the question, necessary same-scope intent context and selected evidence are sent
to the configured generation API; the whole PDF is not uploaded there.

## Empty library to a cited answer

1. Upload one or several text PDFs using **上传 PDF**. Each file is accepted
   separately; other uploads can queue while a prior file is indexing. The default
   size limit is 20 MiB per file, displayed by the page and configurable. The
   200-page maximum is a hard limit. Scanned/mixed scanned or unreadable PDFs fail
   as a whole, with reason and affected pages; OCR is not provided. Blank pages
   preserve original page numbers.
2. Wait for **可提问**, then explicitly check the documents to use. The initial
   selection is empty. A new ready document does not join the selection. Refresh
   restores only still-ready saved IDs; **清空选择** stays empty, including after
   refresh, and blocks submission. More than 50 retained documents shows a design
   capacity warning and still allows uploads.
3. Choose answer language: follow the question, Chinese or English. The explicit
   parameter wins over language wording inside the question. English explanations
   keep source excerpts in their original language.
4. Choose **普通问答** to find relevant clauses within the set; it does not promise
   review of every document. **逐文档比较** checks each selected document, lists
   missing evidence individually and supports at most five documents. Ordinary QA
   can select more than five. Automatic mode displays its resolved mode; ambiguous
   intent returns clarification. Change the selector to override it.
5. For example, select the uploaded sample and ask `What is the waiting period and
   its accident exception?`. Streaming is on by default. Click a numbered citation
   to open that document ID and original page. Same filenames do not identify the
   source. The PDF preview and its dropdown never change query selection.

## Follow-ups and cancellation

Keep selection unchanged for a follow-up such as `What about accidents?`. Each
turn retrieves fresh evidence; an old assistant answer is not a fact source.
Changing selection resets the active context, even if you change back before
asking. The server records a new scope epoch. A question already in progress
keeps its submitted scope and options; later results do not enter a new scope's
context. **重置** starts a new conversation while keeping the document selection.
**停止** stops the current request; truncated output is marked incomplete.

## Delete and retained history

Click **删除** on the specific ID/name and confirm. The document immediately leaves
selection; related streaming answers stop. The UI reports deletion stage, failure
and **重试删除**; it never shows completion before managed cleanup is finished.
Deletion clears the managed original, extracted text, vectors, keyword/source
snapshots, old/temporary copies, historical citation excerpts and related context.
Old citation cards show **来源已删除** without an excerpt or working file link.
Existing answer bodies are retained and may contain earlier quoted words/numbers;
they do not become evidence for follow-ups. External input files, personal backups
and provider-side copies are outside this local erasure boundary. Re-uploading
the same bytes after completed deletion creates a new ID and does not revive old
citations. Keep deletion tombstones when restoring a database or rolling back.

The [batch 8 operation record](reviews/batch-8-operation.md) uses an isolated empty
library, a public synthetic PDF, a real local model and the actual generation API.
Its single successful walkthrough does not replace the separate real-document
quality and capacity acceptance planned in batch 10.
