# Document and scoped query API

Upload one multipart PDF per request (`file` field). The default configurable
`UPLOAD_MAX_BYTES` is 20 MiB (413 on excess); unsupported media types return 415,
invalid PDF signatures/fields return 422. Encrypted, corrupt, scanned/mixed or
over-200-page PDFs fail during background parsing with error details/pages; they
never become ready. New and failed-retry attempts return 202; active duplicate
content returns 200 with the existing ID. Uploads over the 50-document design
capacity remain accepted; the list reports a warning.

```sh
curl -F 'file=@policy.pdf;type=application/pdf' http://localhost:8000/api/v1/documents
curl http://localhost:8000/api/v1/documents
curl http://localhost:8000/api/v1/documents/1
curl http://localhost:8000/api/v1/documents/1/file -o policy.pdf
```

Poll detail until `ready=true`. Names are display fields; use `document_id` and
`file_url` for selection/preview, including when filenames are identical.
`GET /api/v1/corpus` projects the database library, including processing states;
it never imports a manifest or infers readiness from files. Unknown IDs return
404; inactive/deleting/deleted originals return 410 and are not cached.

## BREAKING: explicit query scope

Old calls containing only `question` (or batch JSON arrays) are rejected. Select
ready IDs from the document list. Missing/null/empty scopes, unknown IDs and any
non-ready or deleting/deleted source return 422 before generation or SSE starts.
Duplicate IDs are deduplicated; a mixed valid/invalid list is rejected as a whole.

```sh
curl -H 'Content-Type: application/json' http://localhost:8000/api/v1/queries/ask \
  -d '{"question":"What is the waiting period?","document_ids":[1],"query_mode":"qa","stream":true}'
curl -H 'Content-Type: application/json' http://localhost:8000/api/v1/queries/batch \
  -d '{"questions":["解释等待期","Compare waiting periods"],"document_ids":[1,2],"query_mode":"auto"}'
```

`query_mode=qa` searches the selected set for relevant evidence without promising
coverage of every document. `compare` recalls evidence independently for every
selected document, reusing one locally encoded question vector; at most five
unique IDs may be compared (422 before SSE). `auto` recognizes explicit Chinese
and English comparison wording; ambiguous multi-document wording returns
`answer_kind=clarification`. A manual qa/compare choice takes priority. Ordinary
QA can search more than five documents.

Both transports expose actual `document_ids`, `query_mode`, and, for comparison,
`document_coverage` (`evidence` or `insufficient_evidence` for each ID and name).
Missing evidence is also listed in the final answer. Each document receives its
own evidence budget; an unsafe context returns clarification and asks to narrow
the question/scope, without silently claiming a partial comparison. Scope,
publication attempts and index commit are frozen for the request and rechecked
before generation and final saving. Filter failures disable that path; there is
no unscoped retry or SQL/KG evidence path.

Final citations retain the fusion order within each document and document-ID
order across a comparison. Deduplication uses document/attempt/chunk identity or
exact source text, preserving different clauses on the same page and short
English conditions. `metadata.citation_number` matches the generation prompt's
one-based input order and the final SSE/API/history list. Evidence isn't reordered
after generation.

## Bilingual answers and follow-ups

`answer_language=auto|zh|en`: an explicit parameter wins, followed by a clear
language instruction in the question, then the main question language. Ambiguous
mixed input falls back to Chinese. Responses and SSE start/end include the resolved
`answer_language`. English uses `[Conclusion]`, `[Clause evidence]`, and
`[Uncertainty / boundaries]`; Chinese uses the existing three bracketed titles.
Evidence remains in its original language. Refusal, advice, clarification, local
model/index unavailability, generation unavailability, timeout and cancellation
have local templates in both languages. Service failures never claim missing evidence.

```json
{"question":"等待期多久？","document_ids":[1],"answer_language":"en","session_id":"my-session"}
```

Reuse the returned `session_id` for follow-ups. Omitting it creates a new session.
Sorted, deduplicated selections define the scope fingerprint. Changing selection
creates a new increasing `scope_epoch`, including when returning to an earlier
selection; uploading a new document does not change selection. Late responses
remain attached to their originally accepted epoch. Bare ambiguous references
receive clarification. A follow-up may reuse prior user questions for intent;
assistant answers never supply facts, and every factual answer retrieves fresh
source evidence. Explicit manual modes take priority over inherited comparison.

Ask, SSE end, history detail and history list expose scope IDs, fingerprint, epoch,
resolved language/mode, index commit, source status and terminal answer kind.
All terminal outcomes are saved by the same transaction boundary, including local
refusals and service failures. Deletion/cancellation is checked again before save;
a deleted source cannot be written back as a normal completed answer or citation.
Interrupted streams end with an incomplete terminal outcome, or record cancellation
when the client disconnects. Historical unknown scopes remain unknown and cannot
be used for new follow-ups. Document deletion orchestration is described below.

Offline regression uses explicit encoder/generation stand-ins; real bilingual
model quality still requires the separate batch 10 acceptance run.

## Deleting a source (batch 7)

`DELETE /api/v1/documents/{id}` is idempotent: 202 means deletion is pending;
200 means all managed cleanup completed; unknown IDs return 404. Poll document
list/detail for `deletion_status`, `deletion_stage`, `deletion_task_status` and
`deletion_error`. Failure remains `deleting` with task status `failed`; the worker
retries automatically, and another DELETE requests a retry. No fake percentage
or successful `deleted` status is returned while readers/native work/cleanup remain.

Deletion first persists its tombstone, invalidates session epochs, blocks new
queries and previews, and cancels related preparation and generation/SSE work.
Deleting A during A+B comparison cancels the entire answer; B-only work survives.
Blocked upstream generation is interrupted and closed rather than waiting for a
new token. File streams check source identity on every read. Native encoders
cannot be killed safely: their late results are discarded and cleanup waits for
their document lease. Read leases work across local application/CLI processes.

The worker erases controlled originals, SQLite chunks/checkpoints, source vectors,
BM25/source snapshots and managed old/temporary versions. It rebuilds the remaining
index from existing local checkpoints without loading a model or calling any API.
Historical answer bodies remain, even when they contain old source numbers. Deleted
citations retain only chunk/document identity, name, page and `source_status=deleted`;
there is no `content`, vector, original URL or metadata payload. Related rewrite
materials and session context are invalidated. Late answers/history writes recheck
the durable source state and cannot restore evidence.

Keep the database tombstones when restoring a backup or rolling back the app.
Startup and CLI recovery replay them before resuming ingestion/index service,
including completed tombstones when older managed files were restored. Restored
snapshots/manifests cannot override readiness or revive a source. External original
input directories and user-owned backups are not traversed or erased. Re-uploading
identical bytes after successful deletion creates a new document ID; old citations
remain deleted. This is application-level erasure of readable managed material,
not a secure-wipe promise for storage media or an external generation provider.

## Browser selection preparation

`POST /api/v1/queries/session` accepts `session_id` and explicit `document_ids`.
Nonempty selections must consist entirely of ready IDs. This endpoint records a
scope epoch without generating an answer; an empty list records an explicit clear.
It never retrieves evidence or enables an empty-scope ask. The browser serializes
selection updates so A → empty → A resets follow-up context even without an
intervening question. Ask/batch still require explicit nonempty scopes.

The browser includes the returned `scope_epoch` in each ask snapshot. A delayed
request keeps that exact historical scope epoch and cannot advance or populate a
newly selected epoch, even after A → B → A. Invalid/invalidated or mismatched
epochs return 422. API callers omitting this optional field retain scope-at-request
preparation; callers managing selection separately should bind the returned epoch.
