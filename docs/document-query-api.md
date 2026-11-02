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

Answer-language templates, scope-aware follow-up epochs and the complete deletion
API/UI are subsequent batches. These API regression tests use offline encoder and
generation stand-ins, and do not establish real bilingual model quality.
