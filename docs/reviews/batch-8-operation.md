# Batch 8 real browser operation

Public synthetic PDF, real local embedding and actual generation API; a single walkthrough does not establish batch 10 quality acceptance.

```json
{
  "real_local_embedding": true,
  "real_generation_api": true,
  "work_directory": "C:\\Users\\23657\\AppData\\Local\\Temp\\insur-ui-real-x9dijn1k",
  "steps": [
    "Empty managed library confirmed; submit disabled without selection.",
    "Real local encoding and API generation returned 90-day clause and accident exception with original citation; browser opened PDF by ID.",
    "Deletion completed; file URL returns 410; old card has no excerpt; answer body retained."
  ],
  "ingestion_seconds": 27.91,
  "answer_seconds": 10.28,
  "answer_kind": "answer",
  "answer": "[Conclusion]  \nThe policy imposes a 90‑day waiting period from the effective date, but this waiting period does **not** apply to accidental injuries, which are exempt.\n\n[Clause evidence]  \n- “The waiting period is 90 days from the effective date. Accidental injury is exempt from the waiting period.” [1]  \n- “Sample insurance policy” [2]\n\n[Uncertainty / boundaries]  \nThe provided excerpts contain no additional conditions, exceptions, or variations to the waiting period beyond what is stated above. No further details are available in the supplied evidence.",
  "source_ids": [
    1,
    1
  ],
  "deletion_seconds": 0.67,
  "embedding": {
    "status": "ready",
    "model": "BAAI/bge-m3",
    "revision": "5617a9f61b028005a4858fdac845db406aefb181",
    "actual_revision": "5617a9f61b028005a4858fdac845db406aefb181",
    "device": "cpu",
    "batch_size": 32,
    "dimension": 1024,
    "distance": "cosine",
    "max_input_tokens": 512,
    "embedding_profile": {
      "fingerprint": "03f1923af772246be2f216f8d4b48d5954762e8d335e478340dc85fe7c1df6d5",
      "profile": {
        "actual_revision": "5617a9f61b028005a4858fdac845db406aefb181",
        "chunking": {
          "context": "section-title-and-preceding-source-tail",
          "generation_budget": "utf8-byte-upper-bound",
          "generation_chunk_bytes": 2048,
          "overlap": 40,
          "size": 400,
          "source_offsets": "characters",
          "version": "page-clause-token-v1"
        },
        "dimension": 1024,
        "distance": "cosine",
        "encoder_versions": {
          "langchain-huggingface": "1.2.3",
          "sentence-transformers": "6.2.0",
          "tokenizers": "0.21.4",
          "transformers": "4.53.3"
        },
        "input_budget": {
          "implicit_model_prompt": "disabled",
          "max_tokens": 512,
          "overflow": "reject",
          "special_tokens": "included"
        },
        "local_assets_sha256": null,
        "model_id": "BAAI/bge-m3",
        "normalize_vectors": true,
        "prefixes": {
          "document_sha256": "12ae32cb1ec02d01eda3581b127c1fee3b0dc53572ed6baf239721a03d82e126",
          "query_sha256": "12ae32cb1ec02d01eda3581b127c1fee3b0dc53572ed6baf239721a03d82e126"
        },
        "requested_revision": "5617a9f61b028005a4858fdac845db406aefb181",
        "schema_version": 1,
        "text_preprocessing": {
          "newlines": "space",
          "normalization": "preserve",
          "version": "local-text-v1"
        },
        "tokenizer_sha256": "9f35ab0cdefe78b2050c2de5cf30978fbeb659107b43ded81c091342ef8a6505"
      }
    },
    "warnings": [
      {
        "code": "cpu_execution",
        "message": "CPU encoding is active; processing may be slower"
      }
    ]
  },
  "generation_model": "openai/gpt-oss-120b",
  "status": "passed"
}
```
