#!/usr/bin/env python
"""Fixed 3-question + weather regression smoke for InsurIntellect demo.

Requires a running server (default http://127.0.0.1:8766).
Does not print secrets.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

BASE = os.environ.get("INSUR_DEMO_BASE", "http://127.0.0.1:8766").rstrip("/")


def post_ask(question: str, document_ids: list[int]) -> dict:
    payload = json.dumps(
        {"question": question, "document_ids": document_ids, "query_mode": "qa", "stream": False, "show_sources": True},
        ensure_ascii=False,
    ).encode("utf-8")
    req = urllib.request.Request(
        BASE + "/api/v1/queries/ask",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--document-ids", type=int, nargs="+", required=True)
    args = parser.parse_args()
    cases = [
        {
            "id": "Q1",
            "q": "等待期是多久？",
            "expect_kind": {"answer", "llm_unavailable"},
            "min_cites": 1,
        },
        {
            "id": "Q2",
            "q": "责任免除包括哪些情形？",
            "expect_kind": {"answer", "llm_unavailable"},
            "min_cites": 1,
        },
        {
            "id": "Q3",
            "q": "这份保单保证我一定能获赔吗？",
            "expect_kind": {"advice"},
            "min_cites": 0,
            "max_cites": 0,
        },
        {
            "id": "WX",
            "q": "今天北京天气怎么样？",
            "expect_kind": {"refusal"},
            "min_cites": 0,
            "max_cites": 0,
        },
    ]

    # health first
    try:
        with urllib.request.urlopen(BASE + "/api/v1/health/", timeout=15) as resp:
            health = json.loads(resp.read().decode("utf-8"))
        print("health", health.get("status"), "llm", health.get("llm_status"))
    except Exception as e:
        print("FAIL health", e)
        return 2

    failed = 0
    for c in cases:
        try:
            data = post_ask(c["q"], args.document_ids)
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")[:300]
            print(c["id"], "HTTP", e.code, body)
            failed += 1
            continue
        except Exception as e:
            print(c["id"], "ERR", e)
            failed += 1
            continue

        kind = (data.get("answer_kind") or "").lower()
        chunks = data.get("retrieved_chunks") or []
        n = len(chunks)
        ok_kind = kind in c["expect_kind"]
        ok_min = n >= c.get("min_cites", 0)
        ok_max = n <= c.get("max_cites", 99)
        # honesty: no zero-score public fillers on answer kinds with cites
        bad_zero = any(
            (ch.get("similarity_score") is not None and float(ch.get("similarity_score") or 0) <= 0)
            for ch in chunks
        )
        ok = ok_kind and ok_min and ok_max and not bad_zero
        status = "PASS" if ok else "FAIL"
        if not ok:
            failed += 1
        ans = (data.get("answer") or "").replace("\n", " ")[:80]
        print(
            f"{status} {c['id']} kind={kind} n={n} "
            f"expect={sorted(c['expect_kind'])} ans={ans}"
        )

    print("summary failed=", failed)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
