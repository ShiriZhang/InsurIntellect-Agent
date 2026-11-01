"""Acquire pinned public policy PDFs without importing models or publishing indexes."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import unicodedata
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs/corpus/official-v1.json"
DESTINATION = ROOT / "data/acceptance/official-v1"
MAX_BYTES = 20 * 1024 * 1024
HOSTS = {"resources.pingan.com", "www.income.com.sg", "assets-au-01.kc-usercontent.com"}
# Diagnostic script markers, not a translator or a general language classifier.
SIMPLIFIED = set("保险责医疗约给缴终满岁赔职团体账户华")
TRADITIONAL = set("險責醫療約給繳終滿歲賠職團體賬戶華")


def checked_url(url: str) -> str:
    parsed = urlsplit(url)
    if (parsed.scheme != "https" or parsed.hostname not in HOSTS or parsed.username
            or parsed.password or parsed.port not in (None, 443) or parsed.fragment):
        raise ValueError("unapproved_official_url")
    return url


class OfficialRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        checked_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def load_manifest(path: Path = MANIFEST) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("corpus_id") != "official-v1":
        raise ValueError("unsupported_corpus_manifest")
    documents = manifest["documents"]
    coverage = set()
    ids, hashes = set(), set()
    for entry in documents:
        identifier = entry["id"]
        if not re.fullmatch(r"(?:zh|en)-(?:medical|critical-illness|life)", identifier):
            raise ValueError("invalid_corpus_id")
        for field in ("source_url", "official_landing_url"):
            checked_url(entry[field])
        if entry["language"] not in ("zh-Hans", "en"):
            raise ValueError("unsupported_corpus_language")
        if entry["category"] not in ("medical", "critical-illness", "life"):
            raise ValueError("unsupported_corpus_category")
        if identifier != ("zh" if entry["language"] == "zh-Hans" else "en") + "-" + entry["category"]:
            raise ValueError("inconsistent_corpus_id")
        for field in ("issuer", "jurisdiction", "product", "region_basis", "language_review"):
            if not entry.get(field):
                raise ValueError("missing_provenance_" + field)
        version = entry["visible_version"]
        if not version["text"] or not 1 <= version["page"] <= entry["pages"] <= 200:
            raise ValueError("invalid_visible_version")
        for field in ("issuer_evidence", "region_evidence", "product_evidence"):
            evidence = entry[field]
            if not evidence["text"] or not 1 <= evidence["page"] <= entry["pages"]:
                raise ValueError("invalid_provenance_evidence")
        if (_compact(entry["issuer"]) not in _compact(entry["issuer_evidence"]["text"])
                or _compact(entry["product"]) not in _compact(entry["product_evidence"]["text"])):
            raise ValueError("provenance_label_mismatch")
        collected = datetime.fromisoformat(entry["collected_at"])
        if collected.tzinfo is None:
            raise ValueError("collection_time_requires_timezone")
        if not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]):
            raise ValueError("invalid_sha256")
        if not 0 < entry["size_bytes"] <= MAX_BYTES:
            raise ValueError("invalid_size")
        if identifier in ids or entry["sha256"] in hashes:
            raise ValueError("duplicate_corpus_document")
        ids.add(identifier)
        hashes.add(entry["sha256"])
        coverage.add((entry["language"], entry["category"]))
    expected = {(lang, cat) for lang in ("zh-Hans", "en")
                for cat in ("medical", "critical-illness", "life")}
    if len(documents) != 6 or coverage != expected:
        raise ValueError("incomplete_corpus_coverage")
    return manifest


def download(entry: dict, destination: Path, *, opener=None) -> dict:
    """Only missing pinned files are written; changed official bytes need explicit review."""
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / (entry["id"] + ".pdf")
    if path.exists():
        return {"download": "existing_local_file", "resolved_url": None}
    opener = opener or build_opener(OfficialRedirects())
    request = Request(checked_url(entry["source_url"]), headers={"User-Agent": "InsurIntellect-Corpus/1"})
    with opener.open(request, timeout=45) as response:
        resolved = checked_url(response.geturl())
        data = response.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise ValueError("download_too_large")
    if len(data) != entry["size_bytes"] or hashlib.sha256(data).hexdigest() != entry["sha256"]:
        raise ValueError("official_bytes_changed_review_required")
    if not data.startswith(b"%PDF-"):
        raise ValueError("not_pdf")
    # Exclusive create avoids clobbering either an existing artifact or a concurrent acquisition.
    with path.open("xb") as handle:
        handle.write(data)
    return {"download": "fetched", "resolved_url": resolved}


def _compact(text: str) -> str:
    return "".join(text.split())


def inspect_document(entry: dict, path: Path) -> dict:
    import fitz
    # Reuse the project's actual extraction entry, with no embedding or generation call.
    from scripts.simple_ingest import _extract_pages

    if path.stat().st_size != entry["size_bytes"]:
        raise ValueError("local_file_identity_mismatch")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != entry["sha256"]:
        raise ValueError("local_file_identity_mismatch")
    with fitz.open(path) as document:
        if document.needs_pass or len(document) != entry["pages"] or not 0 < len(document) <= 200:
            raise ValueError("encrypted_or_page_count_mismatch")
    pages = _extract_pages(path)
    if [number for number, _ in pages] != list(range(1, entry["pages"] + 1)):
        # This corpus deliberately contains no blank pages; omissions need candidate review.
        raise ValueError("project_extraction_missing_pages")
    diagnostics = []
    for number, text in pages:
        bad = sum(c == "\ufffd" or unicodedata.category(c) in {"Co", "Cs", "Cn"} for c in text)
        if bad or sum(c.isalnum() for c in text) < 25:
            raise ValueError(f"unreadable_page_{number}")
        diagnostics.append({"page": number, "characters": len(text), "invalid_characters": bad,
                            "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()})
    version = entry["visible_version"]
    if _compact(version["text"]) not in _compact(pages[version["page"] - 1][1]):
        raise ValueError("visible_version_not_in_body")
    for field in ("issuer_evidence", "region_evidence", "product_evidence"):
        evidence = entry[field]
        if _compact(evidence["text"]) not in _compact(pages[evidence["page"] - 1][1]):
            raise ValueError(field + "_not_in_body")
    text = "\n".join(t for _, t in pages)
    simplified = sum(c in SIMPLIFIED for c in text)
    traditional = sum(c in TRADITIONAL for c in text)
    cjk = sum("\u3400" <= c <= "\u9fff" for c in text)
    latin = sum(c.isascii() and c.isalpha() for c in text)
    if entry["language"] == "zh-Hans":
        if simplified < 20 or traditional / max(1, simplified + traditional) > 0.05 or cjk <= latin:
            raise ValueError("simplified_chinese_review_failed")
    elif latin < 100 or cjk > latin * 0.05:
        raise ValueError("english_review_failed")
    return {"id": entry["id"], "status": "collected_project_parse_checked", "sha256": entry["sha256"],
            "size_bytes": len(data), "pages": len(pages), "page_checks": diagnostics,
            "script_check": {"simplified_markers": simplified, "traditional_markers": traditional,
                             "cjk_characters": cjk, "latin_characters": latin},
            "visible_version_verified": True, "provenance_anchors_verified": True,
            "application_ready": False}


def run(manifest_path: Path, destination: Path, *, fetch: bool = False) -> dict:
    manifest = load_manifest(manifest_path)
    results = []
    for entry in manifest["documents"]:
        try:
            acquisition = download(entry, destination) if fetch else {}
            result = inspect_document(entry, destination / (entry["id"] + ".pdf"))
            result.update(acquisition)
        except Exception as exc:
            result = {"id": entry["id"], "status": "candidate_failed", "error": str(exc),
                      "application_ready": False}
        results.append(result)
    return {"corpus_id": manifest["corpus_id"], "checked_at": datetime.now(timezone.utc).isoformat(),
            "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "parser": "scripts.simple_ingest._extract_pages", "pymupdf_version": __import__("fitz").VersionBind,
            "all_six_collected_and_parse_checked": all(r["status"] == "collected_project_parse_checked" for r in results),
            "application_ready": False, "documents": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--destination", type=Path, default=DESTINATION)
    parser.add_argument("--fetch", action="store_true", help="Fetch missing pinned official PDFs; default is offline verification")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.manifest, args.destination, fetch=args.fetch)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["all_six_collected_and_parse_checked"],
                      "documents": [{k: v for k, v in r.items() if k in ("id", "status", "error", "pages")}
                                    for r in report["documents"]]}, ensure_ascii=False))
    raise SystemExit(0 if report["all_six_collected_and_parse_checked"] else 1)


if __name__ == "__main__":
    main()
