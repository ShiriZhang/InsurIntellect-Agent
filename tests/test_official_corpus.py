"""No network, real weights, real personal data or generation service required."""
import copy
import hashlib
import io
import json
from types import SimpleNamespace

import fitz
import pytest

from scripts import official_corpus as corpus


def test_committed_manifest_and_real_parse_report_are_consistent():
    manifest = corpus.load_manifest()
    report = json.loads((corpus.MANIFEST.parent / "official-v1.parse-report.json").read_text(encoding="utf-8"))
    assert report["manifest_sha256"] == hashlib.sha256(corpus.MANIFEST.read_bytes()).hexdigest()
    assert report["all_six_collected_and_parse_checked"]
    assert not report["application_ready"]
    assert len(report["documents"]) == 6
    assert sum(d["pages"] for d in report["documents"]) == 120
    for expected, actual in zip(manifest["documents"], report["documents"]):
        assert (expected["id"], expected["sha256"], expected["pages"]) == (actual["id"], actual["sha256"], actual["pages"])
        assert actual["visible_version_verified"] and actual["provenance_anchors_verified"]
        assert [p["page"] for p in actual["page_checks"]] == list(range(1, actual["pages"] + 1))
        assert not actual["application_ready"]


@pytest.mark.parametrize("mutation,error", [
    (lambda d: d["documents"].pop(), "incomplete_corpus_coverage"),
    (lambda d: d["documents"][0].update(language="zh-Hant"), "unsupported_corpus_language"),
    (lambda d: d["documents"][0].update(id="../outside"), "invalid_corpus_id"),
    (lambda d: d["documents"][0].update(jurisdiction=""), "missing_provenance"),
    (lambda d: d["documents"][0].update(issuer="Incorrect issuer"), "provenance_label_mismatch"),
    (lambda d: d["documents"][0].update(source_url="https://example.org/policy.pdf"), "unapproved_official_url"),
    (lambda d: d["documents"][0].update(sha256="0"), "invalid_sha256"),
    (lambda d: d["documents"][0].update(pages=201), "invalid_visible_version"),
    (lambda d: d["documents"][0].update(sha256=d["documents"][1]["sha256"]), "duplicate_corpus_document"),
])
def test_manifest_rejects_missing_coverage_and_unreviewed_provenance(tmp_path, mutation, error):
    manifest = copy.deepcopy(corpus.load_manifest())
    mutation(manifest)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match=error):
        corpus.load_manifest(path)


@pytest.mark.parametrize("url", ["http://resources.pingan.com/a.pdf", "https://localhost/a.pdf",
    "https://resources.pingan.com.evil.example/a.pdf", "https://user:secret@resources.pingan.com/a.pdf",
    "https://resources.pingan.com:444/a.pdf", "file:///tmp/a.pdf"])
def test_url_and_redirect_host_controls(url):
    with pytest.raises(ValueError, match="unapproved_official_url"):
        corpus.checked_url(url)
    with pytest.raises(ValueError, match="unapproved_official_url"):
        corpus.OfficialRedirects().redirect_request(None, None, 302, "Found", {}, url)


def fake_response(data, url="https://resources.pingan.com/policy.pdf"):
    response = io.BytesIO(data)
    response.geturl = lambda: url
    return response


def download_entry(data):
    return {"id": "en-life", "source_url": "https://resources.pingan.com/policy.pdf",
            "size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def test_fetch_pinned_bytes_and_never_overwrites_existing_file(tmp_path):
    data = b"%PDF-synthetic transport fixture"
    calls = []
    opener = SimpleNamespace(open=lambda request, timeout: calls.append((request.full_url, timeout)) or fake_response(data))
    entry = download_entry(data)
    result = corpus.download(entry, tmp_path, opener=opener)
    assert result["download"] == "fetched"
    path = tmp_path / "en-life.pdf"
    assert path.read_bytes() == data
    path.write_bytes(b"preserve local changes")
    assert corpus.download(entry, tmp_path, opener=opener)["download"] == "existing_local_file"
    assert path.read_bytes() == b"preserve local changes" and len(calls) == 1


@pytest.mark.parametrize("body,url,error", [
    (b"changed", "https://resources.pingan.com/policy.pdf", "official_bytes_changed"),
    (b"%PDF-original", "https://evil.example/policy.pdf", "unapproved_official_url"),
])
def test_fetch_does_not_persist_changed_or_unapproved_content(tmp_path, body, url, error):
    opener = SimpleNamespace(open=lambda *a, **kw: fake_response(body, url))
    with pytest.raises(ValueError, match=error):
        corpus.download(download_entry(b"%PDF-original"), tmp_path, opener=opener)
    assert not list(tmp_path.glob("*.pdf"))


def test_fetch_size_limit_is_bounded(tmp_path, monkeypatch):
    monkeypatch.setattr(corpus, "MAX_BYTES", 10)
    opener = SimpleNamespace(open=lambda *a, **kw: fake_response(b"%PDF-" + b"x" * 20))
    with pytest.raises(ValueError, match="download_too_large"):
        corpus.download(download_entry(b"%PDF-small"), tmp_path, opener=opener)
    assert not list(tmp_path.glob("*.pdf"))


def synthetic_pdf(tmp_path, *, language="en", blank_second=False, traditional=False):
    entry = copy.deepcopy(corpus.load_manifest()["documents"][-1])
    path = tmp_path / "fixture.pdf"
    with fitz.open() as doc:
        page = doc.new_page()
        body = ("Fixture policy version V1\nIncome Insurance Limited Singapore\n"
                "DIRECT Star Term (non-renewable)\n"
                "This synthetic insurance contract is provided for testing only. "
                "A waiting period does not alter any exclusion.\n")
        if language == "zh-Hans":
            body = (("保險責任醫療約定給付賠償團體賬戶" if traditional else "保险责任医疗约定给付赔偿团体账户") + "\n") * 8
        page.insert_textbox(fitz.Rect(30, 30, 560, 750), body, fontname="china-s" if language == "zh-Hans" else "helv")
        if blank_second:
            doc.new_page().draw_rect(fitz.Rect(30, 30, 500, 700))
        doc.save(path)
    data = path.read_bytes()
    anchor = "保險責任" if traditional else "保险责任"
    entry.update(language=language, size_bytes=len(data), sha256=hashlib.sha256(data).hexdigest(), pages=2 if blank_second else 1)
    entry["visible_version"] = {"page": 1, "text": anchor if language == "zh-Hans" else "Fixture policy version V1"}
    for field in ("issuer_evidence", "product_evidence", "region_evidence"):
        entry[field] = {"page": 1, "text": anchor if language == "zh-Hans" else "Singapore"}
    return entry, path


def test_actual_project_parser_preserves_page_identity_and_checks_body_version(tmp_path):
    entry, path = synthetic_pdf(tmp_path)
    result = corpus.inspect_document(entry, path)
    assert result["pages"] == 1 and result["page_checks"][0]["page"] == 1
    assert not result["application_ready"]
    entry["visible_version"]["text"] = "V2-from-filename"
    with pytest.raises(ValueError, match="visible_version_not_in_body"):
        corpus.inspect_document(entry, path)


def test_actual_parser_rejects_one_unreadable_page_in_a_document(tmp_path):
    entry, path = synthetic_pdf(tmp_path, blank_second=True)
    with pytest.raises(ValueError, match="project_extraction_missing_pages"):
        corpus.inspect_document(entry, path)


def test_chinese_script_check_preserves_simplified_and_rejects_traditional(tmp_path):
    entry, path = synthetic_pdf(tmp_path, language="zh-Hans")
    assert corpus.inspect_document(entry, path)["script_check"]["simplified_markers"] > 20
    entry, path = synthetic_pdf(tmp_path, language="zh-Hans", traditional=True)
    with pytest.raises(ValueError, match="simplified_chinese_review_failed"):
        corpus.inspect_document(entry, path)


def test_local_tampering_and_incorrect_provenance_are_rejected(tmp_path):
    entry, path = synthetic_pdf(tmp_path)
    entry["region_evidence"]["text"] = "Some other jurisdiction"
    with pytest.raises(ValueError, match="region_evidence_not_in_body"):
        corpus.inspect_document(entry, path)
    path.write_bytes(path.read_bytes() + b"modified")
    with pytest.raises(ValueError, match="local_file_identity_mismatch"):
        corpus.inspect_document(entry, path)


def test_default_run_is_offline_and_missing_sources_cannot_be_usable(tmp_path, monkeypatch):
    monkeypatch.setattr(corpus, "download", lambda *a, **kw: pytest.fail("implicit network access"))
    report = corpus.run(corpus.MANIFEST, tmp_path)
    assert not report["all_six_collected_and_parse_checked"]
    assert not report["application_ready"]
    assert all(d["status"] == "candidate_failed" for d in report["documents"])
