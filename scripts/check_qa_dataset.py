"""Validate evaluation annotations offline; never run models or mark QA acceptance passed."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from scripts.official_corpus import MANIFEST, ROOT, load_manifest

DATASET = ROOT / "docs/evaluation/bilingual-qa-v1.json"
PARSE_REPORT = ROOT / "docs/corpus/official-v1.parse-report.json"


def compact(value: str) -> str:
    return "".join(value.split()).casefold()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_dataset(dataset: dict, *, manifest_path: Path = MANIFEST,
                  parse_report_path: Path = PARSE_REPORT, pdf_dir: Path | None = None) -> dict:
    manifest = load_manifest(manifest_path)
    sources = {d["id"]: d for d in manifest["documents"]}
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    require(dataset["schema_version"] == 1 and dataset["corpus_id"] == manifest["corpus_id"], "dataset_version_mismatch")
    require(dataset["corpus_manifest_sha256"] == manifest_hash, "corpus_manifest_changed")
    require(dataset["execution_status"] == "not_run", "dataset_is_not_an_execution_report")
    parse_report = json.loads(parse_report_path.read_text(encoding="utf-8"))
    require(parse_report["manifest_sha256"] == manifest_hash, "parse_report_manifest_mismatch")
    require(parse_report["all_six_collected_and_parse_checked"], "corpus_parse_not_checked")
    page_hashes = {(d["id"], p["page"]): p["text_sha256"]
                   for d in parse_report["documents"] for p in d["page_checks"]}
    pages = {}
    if pdf_dir is not None:
        from scripts.simple_ingest import _extract_pages

        for identifier, source in sources.items():
            path = pdf_dir / (identifier + ".pdf")
            require(hashlib.sha256(path.read_bytes()).hexdigest() == source["sha256"], "source_pdf_identity_mismatch:" + identifier)
            extracted = _extract_pages(path)
            require([p for p, _ in extracted] == list(range(1, source["pages"] + 1)), "source_pdf_missing_pages:" + identifier)
            for number, text in extracted:
                require(hashlib.sha256(text.encode("utf-8")).hexdigest() == page_hashes[identifier, number], "source_page_text_changed")
                pages[identifier, number] = compact(text)

    def evidence_checked(source_id, evidence):
        require(source_id in sources, "unknown_source")
        require(type(evidence["page"]) is int and 1 <= evidence["page"] <= sources[source_id]["pages"], "invalid_evidence_page")
        require(len(compact(evidence["anchor"])) >= 6, "empty_or_ambiguous_evidence_anchor")
        key = source_id, evidence["page"]
        require(evidence["page_text_sha256"] == page_hashes[key], "page_annotation_identity_mismatch")
        if pdf_dir is not None:
            require(compact(evidence["anchor"]) in pages[key], "anchor_not_on_source_page:" + source_id + ":" + str(evidence["page"]) + ":" + evidence["anchor"])

    facts = {}
    semantic_keys = set()
    claim_keys = {}
    anchor_keys = {}
    for fact in dataset["facts"]:
        identifier, source_id = fact["id"], fact["source_id"]
        require(identifier not in facts, "duplicate_fact_id")
        require(fact["split"] in {"acceptance", "calibration"}, "invalid_fact_split")
        require(fact["semantic_key"] and fact["semantic_key"] not in semantic_keys, "reused_semantic_fact")
        semantic_keys.add(fact["semantic_key"])
        require(source_id in sources, "unknown_source")
        require(fact["source_sha256"] == sources[source_id]["sha256"], "fact_source_hash_mismatch")
        require(fact["jurisdiction"] == sources[source_id]["jurisdiction"], "jurisdiction_mismatch")
        for field in ("required_points", "conditions", "forbidden_claims", "evidence", "critical_fields"):
            require(bool(fact[field]), "missing_fact_annotation:" + field)
        for field in ("conditions", "forbidden_claims"):
            require(all(isinstance(point, str) and point.strip() for point in fact[field]), "empty_fact_annotation:" + field)
        for point in fact["required_points"]:
            require(isinstance(point, str) and bool(point.strip()), "empty_required_point")
            key = source_id, compact(point)
            require(key not in claim_keys or claim_keys[key] == fact["split"], "calibration_reuses_acceptance_answer")
            claim_keys[key] = fact["split"]
        for field in fact["critical_fields"]:
            require(all(isinstance(field.get(k), str) and field[k].strip() for k in ("kind", "value", "unit", "qualifier")), "incomplete_critical_field")
        for evidence in fact["evidence"]:
            evidence_checked(source_id, evidence)
            key = source_id, compact(evidence["anchor"])
            require(key not in anchor_keys or anchor_keys[key] == fact["split"], "calibration_reuses_acceptance_evidence")
            anchor_keys[key] = fact["split"]
        facts[identifier] = fact

    case_ids, questions = set(), set()
    referenced = set()
    for split, name in (("acceptance", "acceptance"), ("calibration", "calibration")):
        for case in dataset[name]:
            require(case["id"] not in case_ids, "duplicate_case_id")
            case_ids.add(case["id"])
            require(isinstance(case["question"], str) and case["question"].strip(), "empty_question")
            require(compact(case["question"]) not in questions, "reused_question")
            questions.add(compact(case["question"]))
            require(case["fact_ids"] and case["document_ids"], "empty_case_scope_or_facts")
            require(case["question_language"] in {"zh", "en"}, "invalid_question_language")
            require(case["answer_language"] == "auto" and case["expected_answer_language"] == case["question_language"], "answer_language_mismatch")
            require(case["query_mode"] == "qa" and case["expected_kind"] == "answer", "invalid_positive_case")
            require(len(set(case["document_ids"])) == len(case["document_ids"]) and set(case["document_ids"]) <= sources.keys(), "invalid_case_scope")
            target_languages = set()
            for fact_id in case["fact_ids"]:
                require(fact_id in facts and facts[fact_id]["split"] == split, "fact_split_leakage")
                source_id = facts[fact_id]["source_id"]
                require(source_id in case["document_ids"], "evidence_outside_case_scope")
                target_languages.add("zh" if sources[source_id]["language"] == "zh-Hans" else "en")
                referenced.add(fact_id)
            if split == "acceptance":
                require(len(target_languages) == 1 and case["group"] == case["question_language"] + "-to-" + next(iter(target_languages)), "language_group_mismatch")
                require(case["public_citations"] == "required_source_language_and_original_page", "missing_citation_requirement")
    require(set(facts) == referenced, "unreferenced_fact")
    groups = Counter(c["group"] for c in dataset["acceptance"])
    require(groups == {"zh-to-zh": 10, "zh-to-en": 10, "en-to-zh": 10, "en-to-en": 10}, "acceptance_group_counts")
    for group in groups:
        covered = {sources[facts[f]["source_id"]]["category"] for c in dataset["acceptance"] if c["group"] == group for f in c["fact_ids"]}
        require(covered == {"medical", "critical-illness", "life"}, "group_category_coverage")
    require(len(dataset["calibration"]) >= 6, "calibration_too_small")
    require({facts[f]["source_id"] for c in dataset["calibration"] for f in c["fact_ids"]} == set(sources), "calibration_source_coverage")
    categories = Counter()
    for case in dataset["negatives"]:
        require(isinstance(case["question"], str) and case["question"].strip(), "empty_question")
        require(case["id"] not in case_ids and compact(case["question"]) not in questions, "duplicate_negative")
        case_ids.add(case["id"])
        questions.add(compact(case["question"]))
        require(case["category"] in {"insufficient_evidence", "off_topic", "purchase_advice", "claim_guarantee"}, "invalid_negative_category")
        categories[case["category"]] += 1
        require(case["document_ids"] and set(case["document_ids"]) <= sources.keys(), "invalid_negative_scope")
        require(case["expected_answer_language"] == case["question_language"] and case["question_language"] in {"zh", "en"}, "negative_language_mismatch")
        require(case["public_citations"] == "empty" and case["required_behavior"] and case["forbidden_behavior"], "incomplete_negative_boundary")
        expected = "advice" if case["category"] in {"purchase_advice", "claim_guarantee"} else "refusal"
        require(case["expected_kind"] == expected, "wrong_negative_terminal")
        if case["category"] == "insufficient_evidence":
            require(case["review_evidence"], "missing_negative_review_basis")
        for evidence in case["review_evidence"]:
            source_id = evidence["source_id"]
            require(source_id in case["document_ids"], "negative_review_outside_scope")
            require(evidence["source_sha256"] == sources[source_id]["sha256"], "negative_source_hash_mismatch")
            evidence_checked(source_id, evidence)
    require(len(dataset["negatives"]) >= 12 and len(categories) == 4 and min(categories.values()) >= 1, "negative_coverage")
    require({c["question_language"] for c in dataset["negatives"]} == {"zh", "en"}, "negative_language_coverage")
    for case in dataset["interactions"]:
        require(case["id"] not in case_ids, "duplicate_interaction_id")
        case_ids.add(case["id"])
        require(case["execution_status"] == "not_run" and set(case["run_on"]) == {"streaming", "non_streaming"}, "interaction_execution_contract")
        require(case["document_ids"] and set(case["document_ids"]) <= sources.keys(), "invalid_interaction_scope")
        require(case["steps"] and case["required_checks"] and all(s["action"] and s["expected"] for s in case["steps"]), "missing_interaction_steps")
        require(all(f in facts and facts[f]["source_id"] in case["document_ids"] for f in case["fact_ids"]), "invalid_interaction_facts")
    require({"comparison", "followup", "scope_switch", "deletion"} <= {c["category"] for c in dataset["interactions"]}, "interaction_coverage")
    require(all((ROOT / p).is_file() for p in dataset["synthetic_regression"]["paths"]), "missing_synthetic_regression")
    return {"suite_id": dataset["suite_id"], "data_valid": True, "source_anchors_checked": pdf_dir is not None,
            "facts": len(facts), "acceptance_groups": dict(groups), "negatives": len(dataset["negatives"]),
            "calibration": len(dataset["calibration"]), "interactions": len(dataset["interactions"]),
            "real_model_acceptance": "not_run", "independent_human_review": dataset["annotation"]["independent_human_review"]}


def assess_manual_reviews(dataset: dict, reviews: list[dict]) -> dict:
    """Apply the rubric to human judgements, never infer correctness from retrieval scores."""
    expected = {c["id"]: c for c in dataset["acceptance"] + dataset["negatives"]}
    seen = set()
    successes = Counter()
    negative_successes = 0
    critical_failures = []
    for review in reviews:
        identifier = review["case_id"]
        require(identifier in expected and identifier not in seen, "unknown_or_duplicate_review")
        seen.add(identifier)
        for key in ("required_points_correct", "conditions_preserved", "answer_language_correct", "scope_correct"):
            require(type(review.get(key)) is bool, "review_requires_explicit_booleans")
        require(isinstance(review.get("critical_errors"), list) and all(isinstance(e, str) and e for e in review["critical_errors"]), "review_requires_explicit_errors")
        require(review["outcome"] in {"complete", "partial", "refusal", "advice", "unavailable", "retrieval_only"}, "invalid_review_outcome")
        require(review["citation_status"] in {"correct", "empty", "missing", "wrong", "out_of_scope"}, "invalid_review_citations")
        fatal = (bool(review["critical_errors"]) or not review["scope_correct"]
                 or not review["conditions_preserved"] or review["citation_status"] in {"wrong", "out_of_scope"})
        if fatal:
            critical_failures.append(identifier)
        correct = (not fatal and review["required_points_correct"] and review["answer_language_correct"])
        case = expected[identifier]
        if "group" in case:
            if correct and review["outcome"] == "complete" and review["citation_status"] == "correct":
                successes[case["group"]] += 1
        elif correct and review["outcome"] == case["expected_kind"] and review["citation_status"] == "empty":
            negative_successes += 1
    groups = {group: successes[group] for group in ("zh-to-zh", "zh-to-en", "en-to-zh", "en-to-en")}
    return {"complete_answer_counts": groups, "negative_successes": negative_successes,
            "missing_review_ids": sorted(expected.keys() - seen), "critical_failure_ids": critical_failures,
            "rubric_pass": (seen == expected.keys() and not critical_failures and all(n >= 9 for n in groups.values())
                            and negative_successes == len(dataset["negatives"])),
            "actual_runtime_verified": False,
            "note": "Manual judgements only; actual local-model/API evidence and interaction acceptance remain required."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--pdf-dir", type=Path, help="Also verify anchors against pinned local PDFs using the project parser")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--reviews", type=Path, help="Optional manually judged rows; rubric pass is not real-service acceptance")
    args = parser.parse_args()
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    result = check_dataset(dataset, pdf_dir=args.pdf_dir)
    if args.reviews:
        result["manual_rubric"] = assess_manual_reviews(dataset, json.loads(args.reviews.read_text(encoding="utf-8")))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        result["dataset_sha256"] = hashlib.sha256(args.dataset.read_bytes()).hexdigest()
        args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
