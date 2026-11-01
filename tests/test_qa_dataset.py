"""Dataset contracts and rubric checks; no real-model quality claims."""
import copy
import hashlib
import json

import pytest

from scripts.check_qa_dataset import DATASET, assess_manual_reviews, check_dataset


@pytest.fixture
def dataset():
    return json.loads(DATASET.read_text(encoding="utf-8"))


def test_pinned_dataset_counts_categories_and_distinct_calibration(dataset):
    report = check_dataset(dataset)
    assert report["acceptance_groups"] == {"zh-to-zh": 10, "en-to-zh": 10, "zh-to-en": 10, "en-to-en": 10}
    assert report["facts"] == 32 and report["negatives"] == 12
    assert report["calibration"] == 12 and report["interactions"] == 10
    assert not report["source_anchors_checked"] and report["real_model_acceptance"] == "not_run"
    acceptance_facts = {f for c in dataset["acceptance"] for f in c["fact_ids"]}
    calibration_facts = {f for c in dataset["calibration"] for f in c["fact_ids"]}
    assert acceptance_facts.isdisjoint(calibration_facts)


def test_recorded_local_pdf_check_is_bound_to_the_current_dataset():
    report = json.loads((DATASET.parent / "bilingual-qa-v1.check-report.json").read_text(encoding="utf-8"))
    assert report["dataset_sha256"] == hashlib.sha256(DATASET.read_bytes()).hexdigest()
    assert report["source_anchors_checked"] and report["real_model_acceptance"] == "not_run"


@pytest.mark.parametrize("mutate,error", [
    (lambda d: d["acceptance"].pop(), "acceptance_group_counts"),
    (lambda d: d.update(corpus_manifest_sha256="changed"), "corpus_manifest_changed"),
    (lambda d: d["facts"][0].update(source_sha256="changed"), "fact_source_hash_mismatch"),
    (lambda d: d["facts"][0].update(jurisdiction="some other region"), "jurisdiction_mismatch"),
    (lambda d: d["facts"][0]["evidence"][0].update(page=201), "invalid_evidence_page"),
    (lambda d: d["facts"][0]["evidence"][0].update(page_text_sha256="changed"), "page_annotation_identity_mismatch"),
    (lambda d: d["facts"][0].update(conditions=[]), "missing_fact_annotation"),
    (lambda d: d["facts"][0].update(conditions=[""]), "empty_fact_annotation"),
    (lambda d: d["acceptance"][0].update(question=""), "empty_question"),
    (lambda d: d["facts"][0]["critical_fields"][0].update(qualifier=""), "incomplete_critical_field"),
    (lambda d: d["calibration"][0].update(fact_ids=d["acceptance"][0]["fact_ids"]), "fact_split_leakage"),
    (lambda d: d["facts"][-1].update(semantic_key=d["facts"][0]["semantic_key"]), "reused_semantic_fact"),
    (lambda d: d["facts"][-1].update(source_id=d["facts"][0]["source_id"], source_sha256=d["facts"][0]["source_sha256"], jurisdiction=d["facts"][0]["jurisdiction"], required_points=d["facts"][0]["required_points"]), "calibration_reuses_acceptance_answer"),
    (lambda d: d["acceptance"][0].update(document_ids=["en-life"]), "evidence_outside_case_scope"),
    (lambda d: d["acceptance"][0].update(group="zh-to-en"), "language_group_mismatch"),
    (lambda d: d["acceptance"][0].update(expected_answer_language="en"), "answer_language_mismatch"),
    (lambda d: d["negatives"][0].update(public_citations="required"), "incomplete_negative_boundary"),
    (lambda d: d["negatives"][0].update(review_evidence=[]), "missing_negative_review_basis"),
    (lambda d: d["interactions"][0].update(run_on=["streaming"]), "interaction_execution_contract"),
    (lambda d: d["interactions"][0].update(steps=[]), "missing_interaction_steps"),
])
def test_corrupt_or_leaking_annotations_are_rejected(dataset, mutate, error):
    mutate(dataset)
    with pytest.raises(ValueError, match=error):
        check_dataset(dataset)


def reviews_for(dataset):
    return [dict(case_id=c["id"], outcome="complete" if "group" in c else c["expected_kind"],
                 required_points_correct=True, conditions_preserved=True,
                 answer_language_correct=True, scope_correct=True,
                 citation_status="correct" if "group" in c else "empty", critical_errors=[])
            for c in dataset["acceptance"] + dataset["negatives"]]


def test_rubric_accepts_complete_manual_judgements_but_never_certifies_runtime(dataset):
    report = assess_manual_reviews(dataset, reviews_for(dataset))
    assert report["rubric_pass"] and report["negative_successes"] == 12
    assert not report["actual_runtime_verified"]


@pytest.mark.parametrize("outcome", ["retrieval_only", "partial", "refusal", "unavailable"])
def test_hits_refusals_and_partial_answers_never_count_as_complete_success(dataset, outcome):
    reviews = reviews_for(dataset)
    for row in reviews[:20]:
        row["outcome"] = outcome
    report = assess_manual_reviews(dataset, reviews)
    assert not report["rubric_pass"]
    assert report["complete_answer_counts"]["zh-to-zh"] == 0
    assert report["complete_answer_counts"]["en-to-zh"] == 0


@pytest.mark.parametrize("changes", [
    {"critical_errors": ["wrong_deadline"]}, {"scope_correct": False},
    {"conditions_preserved": False}, {"citation_status": "wrong"},
    {"citation_status": "out_of_scope"},
])
def test_nine_successes_never_excuse_a_critical_error(dataset, changes):
    reviews = reviews_for(dataset)
    reviews[0].update(changes)
    report = assess_manual_reviews(dataset, reviews)
    assert report["complete_answer_counts"]["zh-to-zh"] == 9
    assert report["critical_failure_ids"] == [reviews[0]["case_id"]]
    assert not report["rubric_pass"]


def test_missing_citation_is_not_success_and_one_incomplete_answer_per_group_is_allowed(dataset):
    reviews = reviews_for(dataset)
    used = set()
    groups = {c["id"]: c["group"] for c in dataset["acceptance"]}
    for review in reviews:
        group = groups.get(review["case_id"])
        if group and group not in used:
            review["outcome"] = "partial"
            review["citation_status"] = "missing"
            used.add(group)
    report = assess_manual_reviews(dataset, reviews)
    assert set(report["complete_answer_counts"].values()) == {9}
    assert report["rubric_pass"]


def test_all_negatives_must_pass_with_empty_public_citations(dataset):
    reviews = reviews_for(dataset)
    reviews[-1]["citation_status"] = "correct"
    report = assess_manual_reviews(dataset, reviews)
    assert report["negative_successes"] == 11 and not report["rubric_pass"]


def test_missing_duplicate_unknown_and_unjudged_rows_do_not_pass(dataset):
    reviews = reviews_for(dataset)
    assert not assess_manual_reviews(dataset, reviews[:-1])["rubric_pass"]
    with pytest.raises(ValueError, match="unknown_or_duplicate_review"):
        assess_manual_reviews(dataset, reviews + [copy.deepcopy(reviews[0])])
    reviews[0]["required_points_correct"] = "true"
    with pytest.raises(ValueError, match="review_requires_explicit_booleans"):
        assess_manual_reviews(dataset, reviews)


def test_source_mode_rejects_changed_pdf_before_reading_its_annotations(dataset, tmp_path):
    (tmp_path / "zh-medical.pdf").write_bytes(b"not the pinned official policy")
    with pytest.raises(ValueError, match="source_pdf_identity_mismatch"):
        check_dataset(dataset, pdf_dir=tmp_path)
