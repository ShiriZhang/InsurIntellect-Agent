"""Conservative bilingual coverage intent; explicit modes always win."""
import re

from fastapi import HTTPException


def resolve_mode(request):
    mode = request.query_mode
    if mode == "auto":
        text = request.question.casefold().strip()
        document_comparison = re.search(
            r"分别|各自|每份|每一份|各份|"
            r"\b(each|per)\s+(selected\s+)?(polic(?:y|ies)|documents?|plans?|contracts?)"
            r"\b(?!\s+(year|anniversary|term|period)\b)|"
            r"\b(between|across)\s+(these|those|the selected)\s+(policies|documents|plans|contracts)\b", text)
        comparison_wording = re.search(
            r"比较|对比|区别|差异|\b(compare|comparing|versus|vs\.?|differ|differs|differences?)\b|"
            r"\bcomparison\s+(of|between|across)\b", text)
        ordinary_explanation = re.search(
            r"解释|定义|含义|是什么意思|\b(explain|define|definition|meaning)\b|"
            r"\bwhat\b.*\bmeans?\b|\bwhat is the difference between\b", text)
        # A concept comparison (e.g. deductible vs co-payment) need not promise
        # review of every selected policy. A document-directed comparison does.
        if document_comparison:
            mode = "compare"
        elif ordinary_explanation:
            mode = "qa"
        elif comparison_wording:
            mode = "compare"
        elif len(request.document_ids) > 1 and re.search(
                r"这几份|这些|它们|两份|哪份|哪个好|\b(these|those|they|their|better)\b|"
                r"\bwhich\s+(policy|document|plan|one)\b", text):
            mode = "clarification"
        else:
            mode = "qa"
    if mode == "compare" and len(request.document_ids) > 5:
        raise HTTPException(422, {"error": "comparison_limit", "max_documents": 5})
    return mode
