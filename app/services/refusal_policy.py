"""Question-boundary and refusal wording policy for insurance clause RAG."""

from __future__ import annotations

from typing import Any, Dict, List

INSURANCE_KEYWORDS = [
    "保险",
    "条款",
    "等待期",
    "免赔",
    "责任免除",
    "犹豫期",
    "理赔",
    "保额",
    "身故",
    "重疾",
    "投保",
    "保单",
    "除外",
    "酒驾",
    "自杀",
]

ADVICE_OR_GUARANTEE_MARKERS = [
    "一定能获赔",
    "保证获赔",
    "该不该买",
    "应不应该买",
    "推荐购买",
    "帮我配置",
    "今天买",
    "保证理赔",
]

CHITCHAT_OR_OFFTOPIC_MARKERS = ["天气", "北京", "上海", "你好", "讲个笑话", "股票", "足球"]


def refusal_answer(reason: str = "insufficient_evidence") -> str:
    if reason == "advice":
        return """【结论】
本系统不能给出是否购买保险、是否一定获赔的建议或承诺。

【条款依据】
演示语料仅用于说明犹豫期、等待期、责任免除等条款检索；真实核保与理赔取决于正式合同与事故事实。

【不确定/边界】
本系统不构成保险销售或理赔承诺。"""
    return """【结论】
未在已入库条款中找到充分依据，无法就该问题给出有引用支撑的结论。

【条款依据】
当前检索结果为空或相关度不足，请换用条款中的术语（如等待期、免赔额、责任免除）重试。

【不确定/边界】
本系统不构成保险销售或理赔承诺。"""


def is_advice_or_guarantee_question(question: str) -> bool:
    import re
    text = (question or "").casefold()
    return any(marker in text for marker in ADVICE_OR_GUARANTEE_MARKERS) or bool(re.search(
        r"该买|应该买|推荐.*保险|配置.*保险|保证.*获赔|should i (buy|purchase)|recommend.*(policy|insurance)|"
        r"(allocate|allocation|configure).*insurance|guarantee.*(claim|paid|payment)|which.*(buy|purchase)|which\s+(policy|insurance|plan).*choose", text))


def is_off_topic(question: str, chunks: List[Dict[str, Any]]) -> bool:
    import re
    text = (question or "").casefold()
    clause_context = any(k in text for k in INSURANCE_KEYWORDS) or bool(re.search(
        r"\b(policy|insurance|cover|coverage|insured|exclusion|claim)\b", text))
    return bool(re.search(r"讲.*笑话|\bjoke\b", text)) or (not clause_context and bool(
        re.search(r"天气|足球|股票|\b(weather|football|stock price)\b", text)))
