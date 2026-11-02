"""Resolved answer language and controlled terminal text, shared by transports."""
import re

TITLES = {"zh": ("【结论】", "【条款依据】", "【不确定/边界】"),
          "en": ("[Conclusion]", "[Clause evidence]", "[Uncertainty / boundaries]")}
MESSAGES = {
 "refusal": ("未在已入库条款中找到充分依据。", "No sufficient evidence was found in the selected documents."),
 "advice": ("不能提供购买/配置建议或保证获赔。", "I cannot recommend purchases or allocations, or guarantee claim payment."),
 "clarification": ("请明确问题、所指文档或问答方式。", "Please clarify the question, document reference, or query mode."),
 "budget": ("证据超过安全上下文预算，请缩小问题或文档范围。", "Evidence exceeds the safe context budget. Narrow the question or document scope."),
 "local_unavailable": ("本地模型或索引不可用，请检查配置及索引兼容性。", "The local model or index is unavailable. Check configuration and index compatibility."),
 "llm_unavailable": ("生成 API 不可用；尚未完成解释，仅可查看检索原文。", "The generation API is unavailable. Explanation is incomplete; retrieved source text may be viewed."),
 "timeout": ("请求超时，回答尚未完成。", "The request timed out; the answer is incomplete."),
 "cancelled": ("回答已取消或来源已删除，未完成的输出不可作为完整答案。", "The answer was cancelled or its source was deleted. Partial output is not a complete answer."),
 "degraded": ("生成服务出错，回答尚未完成。", "The generation service failed; the answer is incomplete."),
}

def resolve_language(question, explicit="auto"):
    if explicit != "auto":
        return explicit
    text = question.casefold()
    directives = list(re.finditer(r"(?:用|使用)?(中文|英文|英语|汉语)(?:回答|作答|回复|解释)|\bin\s+(english|chinese)\b", text))
    if directives:
        value = directives[-1].group(1) or directives[-1].group(2)
        return "en" if value in {"英文", "英语", "english"} else "zh"
    zh = len(re.findall(r"[\u3400-\u9fff]", text))
    en = len(re.findall(r"[a-z]+", text))
    return "en" if en > zh else "zh"

def terminal(kind, language):
    conclusion, evidence, boundary = TITLES[language]
    text = MESSAGES.get(kind, MESSAGES["degraded"])[language == "en"]
    disclaimer = ("This system does not constitute insurance sales advice or a claim promise."
                  if language == "en" else "本系统不构成保险销售或理赔承诺。")
    return f"{conclusion}\n{text}\n\n{evidence}\n{text}\n\n{boundary}\n{disclaimer}"
