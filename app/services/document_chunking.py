"""Page/clause-first source slices, exact embedding budgets and stable identities."""

from dataclasses import dataclass
from hashlib import sha256
import re

from app.services.embedding_profile import digest


CLAUSE = re.compile(
    r"(?m)^[ \t]*(?:第[一二三四五六七八九十百零\d]+[条章节]|"
    r"[一二三四五六七八九十]+[、．.]|\d+(?:\.\d+)*[、．.)]?\s+|"
    r"(?:Article|Section|Clause)\s+\d+(?:\.\d+)*)", re.IGNORECASE,
)


class ChunkBudgetError(ValueError):
    pass


@dataclass(frozen=True)
class SourceChunk:
    vector_id: str
    page_number: int
    page_chunk_index: int
    chunk_index: int
    content: str
    content_hash: str
    section_title: str | None
    context_before: str
    embedding_text: str
    source_start: int
    source_end: int


def chunk_document(parsed, document_id, attempt, embedding, *, check_cancelled=lambda: None):
    """Never decode/truncate tokens: core slices concatenate back to each source page.

    The generation ceiling uses UTF-8 bytes as a conservative token upper bound,
    leaving the configured prompt/answer reserve intact. Full-request budgeting
    remains the responsibility of evidence preparation.
    """
    policy = embedding.input_policy
    if policy.generation_chunk_bytes < 1:
        raise ChunkBudgetError("generation_context_reserve_exhausted")
    chunks = []
    title = None
    previous_tail = ""
    for page in parsed.pages:
        if not page.text.strip():
            continue
        matches = list(CLAUSE.finditer(page.text))
        boundaries = sorted({0, len(page.text), *(match.start() for match in matches)})
        headings = {match.start(): match for match in matches}
        page_index = 0
        for unit_start, unit_end in zip(boundaries, boundaries[1:]):
            if unit_start in headings:
                line = page.text[unit_start:unit_end].splitlines()[0].strip()
                title = line if len(line) <= 120 else headings[unit_start].group().strip()
                previous_tail = ""
            start = unit_start
            while start < unit_end:
                check_cancelled()
                context = previous_tail[-policy.chunk_overlap:] if policy.chunk_overlap else ""

                def evidence(body):
                    return "\n".join(part for part in (title, context, body) if part)

                def fits(body):
                    value = evidence(body)
                    return (embedding.count_tokens(value) <= embedding.max_input_tokens
                            and len(value.encode("utf-8")) <= policy.generation_chunk_bytes)

                # Overlap is auxiliary; shorten it if it prevents any forward progress.
                while context and not fits(page.text[start:start + 1]):
                    context = context[len(context) // 2 + 1:]
                lo, hi = 0, min(policy.chunk_size, unit_end - start)
                while lo < hi:
                    mid = (lo + hi + 1) // 2
                    if fits(page.text[start:start + mid]):
                        lo = mid
                    else:
                        hi = mid - 1
                if not lo:
                    raise ChunkBudgetError("section_or_character_exceeds_input_budget")
                end = start + lo
                if end < unit_end:
                    # Prefer sentence/paragraph/word boundaries when there is room.
                    minimum = start + max(1, lo // 2)
                    cuts = [page.text.rfind(separator, minimum, end) + len(separator)
                            for separator in ("\n\n", "\n", "。", ";", "；", ". ", " ")]
                    cut = max(cuts)
                    if minimum < cut <= end and fits(page.text[start:cut]):
                        end = cut
                content = page.text[start:end]
                content_hash = sha256(content.encode("utf-8")).hexdigest()
                vector_id = digest([document_id, attempt, page.number, page_index, content_hash])
                chunks.append(SourceChunk(vector_id, page.number, page_index, len(chunks), content,
                                          content_hash, title, context, evidence(content), start, end))
                previous_tail = (previous_tail + content)[-policy.chunk_overlap:] if policy.chunk_overlap else ""
                page_index += 1
                start = end
    return chunks
