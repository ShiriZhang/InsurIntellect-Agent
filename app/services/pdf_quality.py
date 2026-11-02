"""Whole-document PDF quality gate shared by ingestion and corpus inspection."""

from dataclasses import dataclass
from pathlib import Path
import unicodedata

from app.core.config import settings


class PDFQualityError(ValueError):
    def __init__(self, code, pages=()):
        self.code = code
        self.pages = tuple(pages)
        super().__init__(code + (": pages " + ",".join(map(str, pages)) if pages else ""))


@dataclass(frozen=True)
class PDFPage:
    number: int
    text: str


@dataclass(frozen=True)
class ParsedPDF:
    pages: tuple[PDFPage, ...]
    language: str

    @property
    def page_count(self):
        return len(self.pages)


def _union_area(rectangles):
    """Count tiled/overlapping raster regions once, so repeated logos don't add up."""
    xs = sorted({x for rect in rectangles for x in (rect.x0, rect.x1)})
    area = 0
    for left, right in zip(xs, xs[1:]):
        intervals = sorted((rect.y0, rect.y1) for rect in rectangles
                           if rect.x0 < right and rect.x1 > left)
        end = float('-inf')
        height = 0
        for low, high in intervals:
            height += max(0, high - max(low, end))
            end = max(end, high)
        area += (right - left) * height
    return area


def _unreadable_page(page, text):
    bad = sum(c == "\ufffd" or unicodedata.category(c) in {"Co", "Cs", "Cn"}
              or (unicodedata.category(c) == "Cc" and not c.isspace()) for c in text)
    if bad or (text.strip() and not any(c.isalnum() for c in text)):
        return True
    # Large raster areas without a usable text layer are conservatively rejected.
    # Tiny logos and blank pages are allowed; an image alone is not a scan signal.
    area = max(1, page.rect.get_area())
    blocks = [block for block in page.get_text("blocks") if block[6] == 0]
    import fitz

    def lacks_text_layer(rect):
        readable_inside = sum(len(block[4].strip()) for block in blocks
                              if (fitz.Rect(block[:4]) & rect).get_area() > 0)
        text_area = sum((fitz.Rect(block[:4]) & rect).get_area() for block in blocks)
        return readable_inside < 40 or text_area / max(1, rect.get_area()) < 0.05

    unreadable_images = []
    for image in page.get_image_info():
        rect = fitz.Rect(image["bbox"]) & page.rect
        if rect.get_area() > 0 and lacks_text_layer(rect):
            unreadable_images.append(rect)
    if _union_area(unreadable_images) / area >= 0.20:
        return True
    # Filled vector artwork can contain unreadable outlines, even with a header.
    filled = []
    for drawing in page.get_drawings():
        fill = drawing.get("fill")
        rect = drawing["rect"] & page.rect
        if (fill is None or all(channel >= .98 for channel in fill)
                or not lacks_text_layer(rect)):
            continue
        filled.append(rect)
        if rect.get_area() / area >= 0.20:
            return True
    if len(filled) >= 20 and _union_area(filled) / area >= 0.01:
        return True
    return False


def parse_pdf(path, *, max_bytes=None, check_cancelled=lambda: None):
    """Return all original page numbers, or reject the entire document (no OCR)."""
    import fitz

    path = Path(path)
    limit = settings.UPLOAD_MAX_BYTES if max_bytes is None else max_bytes
    if path.stat().st_size > limit:
        raise PDFQualityError("file_too_large")
    with path.open("rb") as source:
        if not source.read(1024).lstrip().startswith(b"%PDF-"):
            raise PDFQualityError("not_pdf")
    check_cancelled()
    try:
        document = fitz.open(path)
    except (RuntimeError, ValueError) as exc:
        raise PDFQualityError("damaged_pdf") from exc
    with document:
        if document.needs_pass or document.is_encrypted or document.metadata.get("encryption"):
            raise PDFQualityError("encrypted_pdf")
        if document.is_repaired:
            raise PDFQualityError("damaged_pdf")
        if len(document) > 200:
            raise PDFQualityError("page_limit_exceeded")
        if not len(document):
            raise PDFQualityError("empty_pdf")
        pages, failures = [], []
        for number in range(1, len(document) + 1):
            check_cancelled()
            try:
                page = document[number - 1]
                text = page.get_text("text")
                if _unreadable_page(page, text):
                    failures.append(number)
                pages.append(PDFPage(number, text))
            except (RuntimeError, ValueError):
                failures.append(number)
        if failures:
            raise PDFQualityError("unreadable_pages", failures)
        text = "".join(page.text for page in pages)
        if not text.strip():
            raise PDFQualityError("no_readable_text")
        cjk = sum("\u3400" <= c <= "\u9fff" for c in text)
        latin = sum(c.isascii() and c.isalpha() for c in text)
        language = "mixed" if cjk and latin > cjk else "zh" if cjk else "en" if latin else "unknown"
        return ParsedPDF(tuple(pages), language)
