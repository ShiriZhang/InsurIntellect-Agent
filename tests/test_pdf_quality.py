import fitz
import pytest

from app.services.pdf_quality import PDFQualityError, _unreadable_page, parse_pdf


def make_pdf(path, texts, *, scan_pages=(), logo=False):
    with fitz.open() as document:
        for number, text in enumerate(texts, 1):
            page = document.new_page()
            if text:
                font = "china-s" if any("\u3400" <= c <= "\u9fff" for c in text) else "helv"
                page.insert_text((40, 50), text, fontname=font)
            if number in scan_pages or logo:
                pixmap = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 100, 100), False)
                pixmap.clear_with(100)
                rect = fitz.Rect(20, 100, 570, 800) if number in scan_pages else fitz.Rect(10, 10, 25, 25)
                page.insert_image(rect, stream=pixmap.tobytes("png"))
        document.save(path)
    return path


@pytest.mark.parametrize("text,language", [("等待期为九十日，意外事故不适用。", "zh"),
                                         ("The waiting period is 90 days, except accidents.", "en")])
def test_source_text_blank_page_numbers_and_decorative_logo(tmp_path, text, language):
    path = make_pdf(tmp_path / 'source.pdf', [text, "", text], logo=True)
    result = parse_pdf(path)
    assert result.page_count == 3 and result.language == language
    assert [p.number for p in result.pages] == [1, 2, 3]
    assert text in result.pages[0].text and not result.pages[1].text.strip()


@pytest.mark.parametrize("texts,scan_pages,bad_pages", [
    ([""], [1], (1,)), (["Readable policy", "header", "Readable policy"], [2], (2,)),
    (["Valid text", "", ""], [2, 3], (2, 3)),
])
def test_unreadable_scan_rejects_whole_document_with_original_pages(tmp_path, texts, scan_pages, bad_pages):
    path = make_pdf(tmp_path / 'scan.pdf', texts, scan_pages=scan_pages)
    with pytest.raises(PDFQualityError) as error:
        parse_pdf(path)
    assert error.value.code == "unreadable_pages" and error.value.pages == bad_pages


@pytest.mark.parametrize("pages,accepted", [(200, True), (201, False)])
def test_page_limit_is_hard(tmp_path, pages, accepted):
    path = make_pdf(tmp_path / 'long.pdf', ["Valid policy text"] * pages)
    if accepted:
        assert parse_pdf(path).page_count == 200
    else:
        with pytest.raises(PDFQualityError, match="page_limit_exceeded"):
            parse_pdf(path)


def test_non_pdf_damaged_encrypted_and_size(tmp_path):
    path = tmp_path / 'invalid.pdf'
    path.write_bytes(b'not a PDF')
    with pytest.raises(PDFQualityError, match="not_pdf"):
        parse_pdf(path)
    path.write_bytes(b'%PDF-1.7\ntruncated')
    with pytest.raises(PDFQualityError, match="damaged_pdf"):
        parse_pdf(path)
    path = make_pdf(tmp_path / 'valid.pdf', ["Valid policy"])
    with pytest.raises(PDFQualityError, match="file_too_large"):
        parse_pdf(path, max_bytes=10)
    with fitz.open(path) as document:
        document.save(tmp_path / 'encrypted.pdf', encryption=fitz.PDF_ENCRYPT_AES_256,
                      owner_pw="owner", user_pw="reader")
    with pytest.raises(PDFQualityError, match="encrypted_pdf"):
        parse_pdf(tmp_path / 'encrypted.pdf')


def test_blank_document_is_not_evidence(tmp_path):
    with pytest.raises(PDFQualityError, match="no_readable_text"):
        parse_pdf(make_pdf(tmp_path / 'blank.pdf', ["", ""]))


@pytest.mark.parametrize("text", ["\ufffd\ufffd", "Policy \ue000 text", "\x00broken", "!@#$%"])
def test_garbled_text_is_rejected(text):
    with fitz.open() as document:
        assert _unreadable_page(document.new_page(), text)


def test_quality_failure_records_actual_garbled_page(tmp_path, monkeypatch):
    path = make_pdf(tmp_path / 'garbled.pdf', ["Valid", "Bad encoding"])
    extract = fitz.Page.get_text
    def garbled(page, option="text", *args, **kwargs):
        if page.number == 1 and option == "text":
            return "\ufffd unreadable"
        return extract(page, option, *args, **kwargs)
    monkeypatch.setattr(fitz.Page, "get_text", garbled)
    with pytest.raises(PDFQualityError) as error:
        parse_pdf(path)
    assert error.value.pages == (2,)


def test_scan_with_readable_header_is_still_rejected(tmp_path):
    path = tmp_path / 'scan-header.pdf'
    with fitz.open() as document:
        page = document.new_page()
        pixmap = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 100, 100), False)
        pixmap.clear_with(100)
        page.insert_image(page.rect, stream=pixmap.tobytes('png'), keep_proportion=False)
        page.insert_text((40, 50), 'Policy conditions and important information for the policyholder')
        document.save(path)
    with pytest.raises(PDFQualityError, match="unreadable_pages"):
        parse_pdf(path)


@pytest.mark.parametrize("header", ["", "Policy header and other readable information for the policyholder"])
def test_unreadable_vector_artwork_is_not_a_blank_page(tmp_path, header):
    path = make_pdf(tmp_path / 'outlines.pdf', ["Readable policy", header])
    with fitz.open(path) as document:
        page = document[1]
        for row in range(5):
            for col in range(10):
                page.draw_rect(fitz.Rect(30 + col * 40, 50 + row * 40,
                                         50 + col * 40, 70 + row * 40), fill=(0, 0, 0))
        document.save(tmp_path / 'outlined-body.pdf')
    with pytest.raises(PDFQualityError) as error:
        parse_pdf(tmp_path / 'outlined-body.pdf')
    assert error.value.pages == (2,)


def test_overlapping_small_images_do_not_add_up_to_a_scan(tmp_path):
    path = make_pdf(tmp_path / 'logos-source.pdf', ["Readable policy", ""])
    with fitz.open(path) as document:
        page = document[1]
        pixmap = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 100, 100), False)
        pixmap.clear_with(100)
        for _ in range(12):
            page.insert_image(fitz.Rect(100, 100, 200, 300), stream=pixmap.tobytes('png'), keep_proportion=False)
        document.save(tmp_path / 'repeated-logos.pdf')
    assert parse_pdf(tmp_path / 'repeated-logos.pdf').page_count == 2


def test_tiled_scan_cannot_masquerade_as_blank_page(tmp_path):
    path = make_pdf(tmp_path / 'tiled-source.pdf', ["Readable policy", ""])
    with fitz.open(path) as document:
        page = document[1]
        pixmap = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 100, 100), False)
        pixmap.clear_with(100)
        for row in range(3):
            for col in range(2):
                rect = fitz.Rect(col * page.rect.width / 2, row * page.rect.height / 3,
                                 (col + 1) * page.rect.width / 2, (row + 1) * page.rect.height / 3)
                page.insert_image(rect, stream=pixmap.tobytes('png'), keep_proportion=False)
        document.save(tmp_path / 'tiled-scan.pdf')
    with pytest.raises(PDFQualityError) as error:
        parse_pdf(tmp_path / 'tiled-scan.pdf')
    assert error.value.pages == (2,)
