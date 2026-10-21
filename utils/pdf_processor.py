from collections.abc import Callable

from pypdf import PdfReader
from pypdf.errors import PdfReadError


class PDFExtractionError(ValueError):
    """Raised when a PDF cannot provide extractable text."""


def extract_pdf_text(
    pdf_path: str,
    progress_callback: Callable[[int, int], None] | None = None,
) -> str:
    try:
        reader = PdfReader(pdf_path)
        if reader.is_encrypted:
            raise PDFExtractionError(
                "This PDF is password-protected. Remove its password and upload it again."
            )

        total_pages = len(reader.pages)
        pages = []
        update_interval = max(1, total_pages // 100)
        for page_number, page in enumerate(reader.pages, start=1):
            pages.append(page.extract_text() or "")
            if progress_callback and (
                page_number % update_interval == 0 or page_number == total_pages
            ):
                progress_callback(page_number, total_pages)
    except PdfReadError as exc:
        raise PDFExtractionError(
            "This PDF could not be read. Check that it is a valid, unencrypted PDF."
        ) from exc

    text = "\n\n".join(page.strip() for page in pages if page.strip())
    if not text:
        raise PDFExtractionError(
            "No selectable text was found in this PDF. It may contain scanned images; "
            "image-only PDFs are not supported because OCR is not enabled."
        )
    return text
