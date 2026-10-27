"""
pdf_parser.py - Robust PDF text extraction using PyMuPDF.

Extracts plain text from PDF files or byte streams with comprehensive handling
for edge cases: empty files, corrupt streams, password-protected files,
and image-only (scanned) pages.

No external APIs, cloud services, or paid subscriptions required.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

try:
    import pymupdf as fitz  # PyMuPDF 2.0+ official import
    PYMUPDF_AVAILABLE = True
except ImportError:
    try:
        import fitz  # Fallback for older PyMuPDF releases
        PYMUPDF_AVAILABLE = True
    except ImportError:
        PYMUPDF_AVAILABLE = False


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ParseResult:
    """Structured result of a PDF parse operation."""
    text: str = ""
    page_count: int = 0
    success: bool = True
    error: str = ""
    warnings: list[str] = field(default_factory=list)

    @property
    def word_count(self) -> int:
        """Calculate total words in extracted text."""
        return len(self.text.split())

    @property
    def char_count(self) -> int:
        """Calculate total characters in extracted text."""
        return len(self.text)


# ---------------------------------------------------------------------------
# Text cleaning helper
# ---------------------------------------------------------------------------

def clean_extracted_text(text: str) -> str:
    """
    Normalise raw extracted text.
    - Strips null bytes and control characters
    - Standardises line endings (CRLF / CR -> LF)
    - Collapses irregular spaces and tabs
    - Bounds excessive consecutive newlines to maximum 2
    """
    if not text:
        return ""
    text = text.replace("\x00", "")
    text = re.sub(r"\r\n|\r", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Core parser
# ---------------------------------------------------------------------------

def extract_text_from_pdf(
    source: Union[str, Path, bytes, bytearray, io.BytesIO],
    *,
    max_pages: int = 50,
    min_chars: int = 10,
) -> ParseResult:
    """
    Extract plain text from a PDF file path or byte stream.

    Parameters
    ----------
    source : str | Path | bytes | bytearray | io.BytesIO
        Path to local PDF file, raw bytes, or BytesIO stream.
    max_pages : int, default=50
        Upper bound on pages processed to safeguard memory.
    min_chars : int, default=10
        Minimum characters expected per page before flagging a warning.

    Returns
    -------
    ParseResult
        Structured container with text, counts, error details, and warnings.
    """
    if not PYMUPDF_AVAILABLE:
        return ParseResult(
            success=False,
            error="PyMuPDF is not installed. Install via: pip install pymupdf",
        )

    result = ParseResult()

    # Normalise input into bytes
    pdf_bytes: bytes | None = None

    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            return ParseResult(success=False, error=f"File not found: {path}")
        if not path.is_file():
            return ParseResult(success=False, error=f"Specified path is not a file: {path}")
        try:
            if path.stat().st_size == 0:
                return ParseResult(success=False, error="PDF file is empty (0 bytes).")
            pdf_bytes = path.read_bytes()
        except Exception as exc:
            return ParseResult(success=False, error=f"Failed to read file: {exc}")
    elif isinstance(source, (bytes, bytearray)):
        pdf_bytes = bytes(source)
    elif isinstance(source, io.BytesIO):
        source.seek(0)
        pdf_bytes = source.read()
    else:
        return ParseResult(
            success=False,
            error=f"Unsupported source type: {type(source).__name__}",
        )

    if not pdf_bytes or len(pdf_bytes.strip()) == 0:
        return ParseResult(success=False, error="PDF content is empty (0 bytes received).")

    # Open document using PyMuPDF stream
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        return ParseResult(
            success=False,
            error=f"Cannot open document (corrupt file or non-PDF binary): {exc}",
        )

    # Check password protection
    try:
        if doc.needs_pass:
            doc.close()
            return ParseResult(
                success=False,
                error="PDF is encrypted / password-protected. Please upload an unlocked PDF.",
            )
    except Exception:
        pass

    result.page_count = doc.page_count
    if result.page_count == 0:
        doc.close()
        return ParseResult(success=False, error="PDF contains 0 pages.")

    pages_to_process = min(result.page_count, max_pages)
    if pages_to_process < result.page_count:
        result.warnings.append(
            f"Processed only first {max_pages} of {result.page_count} pages."
        )

    text_parts: list[str] = []
    for page_idx in range(pages_to_process):
        try:
            page = doc.load_page(page_idx)
            page_text = page.get_text("text")
            if len(page_text.strip()) < min_chars:
                result.warnings.append(
                    f"Page {page_idx + 1} contains very few characters ({len(page_text.strip())}); "
                    "it may be scanned, image-only, or blank."
                )
            text_parts.append(page_text)
        except Exception as page_exc:
            result.warnings.append(f"Page {page_idx + 1} extraction notice: {page_exc}")

    doc.close()

    raw_text = "\n".join(text_parts)
    cleaned = clean_extracted_text(raw_text)
    result.text = cleaned

    if not cleaned:
        result.success = False
        result.error = (
            "No extractable text found. The PDF likely contains scanned images "
            "rather than digital text layers. An OCR preprocessor (e.g. Tesseract) "
            "would be required for scanned images."
        )
        return result

    return result


# ---------------------------------------------------------------------------
# Streamlit file uploader wrapper
# ---------------------------------------------------------------------------

def parse_uploaded_pdf(uploaded_file) -> ParseResult:
    """
    Parse an UploadedFile object returned by Streamlit's st.file_uploader.
    """
    if uploaded_file is None:
        return ParseResult(success=False, error="No file uploaded.")

    filename = getattr(uploaded_file, "name", "document.pdf")
    if not filename.lower().endswith(".pdf"):
        return ParseResult(
            success=False,
            error=f"Invalid file extension for '{filename}'. Please upload a .pdf file.",
        )

    try:
        uploaded_file.seek(0)
        raw_bytes = uploaded_file.read()
    except Exception as exc:
        return ParseResult(success=False, error=f"Could not read uploaded stream: {exc}")

    if not raw_bytes:
        return ParseResult(success=False, error="Uploaded file stream is empty.")

    return extract_text_from_pdf(io.BytesIO(raw_bytes))
