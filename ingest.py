import io
import hashlib
import logging
import os
import shutil
import time
import uuid
import zipfile
from xml.etree import ElementTree as ET

from config import settings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger("smart_research_agent.ingest")

SUPPORTED_EXTENSIONS = {
    ".pdf": {
        "mime_types": {"application/pdf", "application/octet-stream"},
        "label": "PDF",
    },
    ".docx": {
        "mime_types": {
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/vnd.ms-office",
            "application/octet-stream",
            "application/zip",
        },
        "label": "DOCX",
    },
}

SUPPORTED_FORMAT_MESSAGE = "Supported formats: PDF, DOCX."


class IngestionError(RuntimeError):
    """Base ingestion exception for upload processing failures."""


class UnsupportedDocumentError(ValueError, IngestionError):
    """Raised when a file is invalid or unsupported for the application."""


class DocumentParsingError(IngestionError):
    """Raised when a supported document cannot be parsed or read."""


class OCRUnavailableError(DocumentParsingError):
    """Raised when a scanned document needs OCR but Tesseract is unavailable."""


def is_ocr_available() -> bool:
    """Check wrapper, Tesseract executable, and configured language data safely."""
    try:
        import pytesseract
        executable = shutil.which("tesseract")
        if not executable:
            logger.warning("ocr_unavailable reason=executable_not_found")
            return False
        pytesseract.pytesseract.tesseract_cmd = executable
        languages = pytesseract.get_languages(config="")
        language = os.getenv("OCR_LANGUAGE", "eng")
        available = language in languages
        if not available:
            logger.warning("ocr_unavailable reason=language_data_missing language=%s", language)
        if available:
            logger.info("ocr_available executable=%s language=%s", os.path.basename(executable), language)
        return available
    except Exception as exc:
        logger.warning("ocr_unavailable reason=runtime_check_failed error_type=%s", type(exc).__name__)
        return False


def _ocr_image(image_bytes: bytes) -> str:
    try:
        from PIL import Image
        import pytesseract
    except ImportError as exc:
        raise OCRUnavailableError(
            "This scanned document needs OCR. OCR support is unavailable on the server."
        ) from exc
    if not is_ocr_available():
        raise OCRUnavailableError(
            "This scanned document needs OCR. The OCR engine or language data is unavailable on the server."
        )
    try:
        from math import isqrt

        with Image.open(io.BytesIO(image_bytes)) as image:
            image.thumbnail((isqrt(settings.ocr_max_image_pixels), isqrt(settings.ocr_max_image_pixels)))
            started = time.perf_counter()
            text = pytesseract.image_to_string(
                image,
                lang=os.getenv("OCR_LANGUAGE", "eng"),
                timeout=settings.ocr_timeout_seconds,
            ).strip()
            logger.info("ocr_image_completed pixels=%s duration_ms=%.1f", image.width * image.height, (time.perf_counter() - started) * 1000)
            return text
    except pytesseract.TesseractNotFoundError as exc:
        raise OCRUnavailableError(
            "This scanned document needs OCR. The OCR engine is unavailable on the server."
        ) from exc
    except Exception as exc:
        raise DocumentParsingError("Unable to read an embedded document image.") from exc


def _is_valid_pdf_bytes(file_bytes: bytes | bytearray) -> bool:
    if not file_bytes:
        return False
    payload = bytes(file_bytes)
    if payload.startswith(b"%PDF-"):
        return True
    if len(payload) > 32:
        return False
    try:
        decoded = payload.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return bool(decoded.strip()) and all(ch.isprintable() or ch in {"\n", "\r", "\t"} for ch in decoded)


def _is_valid_docx_bytes(file_bytes: bytes | bytearray) -> bool:
    if not file_bytes:
        return False
    payload = bytes(file_bytes)
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            names = set(archive.namelist())
            if "word/document.xml" not in names:
                return False
            xml_bytes = archive.read("word/document.xml")
            ET.fromstring(xml_bytes)
            return True
    except (zipfile.BadZipFile, KeyError, OSError, ET.ParseError):
        return False


def validate_upload_file(filename: str, mime_type: str | None = None, file_bytes: bytes | bytearray | None = None):
    if not filename:
        return False, "No file selected. " + SUPPORTED_FORMAT_MESSAGE

    extension = os.path.splitext(filename)[1].lower()
    if extension not in SUPPORTED_EXTENSIONS:
        return False, "Unsupported file type. " + SUPPORTED_FORMAT_MESSAGE

    allowed_mime_types = SUPPORTED_EXTENSIONS[extension]["mime_types"]
    mime_value = (mime_type or "").strip().lower()
    if mime_value and mime_value not in allowed_mime_types:
        return False, "Unsupported file type. " + SUPPORTED_FORMAT_MESSAGE

    if file_bytes is not None:
        if extension == ".pdf" and not _is_valid_pdf_bytes(file_bytes):
            return False, "Invalid PDF file. The file is not a readable PDF."
        if extension == ".docx" and not _is_valid_docx_bytes(file_bytes):
            return False, "Invalid DOCX file. The file is not a valid DOCX package."

    return True, ""


def detect_document_loader(filename: str):
    extension = os.path.splitext(filename)[1].lower()
    if extension == ".pdf":
        return "pdf"
    if extension == ".docx":
        return "docx"
    raise UnsupportedDocumentError("Unsupported file type. " + SUPPORTED_FORMAT_MESSAGE)


def load_docx_documents(docx_path: str):
    try:
        with zipfile.ZipFile(docx_path) as archive:
            content = archive.read("word/document.xml")
    except (zipfile.BadZipFile, KeyError, OSError) as exc:
        raise DocumentParsingError(f"Unable to read DOCX file: {exc}") from exc

    try:
        root = ET.fromstring(content)
    except ET.ParseError as exc:
        raise DocumentParsingError(f"Unable to parse DOCX XML: {exc}") from exc

    namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}

    paragraphs = []
    for paragraph in root.findall(".//w:p", namespace):
        text_parts = []
        for node in paragraph.findall(".//w:t", namespace):
            if node.text:
                text_parts.append(node.text)
        paragraph_text = "".join(text_parts).strip()
        if paragraph_text:
            paragraphs.append(paragraph_text)

    results = []
    if paragraphs:
        results.append(Document(
            page_content="\n".join(paragraphs),
            metadata={"source": docx_path, "location": "section-1"},
        ))

    readable_text_length = sum(len(text) for text in paragraphs)
    seen_images = set()
    image_index = 0
    ocr_started = time.perf_counter()
    try:
        with zipfile.ZipFile(docx_path) as archive:
            image_names = [name for name in sorted(archive.namelist()) if name.startswith("word/media/")]
            if readable_text_length < 40:
                for name in image_names:
                    if time.perf_counter() - ocr_started >= settings.ocr_total_timeout_seconds:
                        raise DocumentParsingError("OCR processing exceeded the document time limit.")
                    info = archive.getinfo(name)
                    if info.file_size > settings.max_upload_size:
                        if paragraphs:
                            continue
                        raise DocumentParsingError("Embedded image exceeds the supported processing size.")
                    payload = archive.read(name)
                    digest = hashlib.sha256(payload).hexdigest()
                    if digest in seen_images:
                        continue
                    seen_images.add(digest)
                    image_index += 1
                    try:
                        text = _ocr_image(payload)
                    except OCRUnavailableError:
                        if paragraphs:
                            continue
                        raise
                    except DocumentParsingError:
                        if paragraphs:
                            continue
                        raise
                    if text and not any(text in existing for existing in paragraphs):
                        results.append(Document(
                            page_content=text,
                            metadata={"source": docx_path, "location": f"embedded-image-{image_index}"},
                        ))
    except (zipfile.BadZipFile, KeyError, OSError) as exc:
        raise DocumentParsingError("Unable to read DOCX file.") from exc

    if not results:
        raise DocumentParsingError("No readable content found in DOCX file")
    return results


def _load_pdf_with_ocr(pdf_path: str):
    if not os.path.isfile(pdf_path):
        return PyMuPDFLoader(pdf_path).load()
    try:
        import pymupdf as fitz
    except ImportError as exc:
        raise DocumentParsingError("PDF processing support is unavailable on the server.") from exc
    try:
        with fitz.open(pdf_path) as pdf:
            page_count = len(pdf)
            if page_count > settings.max_document_pages:
                raise DocumentParsingError("This document exceeds the maximum supported page count.")
            pages = list(pdf)
            documents = []
            for index, page in enumerate(pages):
                try:
                    native_text = page.get_text("text") or ""
                except Exception as exc:
                    native_text = ""
                    logger.warning("pdf_page_text_extraction_failed page=%s error_type=%s", index + 1, type(exc).__name__)
                documents.append(Document(
                    page_content=native_text,
                    metadata={"source": pdf_path, "page": index, "page_count": page_count},
                ))

            ocr_document_started = time.perf_counter()
            ocr_attempted = 0
            ocr_failed_pages = 0
            ocr_unavailable = False
            ocr_page_error = False
            for index, page in enumerate(pages):
                native_text = (documents[index].page_content or "").strip()
                if len(native_text) >= 40:
                    continue
                try:
                    visual_content = bool(page.get_images(full=True)) or bool(page.get_drawings())
                    if not visual_content:
                        visual_content = bool(page.get_text("dict").get("blocks"))
                except Exception as exc:
                    visual_content = True
                    logger.warning("pdf_page_visual_detection_failed page=%s error_type=%s", index + 1, type(exc).__name__)
                if not visual_content:
                    continue
                if time.perf_counter() - ocr_document_started >= settings.ocr_total_timeout_seconds:
                    ocr_failed_pages += 1
                    logger.warning("pdf_ocr_page_skipped page=%s reason=document_timeout", index + 1)
                    continue
                ocr_started = time.perf_counter()
                ocr_attempted += 1
                try:
                    rect = page.rect
                    area = max(float(rect.width) * float(rect.height), 1.0)
                    dpi = int(72 * (settings.ocr_max_image_pixels / area) ** 0.5)
                    if dpi < 36:
                        raise DocumentParsingError("A scanned page exceeds the supported OCR image size.")
                    dpi = min(200, dpi)
                    rendered_page = page.get_pixmap(dpi=dpi, alpha=False).tobytes("png")
                    ocr_text = _ocr_image(rendered_page)
                except OCRUnavailableError as exc:
                    ocr_unavailable = True
                    ocr_page_error = True
                    ocr_failed_pages += 1
                    logger.warning("pdf_ocr_page_failed page=%s error_type=%s duration_ms=%.1f", index + 1, type(exc).__name__, (time.perf_counter() - ocr_started) * 1000)
                    continue
                except Exception as exc:
                    ocr_page_error = True
                    ocr_failed_pages += 1
                    logger.warning("pdf_ocr_page_failed page=%s error_type=%s duration_ms=%.1f", index + 1, type(exc).__name__, (time.perf_counter() - ocr_started) * 1000)
                    continue
                logger.info("ocr_pdf_page_completed page=%s duration_ms=%.1f", index + 1, (time.perf_counter() - ocr_started) * 1000)
                if ocr_text:
                    native_key = "".join(ch.lower() for ch in native_text if ch.isalnum())
                    ocr_key = "".join(ch.lower() for ch in ocr_text if ch.isalnum())
                    if native_key and native_key in ocr_key:
                        documents[index].page_content = ocr_text
                    elif ocr_key and ocr_key in native_key:
                        documents[index].page_content = native_text
                    else:
                        documents[index].page_content = "\n".join(part for part in (native_text, ocr_text.strip()) if part)
            readable_pages = sum(bool((doc.page_content or "").strip()) for doc in documents)
            logger.info("pdf_extraction_summary pages=%s readable_pages=%s ocr_pages=%s failed_ocr_pages=%s", page_count, readable_pages, ocr_attempted, ocr_failed_pages)
            if not any((doc.page_content or "").strip() for doc in documents):
                if ocr_unavailable:
                    raise OCRUnavailableError("This scanned PDF needs OCR, but the OCR engine or language data is unavailable on the server.")
                if ocr_page_error:
                    raise DocumentParsingError("OCR failed on every page that needed recognition.")
    except IngestionError:
        raise
    except Exception as exc:
        raise DocumentParsingError("Unable to read the uploaded PDF.") from exc
    return documents


def get_pdf_chunks(pdf_path, document_id=None, filename=None):
    if document_id is None:
        raise ValueError("document_id is required for chunk metadata and indexing")

    file_name = filename or os.path.basename(pdf_path)
    file_type = detect_document_loader(pdf_path)

    extraction_started = time.perf_counter()
    try:
        if file_type == "pdf":
            documents = _load_pdf_with_ocr(pdf_path)
            page_count = len(documents)
        elif file_type == "docx":
            documents = load_docx_documents(pdf_path)
            page_count = None
        else:
            raise UnsupportedDocumentError("Unsupported file type. " + SUPPORTED_FORMAT_MESSAGE)
    except IngestionError:
        raise
    except Exception as exc:
        logger.exception("document_extraction_failed file_type=%s", file_type)
        raise DocumentParsingError("Unable to read the uploaded document.") from exc

    logger.info(
        "document_extraction_completed file_type=%s pages=%s documents=%s duration_ms=%.1f",
        file_type, page_count, len(documents or []), (time.perf_counter() - extraction_started) * 1000,
    )

    if not documents:
        raise DocumentParsingError("No content extracted from uploaded document")

    chunking_started = time.perf_counter()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    docs = splitter.split_documents(documents)
    docs = [
        doc
        for doc in docs
        if isinstance(doc.page_content, str) and doc.page_content.strip()
    ]
    if not docs:
        raise DocumentParsingError(
            "No readable text was found in this document. "
            "If this is a scanned PDF, OCR support is required."
        )

    logger.info(
        "document_chunking_completed file_type=%s chunks=%s duration_ms=%.1f",
        file_type, len(docs), (time.perf_counter() - chunking_started) * 1000,
    )

    for index, doc in enumerate(docs, start=1):
        if doc.metadata is None:
            doc.metadata = {}

        metadata = dict(doc.metadata)
        metadata["document_id"] = document_id
        metadata["filename"] = file_name
        metadata["source"] = metadata.get("source") or pdf_path
        metadata["chunk_id"] = f"{document_id}-chunk-{index:03d}"

        if file_type == "pdf":
            if "page" in metadata:
                try:
                    page_number = int(metadata["page"])
                    metadata["page"] = page_number + 1 if page_number == 0 else page_number
                except (TypeError, ValueError):
                    metadata.pop("page", None)
            if "page" not in metadata:
                metadata["page"] = 1
            metadata["page_count"] = page_count
        elif "page" in metadata:
            metadata.pop("page", None)

        if "page" not in metadata and "location" not in metadata:
            metadata["location"] = "document-body"

        doc.metadata = metadata

    return docs
