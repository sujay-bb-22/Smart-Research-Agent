import io
import os
import uuid
import zipfile
from xml.etree import ElementTree as ET

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

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

    if not paragraphs:
        raise DocumentParsingError("No readable content found in DOCX file")

    return [
        Document(
            page_content="\n".join(paragraphs),
            metadata={"source": docx_path, "location": "section-1"},
        )
    ]


def get_pdf_chunks(pdf_path, document_id=None, filename=None):
    if document_id is None:
        raise ValueError("document_id is required for chunk metadata and indexing")

    file_name = filename or os.path.basename(pdf_path)
    file_type = detect_document_loader(pdf_path)

    try:
        if file_type == "pdf":
            loader = PyMuPDFLoader(pdf_path)
            documents = loader.load()
        elif file_type == "docx":
            documents = load_docx_documents(pdf_path)
        else:
            raise UnsupportedDocumentError("Unsupported file type. " + SUPPORTED_FORMAT_MESSAGE)
    except IngestionError:
        raise
    except Exception as exc:
        raise DocumentParsingError(f"Unable to read the uploaded document: {exc}") from exc

    if not documents:
        raise DocumentParsingError("No content extracted from uploaded document")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    docs = splitter.split_documents(documents)
    for index, doc in enumerate(docs, start=1):
        if doc.metadata is None:
            doc.metadata = {}

        metadata = dict(doc.metadata)
        metadata["document_id"] = document_id
        metadata["filename"] = file_name
        metadata["source"] = metadata.get("source") or pdf_path
        metadata["chunk_id"] = f"{document_id}-chunk-{index:03d}"

        if "page" not in metadata and "location" not in metadata:
            metadata["location"] = "document-body"

        doc.metadata = metadata

    return docs