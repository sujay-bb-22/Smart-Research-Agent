import pytest

from ingest import detect_document_loader, validate_upload_file


def test_supported_file_formats_are_accepted():
    is_valid, message = validate_upload_file(
        "sample.pdf",
        "application/pdf",
    )
    assert is_valid is True
    assert "Supported formats" not in message

    is_valid, message = validate_upload_file(
        "sample.docx",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )
    assert is_valid is True
    assert "Supported formats" not in message


def test_unsupported_file_formats_are_rejected():
    is_valid, message = validate_upload_file("notes.txt", "text/plain")
    assert is_valid is False
    assert "Supported formats: PDF, DOCX" in message

    is_valid, message = validate_upload_file("image.png", "image/png")
    assert is_valid is False
    assert "Supported formats: PDF, DOCX" in message


def test_loader_detection_uses_correct_document_type():
    assert detect_document_loader("sample.pdf") == "pdf"
    assert detect_document_loader("sample.docx") == "docx"

    with pytest.raises(ValueError):
        detect_document_loader("sample.txt")
