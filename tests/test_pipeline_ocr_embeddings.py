import zipfile
from types import SimpleNamespace

import pytest

import ingest
import main


def test_docx_ocr_recovers_embedded_image_text_and_table_text(monkeypatch, tmp_path):
    path = tmp_path / "scan.docx"
    xml = b'''<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:tbl><w:tr><w:tc><w:p><w:r><w:t>Table cell</w:t></w:r></w:p></w:tc></w:tr></w:tbl></w:body></w:document>'''
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("word/document.xml", xml)
        archive.writestr("word/media/image1.png", b"image-bytes")
    monkeypatch.setattr(ingest, "_ocr_image", lambda payload: "Scanned image text")
    monkeypatch.setattr(ingest, "is_ocr_available", lambda: True)

    docs = ingest.load_docx_documents(str(path))

    assert "Table cell" in docs[0].page_content
    assert any(doc.page_content == "Scanned image text" for doc in docs)
    assert any(doc.metadata["location"].startswith("embedded-image-") for doc in docs)


def test_pdf_ocr_recovers_text_for_scanned_image_page(monkeypatch, tmp_path):
    path = tmp_path / "scan.pdf"
    path.write_bytes(b"placeholder")

    class Page:
        rect = SimpleNamespace(width=612, height=792)

        def get_images(self, full=True):
            return [(1,)]

        def get_text(self, *_):
            return ""

        def get_pixmap(self, **kwargs):
            return SimpleNamespace(tobytes=lambda _: b"rendered-image")

    class Pdf:
        def __enter__(self):
            return [Page()]

        def __exit__(self, *args):
            return False

    import pymupdf
    monkeypatch.setattr(ingest, "PyMuPDFLoader", lambda _: SimpleNamespace(load=lambda: [SimpleNamespace(page_content="", metadata={"page": 0})]))
    monkeypatch.setattr(pymupdf, "open", lambda _: Pdf())
    monkeypatch.setattr(ingest, "_ocr_image", lambda _: "Scanned PDF text recovered")
    monkeypatch.setattr(ingest, "is_ocr_available", lambda: True)

    docs = ingest._load_pdf_with_ocr(str(path))

    assert docs[0].page_content == "Scanned PDF text recovered"


def test_text_pdf_does_not_open_ocr_renderer(monkeypatch, tmp_path):
    path = tmp_path / "text.pdf"
    path.write_bytes(b"placeholder")
    class TextPage:
        def get_text(self, *_):
            return "A" * 100

        def get_images(self, full=True):
            return []

    class Pdf:
        def __enter__(self):
            return [TextPage()]

        def __exit__(self, *args):
            return False

    import pymupdf
    monkeypatch.setattr(pymupdf, "open", lambda _: Pdf())

    docs = ingest._load_pdf_with_ocr(str(path))
    assert docs[0].page_content == "A" * 100


def test_ocr_availability_requires_executable_and_language(monkeypatch):
    import pytesseract
    monkeypatch.setattr(ingest.shutil, "which", lambda name: "/usr/bin/tesseract")
    monkeypatch.setattr(pytesseract, "get_languages", lambda config="": ["eng"])
    assert ingest.is_ocr_available()
    monkeypatch.setattr(ingest.shutil, "which", lambda name: None)
    assert not ingest.is_ocr_available()
    monkeypatch.setattr(ingest.shutil, "which", lambda name: "/usr/bin/tesseract")
    monkeypatch.setattr(pytesseract, "get_languages", lambda config="": [])
    assert not ingest.is_ocr_available()


@pytest.mark.parametrize("mode", ["empty", "unavailable", "failure"])
def test_scanned_pdf_ocr_empty_unavailable_and_failure_are_reported(monkeypatch, tmp_path, mode):
    path = tmp_path / "scan.pdf"
    path.write_bytes(b"placeholder")

    class Page:
        rect = SimpleNamespace(width=612, height=792)
        def get_images(self, full=True): return [(1,)]
        def get_text(self, *_): return ""
        def get_pixmap(self, **kwargs): return SimpleNamespace(tobytes=lambda _: b"rendered")

    class Pdf:
        def __enter__(self): return [Page()]
        def __exit__(self, *args): return False

    import pymupdf
    monkeypatch.setattr(pymupdf, "open", lambda _: Pdf())
    if mode == "empty":
        monkeypatch.setattr(ingest, "_ocr_image", lambda _: "")
        docs = ingest._load_pdf_with_ocr(str(path))
        with pytest.raises(ingest.DocumentParsingError, match="No readable text"):
            ingest.get_pdf_chunks(str(path), document_id="doc-empty")
        assert docs[0].page_content == ""
    elif mode == "unavailable":
        monkeypatch.setattr(ingest, "_ocr_image", lambda _: (_ for _ in ()).throw(ingest.OCRUnavailableError("unavailable")))
        with pytest.raises(ingest.OCRUnavailableError):
            ingest._load_pdf_with_ocr(str(path))
    else:
        monkeypatch.setattr(ingest, "_ocr_image", lambda _: (_ for _ in ()).throw(ingest.DocumentParsingError("OCR failed")))
        with pytest.raises(ingest.DocumentParsingError, match="OCR failed"):
            ingest._load_pdf_with_ocr(str(path))


def test_pdf_chunks_keep_page_document_filename_metadata_and_drop_empty(monkeypatch, tmp_path):
    path = tmp_path / "mixed.pdf"
    monkeypatch.setattr(ingest, "_load_pdf_with_ocr", lambda _: [
        ingest.Document(page_content="OCR page text", metadata={"page": 1, "page_count": 2}),
        ingest.Document(page_content="  ", metadata={"page": 2, "page_count": 2}),
    ])
    docs = ingest.get_pdf_chunks(str(path), document_id="doc-42", filename="scan.pdf")
    assert len(docs) == 1
    assert docs[0].metadata["page"] == 1
    assert docs[0].metadata["page_count"] == 2
    assert docs[0].metadata["document_id"] == "doc-42"
    assert docs[0].metadata["filename"] == "scan.pdf"


def test_ocr_extracted_chunks_index_and_document_status_transitions(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir(exist_ok=True)
    document_id = "ocr-indexed-doc"
    main._register_document(document_id, "scan.pdf", "data/scan.pdf", status="pending", user_id="u1", workspace_id="w1")
    chunk = main.Document(
        page_content="Recovered scanned research text.",
        metadata={"document_id": document_id, "filename": "scan.pdf", "page": 4},
    )
    monkeypatch.setattr(main, "_call_ingestion_with_metadata", lambda *args: [chunk])
    indexed = []
    monkeypatch.setattr(main, "index_document_chunks", lambda docs, *args, **kwargs: indexed.extend(docs))
    main._process_document_upload(document_id, "data/scan.pdf", "scan.pdf", "u1", "w1")
    record = main._load_document_registry()[document_id]
    assert record["status"] == "completed"
    assert len(indexed) == 1
    assert indexed[0].page_content == "Recovered scanned research text."
    assert indexed[0].metadata["page"] == 4
    assert indexed[0].metadata["user_id"] == "u1"
    assert indexed[0].metadata["workspace_id"] == "w1"


def test_ocr_text_uses_validated_embeddings_and_existing_vector_collection(monkeypatch):
    inserted = {}

    class FakeDB:
        def add_embeddings(self, texts, embeddings, metadatas=None, ids=None):
            inserted.update(texts=texts, embeddings=embeddings, metadatas=metadatas, ids=ids)

    class FakeEmbeddings:
        def embed_documents(self, texts):
            return [[0.25, 0.75] for _ in texts]

    fake_db = FakeDB()
    monkeypatch.setattr(main, "db", fake_db)
    monkeypatch.setattr(main, "embeddings", FakeEmbeddings())
    monkeypatch.setattr(main, "ensure_vector_store_compatible", lambda *args, **kwargs: None)
    monkeypatch.setattr(main, "_write_vector_config", lambda *args, **kwargs: None)
    monkeypatch.setattr(main.settings, "embedding_dimension", 2)
    monkeypatch.setattr(main.settings, "vector_store_path", "existing-db")
    docs = main._prepare_documents_for_indexing(
        [main.Document(page_content="OCR recovered text", metadata={"page": 3})],
        "ocr-doc", "scan.pdf", "user-a", "workspace-a",
    )
    main.index_document_chunks(docs, "ocr-doc")
    assert inserted["texts"] == ["OCR recovered text"]
    assert inserted["embeddings"] == [[0.25, 0.75]]
    assert inserted["ids"] == ["ocr-doc-chunk-001"]
    assert inserted["metadatas"][0]["page"] == 3
    assert inserted["metadatas"][0]["user_id"] == "user-a"
    assert inserted["metadatas"][0]["workspace_id"] == "workspace-a"


def test_ocr_failure_marks_document_failed(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir(exist_ok=True)
    document_id = "ocr-failed-doc"
    main._register_document(document_id, "scan.pdf", "data/scan.pdf", status="pending")
    monkeypatch.setattr(main, "_call_ingestion_with_metadata", lambda *args: (_ for _ in ()).throw(ingest.OCRUnavailableError("not available")))
    main._process_document_upload(document_id, "data/scan.pdf", "scan.pdf", "u", "w")
    record = main._load_document_registry()[document_id]
    assert record["status"] == "failed"
    assert "OCR" in record["error"]


class ValidEmbedding:
    def embed_documents(self, texts):
        return [[0.1, 0.2] for _ in texts]


@pytest.mark.parametrize("vectors", [[], [[float("nan")]], [[0.1]], [["bad"]]])
def test_embedding_response_validation_rejects_bad_vectors(monkeypatch, vectors):
    monkeypatch.setattr(main.settings, "embedding_dimension", 2)
    with pytest.raises(RuntimeError, match="invalid|inconsistent|incompatible"):
        main._validated_embedding_vectors(["chunk"], SimpleNamespace(embed_documents=lambda _: vectors))


def test_embedding_response_validation_accepts_finite_vectors(monkeypatch):
    monkeypatch.setattr(main.settings, "embedding_dimension", 2)
    assert main._validated_embedding_vectors(["chunk"], ValidEmbedding()) == [[0.1, 0.2]]


def test_embedding_requests_are_batched_at_provider_limit_and_keep_order(monkeypatch):
    monkeypatch.setattr(main.settings, "embedding_dimension", None)
    calls = []

    class BatchEmbedding:
        def embed_documents(self, texts):
            calls.append(list(texts))
            return [[float(int(text))] for text in texts]

    texts = [str(index) for index in range(101)]
    vectors = main._validated_embedding_vectors(texts, BatchEmbedding())

    assert [len(batch) for batch in calls] == [100, 1]
    assert vectors == [[float(index)] for index in range(101)]


def test_embedding_initialization_requires_google_credentials(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GOOGLE_API_KEY"):
        main._build_embedding_function()
