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
