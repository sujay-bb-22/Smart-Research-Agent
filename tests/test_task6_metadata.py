from pathlib import Path

from fastapi.testclient import TestClient

import ingest
import main

client = TestClient(main.app)


class FakeLoader:
    def __init__(self, path):
        self.path = path

    def load(self):
        return [
            {
                "page_content": "alpha beta gamma",
                "metadata": {"source": self.path, "page": 1},
            },
            {
                "page_content": "delta epsilon zeta",
                "metadata": {"source": self.path, "page": 2},
            },
        ]


class FakeSplitter:
    def __init__(self, *args, **kwargs):
        pass

    def split_documents(self, documents):
        return [
            type("Chunk", (), {"page_content": doc["page_content"], "metadata": {**doc["metadata"], "document_id": "doc-123", "filename": "report.pdf", "chunk_id": f"doc-123-chunk-{idx:03d}"}})()
            for idx, doc in enumerate(documents, start=1)
        ]


def test_get_pdf_chunks_adds_required_metadata_to_each_chunk(monkeypatch):
    monkeypatch.setattr("ingest.detect_document_loader", lambda path: "pdf")
    monkeypatch.setattr("ingest.PyMuPDFLoader", FakeLoader)
    monkeypatch.setattr("ingest.RecursiveCharacterTextSplitter", FakeSplitter)

    docs = ingest.get_pdf_chunks("report.pdf", document_id="doc-123", filename="report.pdf")

    assert len(docs) == 2
    assert all("document_id" in doc.metadata for doc in docs)
    assert all("filename" in doc.metadata for doc in docs)
    assert all("page" in doc.metadata for doc in docs)
    assert all("chunk_id" in doc.metadata for doc in docs)
    assert {doc.metadata["document_id"] for doc in docs} == {"doc-123"}
    assert {doc.metadata["filename"] for doc in docs} == {"report.pdf"}
    assert len({doc.metadata["chunk_id"] for doc in docs}) == 2
    assert all(doc.metadata["chunk_id"].startswith("doc-123-chunk-") for doc in docs)


def test_docx_chunks_preserve_location_metadata(monkeypatch):
    xml = '''
    <w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
      <w:body>
        <w:p><w:t>Alpha</w:t></w:p>
        <w:p><w:t>Beta</w:t></w:p>
      </w:body>
    </w:document>
    '''

    tmp_path = Path("data") / "notes.docx"
    tmp_path.parent.mkdir(exist_ok=True)
    with tmp_path.open("wb") as fh:
        fh.write(b"not a real docx")

    monkeypatch.setattr("ingest.detect_document_loader", lambda path: "docx")

    original = ingest.load_docx_documents

    def fake_load_docx_documents(path):
        paragraphs = ["Alpha", "Beta"]
        return [
            type("Chunk", (), {"page_content": "\n".join(paragraphs), "metadata": {"source": path, "location": "section-1"}})()
        ]

    monkeypatch.setattr(ingest, "load_docx_documents", fake_load_docx_documents)

    docs = ingest.get_pdf_chunks(str(tmp_path), document_id="doc-456", filename="notes.docx")

    assert all("document_id" in doc.metadata for doc in docs)
    assert all("filename" in doc.metadata for doc in docs)
    assert all("location" in doc.metadata for doc in docs)
    assert all("chunk_id" in doc.metadata for doc in docs)
    assert docs[0].metadata["filename"] == "notes.docx"
    assert docs[0].metadata["location"] == "section-1"

    monkeypatch.setattr(ingest, "load_docx_documents", original)


def test_upload_flow_puts_metadata_on_every_chunk(monkeypatch):
    class FakeDB:
        def __init__(self):
            self.documents = []

        def add_documents(self, documents):
            self.documents = documents

    db = FakeDB()
    monkeypatch.setattr(main, "load_db", lambda: None)
    monkeypatch.setattr(main, "db", db)
    monkeypatch.setattr(main, "retriever", object())

    def fake_get_pdf_chunks(path, document_id=None, filename=None):
        return [
            type("Chunk", (), {"page_content": "chunk one", "metadata": {"source": path, "page": 1, "document_id": document_id, "filename": filename, "chunk_id": f"{document_id}-chunk-001"}})(),
            type("Chunk", (), {"page_content": "chunk two", "metadata": {"source": path, "page": 2, "document_id": document_id, "filename": filename, "chunk_id": f"{document_id}-chunk-002"}})(),
        ]

    monkeypatch.setattr(main, "get_pdf_chunks", fake_get_pdf_chunks)

    response = client.post(
        "/upload",
        files={"file": ("report.pdf", b"%PDF-1.4", "application/pdf")},
    )

    assert response.status_code == 200
    assert len(db.documents) == 2
    assert all("document_id" in doc.metadata for doc in db.documents)
    assert all("filename" in doc.metadata for doc in db.documents)
    assert all("chunk_id" in doc.metadata for doc in db.documents)
    assert {doc.metadata["document_id"] for doc in db.documents} == {db.documents[0].metadata["document_id"]}
    assert {doc.metadata["filename"] for doc in db.documents} == {"report.pdf"}
