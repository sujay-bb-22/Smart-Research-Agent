import uuid
from types import SimpleNamespace

from fastapi.testclient import TestClient

import main

client = TestClient(main.app)


class FakeDB:
    def __init__(self):
        self.documents = []

    def add_documents(self, documents):
        self.documents = documents

    def delete(self, where=None):
        return None

    def delete_collection(self):
        return None


class FakeRetriever:
    def invoke(self, query):
        return [
            SimpleNamespace(
                page_content="This is a relevant document snippet.",
                metadata={"page": 1, "document_id": "doc-123"},
            )
        ]


def test_generate_document_id_is_uuid_based_and_not_filename():
    document_id = main.generate_document_id("report.pdf")

    assert isinstance(document_id, str)
    uuid.UUID(document_id)
    assert document_id != "report.pdf"


def test_upload_assigns_single_document_id_to_all_chunks(monkeypatch):
    db = FakeDB()
    monkeypatch.setattr(main, "load_db", lambda: None)
    monkeypatch.setattr(main, "db", db)
    monkeypatch.setattr(main, "retriever", FakeRetriever())

    def fake_get_pdf_chunks(path, document_id=None):
        return [
            SimpleNamespace(
                page_content="chunk one",
                metadata={"source": path, "page": 1, "document_id": document_id},
            ),
            SimpleNamespace(
                page_content="chunk two",
                metadata={"source": path, "page": 2, "document_id": document_id},
            ),
        ]

    monkeypatch.setattr(main, "get_pdf_chunks", fake_get_pdf_chunks)

    response = client.post(
        "/upload",
        files={"file": ("report.pdf", b"%PDF-1.4", "application/pdf")},
    )

    assert response.status_code == 200
    assert len(db.documents) == 2
    assert db.documents[0].metadata["document_id"] == db.documents[1].metadata["document_id"]
    assert db.documents[0].metadata["document_id"] != "report.pdf"
    uuid.UUID(db.documents[0].metadata["document_id"])


def test_different_uploaded_documents_receive_different_document_ids():
    first_document_id = main.generate_document_id("alpha.pdf")
    second_document_id = main.generate_document_id("beta.pdf")

    assert first_document_id != second_document_id
    uuid.UUID(first_document_id)
    uuid.UUID(second_document_id)
    assert first_document_id != "alpha.pdf"
    assert second_document_id != "beta.pdf"


def test_get_pdf_chunks_applies_same_document_id_to_every_split_chunk(monkeypatch):
    class FakeLoader:
        def __init__(self, path):
            self.path = path

        def load(self):
            return [
                SimpleNamespace(
                    page_content="alpha beta gamma",
                    metadata={"source": self.path, "page": 1},
                ),
                SimpleNamespace(
                    page_content="delta epsilon zeta",
                    metadata={"source": self.path, "page": 2},
                ),
            ]

    class FakeSplitter:
        def __init__(self, *args, **kwargs):
            pass

        def split_documents(self, documents):
            return [
                SimpleNamespace(
                    page_content=document.page_content,
                    metadata={**document.metadata, "document_id": "doc-123"},
                )
                for document in documents
            ]

    monkeypatch.setattr("ingest.detect_document_loader", lambda path: "pdf")
    monkeypatch.setattr("ingest.PyMuPDFLoader", FakeLoader)
    monkeypatch.setattr("ingest.RecursiveCharacterTextSplitter", FakeSplitter)

    docs = main.get_pdf_chunks("report.pdf", document_id="doc-123")

    assert len(docs) == 2
    assert {doc.metadata["document_id"] for doc in docs} == {"doc-123"}
