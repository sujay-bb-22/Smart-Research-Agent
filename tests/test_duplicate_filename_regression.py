import os
from types import SimpleNamespace

from fastapi.testclient import TestClient
from langchain_core.documents import Document
from langchain_core.embeddings import FakeEmbeddings

import main

client = TestClient(main.app)


def test_duplicate_filename_uploads_keep_distinct_storage_and_document_ids(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    os.makedirs("data", exist_ok=True)
    os.makedirs("db", exist_ok=True)
    main.db = None
    main.retriever = None
    main.embeddings = None
    monkeypatch.setattr(main, "embeddings", type("FakeEmbeddings", (), {"embed_documents": lambda self, texts: [[0.1, 0.2, 0.3] for _ in texts]})())

    def fake_get_pdf_chunks(path, document_id=None, filename=None):
        if "A" in path:
            content = "Document A content"
        else:
            content = "Document B content"

        return [
            SimpleNamespace(
                page_content=content,
                metadata={
                    "source": path,
                    "page": 1,
                    "document_id": document_id,
                    "filename": filename,
                    "chunk_id": f"{document_id}-chunk-001",
                },
            )
        ]

    monkeypatch.setattr(main, "get_pdf_chunks", fake_get_pdf_chunks)
    monkeypatch.setattr(main, "load_db", lambda: None)

    first_response = client.post(
        "/upload",
        files={"file": ("report.pdf", b"A", "application/pdf")},
    )
    second_response = client.post(
        "/upload",
        files={"file": ("report.pdf", b"B", "application/pdf")},
    )

    assert first_response.status_code == 200
    assert second_response.status_code == 200

    data_files = sorted(
        filename for filename in os.listdir("data") if filename != "documents.json"
    )
    assert len(data_files) == 2
    assert all(file.endswith(".pdf") for file in data_files)

    doc_ids = {doc.metadata["document_id"] for doc in main.db.documents}
    assert len(doc_ids) == 2
    assert all(doc_id != "report.pdf" for doc_id in doc_ids)
    assert {doc.metadata["filename"] for doc in main.db.documents} == {"report.pdf"}

    stored_contents = sorted(
        open(os.path.join("data", filename), "rb").read().decode("latin-1")
        for filename in data_files
    )
    assert stored_contents == ["A", "B"]


def test_delete_by_document_id_targets_the_selected_duplicate(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    os.makedirs("data", exist_ok=True)
    main.db = None
    main.retriever = None
    main.embeddings = None
    main._register_document("doc-a", "report.pdf", "data/doc-a.pdf", status="completed")
    main._register_document("doc-b", "report.pdf", "data/doc-b.pdf", status="completed")
    open("data/doc-a.pdf", "wb").write(b"A")
    open("data/doc-b.pdf", "wb").write(b"B")

    class DeleteDB:
        def __init__(self):
            self.deleted = []

        def delete(self, where=None):
            self.deleted.append(where)

        def get(self, **kwargs):
            return {"documents": [], "metadatas": []}

    fake_db = DeleteDB()
    main.db = fake_db
    response = client.post(
        "/delete_file",
        json={"filename": "report.pdf", "document_id": "doc-b"},
    )

    assert response.status_code == 200, response.text
    assert os.path.exists("data/doc-a.pdf")
    assert not os.path.exists("data/doc-b.pdf")
    assert "doc-b" in str(fake_db.deleted)
    assert set(main._load_document_registry()) == {"doc-a"}
