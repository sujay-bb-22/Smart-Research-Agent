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
