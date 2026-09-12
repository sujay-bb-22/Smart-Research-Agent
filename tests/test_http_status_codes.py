import os
from types import SimpleNamespace

os.environ.setdefault("GROQ_API_KEY", "test-key")

import pytest
from fastapi import status
from fastapi.testclient import TestClient

import main

client = TestClient(main.app)


class FakeDB:
    def __init__(self):
        self.deleted = False

    def add_documents(self, documents):
        return None

    def delete(self, where=None):
        return None

    def delete_collection(self):
        return None


class FakeRetriever:
    def invoke(self, query):
        return [
            SimpleNamespace(
                page_content="This is a relevant document snippet.",
                metadata={"page": 1},
            )
        ]


class FakeLLM:
    async def astream(self, prompt):
        yield SimpleNamespace(content="Answer from the document.\nSUGGESTIONS: [\"Follow-up question 1\", \"Follow-up question 2\", \"Follow-up question 3\"]")


@pytest.fixture(autouse=True)
def reset_backend_state(monkeypatch):
    original_remove = os.remove
    original_listdir = os.listdir

    main.db = None
    main.retriever = None
    main.embeddings = None
    os.makedirs("data", exist_ok=True)
    for filename in original_listdir("data"):
        original_remove(os.path.join("data", filename))

    yield

    main.db = None
    main.retriever = None
    main.embeddings = None
    for filename in original_listdir("data"):
        original_remove(os.path.join("data", filename))


def test_upload_missing_file_returns_400():
    response = client.post("/upload", data={})
    assert response.status_code == status.HTTP_400_BAD_REQUEST


def test_upload_unsupported_extension_returns_400():
    response = client.post(
        "/upload",
        files={"file": ("notes.txt", b"hello world", "text/plain")},
    )
    assert response.status_code == status.HTTP_400_BAD_REQUEST


def test_upload_invalid_mime_type_returns_400():
    response = client.post(
        "/upload",
        files={"file": ("sample.pdf", b"%PDF-1.4", "application/x-bad-mime")},
    )
    assert response.status_code == status.HTTP_400_BAD_REQUEST


def test_upload_success_returns_200(monkeypatch):
    def fake_load_db():
        main.embeddings = object()
        main.db = FakeDB()
        main.retriever = FakeRetriever()

    monkeypatch.setattr(main, "load_db", fake_load_db)
    monkeypatch.setattr(
        main,
        "get_pdf_chunks",
        lambda path: [SimpleNamespace(page_content="chunk text", metadata={"source": path, "page": 1})],
    )

    response = client.post(
        "/upload",
        files={"file": ("sample.pdf", b"%PDF-1.4", "application/pdf")},
    )

    assert response.status_code == status.HTTP_200_OK


def test_upload_server_failure_returns_500(monkeypatch):
    def fake_load_db():
        raise RuntimeError("embedding init failed")

    monkeypatch.setattr(main, "load_db", fake_load_db)
    monkeypatch.setattr(
        main,
        "get_pdf_chunks",
        lambda path: [SimpleNamespace(page_content="chunk text", metadata={"source": path, "page": 1})],
    )

    response = client.post(
        "/upload",
        files={"file": ("sample.pdf", b"%PDF-1.4", "application/pdf")},
    )

    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR


def test_delete_missing_file_returns_404():
    response = client.post("/delete_file", json={"filename": "missing.pdf"})
    assert response.status_code == status.HTTP_404_NOT_FOUND


def test_delete_success_returns_200():
    file_path = os.path.join("data", "sample.pdf")
    with open(file_path, "wb") as fh:
        fh.write(b"placeholder")
    main.db = FakeDB()

    response = client.post("/delete_file", json={"filename": "sample.pdf"})

    assert response.status_code == status.HTTP_200_OK


def test_delete_failure_returns_500(monkeypatch):
    file_path = os.path.join("data", "sample.pdf")
    with open(file_path, "wb") as fh:
        fh.write(b"placeholder")

    main.db = FakeDB()
    monkeypatch.setattr(main.os.path, "exists", lambda path: True)
    monkeypatch.setattr(main.os, "remove", lambda path: (_ for _ in ()).throw(OSError("disk fail")))

    response = client.post("/delete_file", json={"filename": "sample.pdf"})

    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR


def test_clear_success_returns_200():
    file_path = os.path.join("data", "sample.pdf")
    with open(file_path, "wb") as fh:
        fh.write(b"placeholder")
    main.db = FakeDB()

    response = client.post("/clear")

    assert response.status_code == status.HTTP_200_OK


def test_clear_failure_returns_500(monkeypatch):
    main.db = FakeDB()
    monkeypatch.setattr(main.db, "delete_collection", lambda: (_ for _ in ()).throw(RuntimeError("db failure")))

    response = client.post("/clear")

    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR


def test_ask_empty_question_returns_400():
    response = client.post("/ask", json={"question": "   ", "history": []})
    assert response.status_code == status.HTTP_400_BAD_REQUEST


def test_ask_without_documents_returns_404(monkeypatch):
    monkeypatch.setattr(main, "load_db", lambda: None)
    response = client.post("/ask", json={"question": "What is this?", "history": []})
    assert response.status_code == status.HTTP_404_NOT_FOUND


def test_ask_valid_request_returns_200_streaming(monkeypatch):
    main.retriever = FakeRetriever()
    main.llm = FakeLLM()
    response = client.post("/ask", json={"question": "What is in the document?", "history": []})
    assert response.status_code == status.HTTP_200_OK
    assert "text/event-stream" in response.headers.get("content-type", "")


def test_files_success_returns_200():
    file_path = os.path.join("data", "sample.pdf")
    with open(file_path, "wb") as fh:
        fh.write(b"placeholder")

    response = client.get("/files")

    assert response.status_code == status.HTTP_200_OK


def test_files_failure_returns_500(monkeypatch):
    monkeypatch.setattr(main.os, "listdir", lambda path: (_ for _ in ()).throw(OSError("filesystem failure")))

    response = client.get("/files")

    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
