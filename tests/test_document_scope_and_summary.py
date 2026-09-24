import pytest
from fastapi.testclient import TestClient

import main

client = TestClient(main.app)


def test_validate_selected_document_ids_rejects_unknown_ids():
    known = {"doc-a": {"document_id": "doc-a"}, "doc-b": {"document_id": "doc-b"}}

    valid = main.validate_selected_document_ids(["doc-a", "missing"], known)

    assert valid == ["doc-a"]


def test_validate_selected_document_ids_rejects_non_completed_statuses():
    known = {
        "doc-a": {"document_id": "doc-a", "status": "processing"},
        "doc-b": {"document_id": "doc-b", "status": "completed"},
        "doc-c": {"document_id": "doc-c", "status": "failed"},
    }

    valid = main.validate_selected_document_ids(["doc-a", "doc-b", "doc-c"], known)

    assert valid == ["doc-b"]


def test_summary_question_detection():
    assert main.is_summary_question("Summarize this document for me") is True
    assert main.is_summary_question("What is the main conclusion?") is True
    assert main.is_summary_question("Explain the key findings") is True
    assert main.is_summary_question("Tell me about the funding numbers") is False


def test_ask_question_filters_retrieval_to_selected_documents(monkeypatch):
    class FakeDB:
        def __init__(self):
            self.calls = []

        def similarity_search(self, query, k=3, filter=None):
            self.calls.append({"query": query, "k": k, "filter": filter})
            return [
                type("Doc", (), {"page_content": "alpha answer", "metadata": {"document_id": "doc-a", "filename": "a.pdf", "page": 1}})(),
                type("Doc", (), {"page_content": "beta answer", "metadata": {"document_id": "doc-b", "filename": "b.pdf", "page": 2}})(),
            ]

    class FakeRetriever:
        def invoke(self, query):
            raise AssertionError("retriever.invoke should not be used when selected_document_ids are supplied")

    fake_db = FakeDB()
    monkeypatch.setattr(main, "db", fake_db)
    monkeypatch.setattr(main, "retriever", FakeRetriever())
    monkeypatch.setattr(main, "load_db", lambda: None)

    async def fake_stream(prompt):
        yield type("Chunk", (), {"content": "Only A is relevant."})()

    monkeypatch.setattr(main, "get_llm", lambda: type("L", (), {"astream": fake_stream})())

    response = client.post(
        "/ask",
        json={
            "question": "What did document A say?",
            "selected_document_ids": ["doc-a"],
            "history": [],
        },
    )

    assert response.status_code == 200
    assert fake_db.calls[0]["filter"] == {"document_id": {"$in": ["doc-a"]}}
