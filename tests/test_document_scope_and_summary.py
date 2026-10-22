import pytest
from fastapi.testclient import TestClient

import main

client = TestClient(main.app)


def test_validate_selected_document_ids_rejects_unknown_ids():
    known = {"doc-a": {"document_id": "doc-a"}, "doc-b": {"document_id": "doc-b"}}

    valid = main.validate_selected_document_ids(["doc-a", "missing"], known)

    assert valid == ["doc-a"]


def test_validate_selected_document_ids_handles_empty_unknown_duplicate_and_malformed_input():
    known = {
        "doc-a": {"document_id": "doc-a", "status": "completed"},
        "doc-b": {"document_id": "doc-b", "status": "completed"},
    }

    assert main.validate_selected_document_ids([], known) == []
    assert main.validate_selected_document_ids(["missing-a", "missing-b"], known) == []
    assert main.validate_selected_document_ids(
        [" doc-b ", "doc-a", "doc-b", None, 12, {}], known
    ) == ["doc-b", "doc-a"]
    assert main.validate_selected_document_ids("doc-a", known) == []
    assert main.validate_selected_document_ids(12, known) == []
    assert main.validate_selected_document_ids([None, 12, {}], {}) == []
    # An empty registry is a supported legacy mode; valid IDs are still scoped
    # by their Chroma metadata later in retrieval.
    assert main.validate_selected_document_ids(["legacy-id"], {}) == ["legacy-id"]


def test_validate_selected_document_ids_preserves_legacy_missing_status_and_rejects_deleted():
    known = {
        "legacy": {"document_id": "legacy"},
        "deleted": {"document_id": "deleted", "status": "completed", "deleted": True},
        "pending": {"document_id": "pending", "status": "pending"},
    }

    assert main.validate_selected_document_ids(
        ["legacy", "deleted", "pending"], known
    ) == ["legacy"]


def test_validate_selected_document_ids_enforces_status_and_user_workspace_scope():
    known = {
        "pending": {"status": "pending", "user_id": "u1", "workspace_id": "w1"},
        "processing": {"status": "processing", "user_id": "u1", "workspace_id": "w1"},
        "completed": {"status": "completed", "user_id": "u1", "workspace_id": "w1"},
        "failed": {"status": "failed", "user_id": "u1", "workspace_id": "w1"},
        "other-user": {"status": "completed", "user_id": "u2", "workspace_id": "w1"},
        "other-workspace": {"status": "completed", "user_id": "u1", "workspace_id": "w2"},
    }

    assert main.validate_selected_document_ids(
        list(known), known, user_id="u1", workspace_id="w1"
    ) == ["completed"]


def test_register_document_defaults_to_pending(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    main._register_document("doc-new", "new.pdf", "data/doc-new.pdf")

    assert main._load_document_registry()["doc-new"]["status"] == "pending"


def test_validate_selected_document_ids_rejects_non_completed_statuses():
    known = {
        "doc-a": {"document_id": "doc-a", "status": "processing"},
        "doc-b": {"document_id": "doc-b", "status": "completed"},
        "doc-c": {"document_id": "doc-c", "status": "failed"},
    }

    valid = main.validate_selected_document_ids(["doc-a", "doc-b", "doc-c"], known)

    assert valid == ["doc-b"]


def test_unselected_retrieval_drops_explicitly_unready_registry_documents():
    ready = type("Doc", (), {"metadata": {"document_id": "ready"}})()
    pending = type("Doc", (), {"metadata": {"document_id": "pending"}})()
    legacy = type("Doc", (), {"metadata": {"document_id": "legacy"}})()

    filtered = main._filter_documents_by_ready_status(
        [ready, pending, legacy],
        {
            "ready": {"status": "completed"},
            "pending": {"status": "processing"},
            "legacy": {},
        },
    )

    assert filtered == [ready, legacy]


def test_recovery_finds_persisted_pending_jobs_and_marks_missing_files_failed(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    existing_path = tmp_path / "data" / "ready.pdf"
    existing_path.parent.mkdir(parents=True, exist_ok=True)
    existing_path.write_bytes(b"stored")
    main._register_document("pending-doc", "ready.pdf", str(existing_path), status="pending")
    main._register_document("lost-doc", "lost.pdf", str(existing_path.parent / "lost.pdf"), status="processing")

    jobs = main._recoverable_document_jobs()

    assert [job["document_id"] for job in jobs] == ["pending-doc"]
    assert main._load_document_registry()["lost-doc"]["status"] == "failed"


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

    monkeypatch.setattr(main, "llm", type("L", (), {"astream": fake_stream})())

    response = client.post(
        "/ask",
        json={
            "question": "What did document A say?",
            "selected_document_ids": ["doc-a"],
            "history": [],
        },
    )

    assert response.status_code == 200
    scope_filter = fake_db.calls[0]["filter"]
    assert main._metadata_matches_filter(
        {
            "document_id": "doc-a",
            "user_id": main.settings.default_user_id,
            "workspace_id": main.settings.default_workspace_id,
        },
        scope_filter,
    )
    assert not main._metadata_matches_filter(
        {
            "document_id": "doc-b",
            "user_id": main.settings.default_user_id,
            "workspace_id": main.settings.default_workspace_id,
        },
        scope_filter,
    )
