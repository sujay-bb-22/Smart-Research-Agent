from fastapi.testclient import TestClient

import main

client = TestClient(main.app)


def test_upload_records_user_and_workspace(monkeypatch):
    class FakeDB:
        def add_documents(self, documents):
            self.documents = documents

    monkeypatch.setattr(main.settings, "auth_required", False)
    monkeypatch.setattr(main, "load_db", lambda: None)
    monkeypatch.setattr(main, "db", FakeDB())
    monkeypatch.setattr(main, "retriever", object())
    monkeypatch.setattr(
        main,
        "_call_ingestion_with_metadata",
        lambda *args, **kwargs: [main.Document(page_content="Alpha content", metadata={"document_id": "doc-123", "filename": "alpha.pdf", "user_id": "user-1", "workspace_id": "workspace-1"})],
    )

    response = client.post(
        "/upload",
        files={"file": ("alpha.pdf", b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF", "application/pdf")},
        headers={"x-user-id": "user-1", "x-workspace-id": "workspace-1"},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    record = main._load_document_registry()[body["document_id"]]
    assert record["user_id"] == "user-1"
    assert record["workspace_id"] == "workspace-1"


def test_upload_library_and_chat_share_default_document_scope(monkeypatch, tmp_path):
    class FakeDB:
        def __init__(self):
            self.documents = []

        def add_documents(self, documents):
            self.documents.extend(documents)

    class FakeLLM:
        async def astream(self, prompt):
            yield type("Chunk", (), {"content": "Alpha is the indexed finding."})()

    fake_db = FakeDB()
    monkeypatch.setattr(main.settings, "auth_required", False)
    monkeypatch.setattr(main.settings, "vector_store_path", str(tmp_path / "db"))
    monkeypatch.setattr(main, "load_db", lambda: None)
    monkeypatch.setattr(main, "db", fake_db)
    monkeypatch.setattr(main, "retriever", object())
    monkeypatch.setattr(main, "embeddings", object())
    monkeypatch.setattr(main, "llm", FakeLLM())
    monkeypatch.setattr(main, "lexical_index", main.LexicalIndex())
    monkeypatch.setattr(
        main,
        "get_pdf_chunks",
        lambda path, document_id=None, filename=None: [
            main.Document(
                page_content="Alpha is the indexed finding.",
                metadata={"source": path, "page": 1, "document_id": document_id, "filename": filename},
            )
        ],
    )

    upload = client.post(
        "/upload",
        files={"file": ("alpha.pdf", b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF", "application/pdf")},
    )

    assert upload.status_code == 200, upload.text
    document_id = upload.json()["document_id"]
    assert upload.json()["status"] == "completed"

    library = client.get("/files")
    assert library.status_code == 200
    assert any(item["document_id"] == document_id for item in library.json()["files"])

    retrieved_scopes = []
    document = main.Document(
        page_content="Alpha is the indexed finding.",
        metadata={
            "document_id": document_id,
            "filename": "alpha.pdf",
            "page": 1,
            "user_id": main.settings.default_user_id,
            "workspace_id": main.settings.default_workspace_id,
        },
    )

    def retrieve(query, db_handle, top_k, scope_filter):
        retrieved_scopes.append(scope_filter)
        return [document], {id(document): 0.9}

    monkeypatch.setattr(main, "build_hybrid_candidate_documents", retrieve)
    answer = client.post(
        "/ask",
        json={"question": "What is the finding?", "selected_document_ids": [document_id], "history": []},
    )

    assert answer.status_code == 200, answer.text
    assert "Alpha is the indexed finding." in answer.text
    assert main._metadata_matches_filter(document.metadata, retrieved_scopes[0])


def test_validate_selected_document_ids_enforces_user_and_workspace_scope():
    registry = {
        "doc-a": {"document_id": "doc-a", "status": "completed", "user_id": "user-1", "workspace_id": "workspace-1"},
        "doc-b": {"document_id": "doc-b", "status": "completed", "user_id": "user-2", "workspace_id": "workspace-1"},
        "doc-c": {"document_id": "doc-c", "status": "completed", "user_id": "user-1", "workspace_id": "workspace-2"},
    }

    valid = main.validate_selected_document_ids(["doc-a", "doc-b", "doc-c"], registry, user_id="user-1", workspace_id="workspace-1")

    assert valid == ["doc-a"]


def test_list_files_filters_to_current_user_and_workspace(monkeypatch):
    monkeypatch.setattr(main.settings, "auth_required", False)
    main._register_document("doc-a", "alpha.pdf", "data/doc-a.pdf", file_hash="hash-a", status="completed", user_id="user-1", workspace_id="workspace-1")
    main._register_document("doc-b", "beta.pdf", "data/doc-b.pdf", file_hash="hash-b", status="completed", user_id="user-2", workspace_id="workspace-1")
    main._register_document("doc-c", "gamma.pdf", "data/doc-c.pdf", file_hash="hash-c", status="completed", user_id="user-1", workspace_id="workspace-2")

    response = client.get("/files", headers={"x-user-id": "user-1", "x-workspace-id": "workspace-1"})

    assert response.status_code == 200
    names = {file["name"] for file in response.json()["files"]}
    assert names == {"alpha.pdf"}
