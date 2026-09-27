from fastapi.testclient import TestClient

import main

client = TestClient(main.app)


def test_upload_records_user_and_workspace(monkeypatch):
    monkeypatch.setattr(main.settings, "auth_required", False)
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
