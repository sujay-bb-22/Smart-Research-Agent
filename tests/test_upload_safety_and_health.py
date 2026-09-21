import os

from fastapi.testclient import TestClient

import main

client = TestClient(main.app)


def test_sanitize_filename_blocks_path_traversal():
    assert main.sanitize_filename("../../etc/passwd.pdf") == "passwd.pdf"
    assert main.sanitize_filename("report.pdf") == "report.pdf"
    assert main.sanitize_filename("..\\secret\\report.docx") == "secret_report.docx"


def test_upload_rejects_empty_and_oversized_file(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    os.makedirs("data", exist_ok=True)
    os.makedirs("db", exist_ok=True)
    main.db = None
    main.retriever = None
    main.embeddings = None

    empty_response = client.post(
        "/upload",
        files={"file": ("report.pdf", b"", "application/pdf")},
    )
    assert empty_response.status_code == 400

    large_bytes = b"A" * (main.MAX_UPLOAD_BYTES + 1)
    large_response = client.post(
        "/upload",
        files={"file": ("huge.pdf", large_bytes, "application/pdf")},
    )
    assert large_response.status_code == 413


def test_health_and_ready_endpoints():
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    ready = client.get("/ready")
    assert ready.status_code in {200, 503}
    payload = ready.json()
    assert "status" in payload
