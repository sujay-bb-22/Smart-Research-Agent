from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import FakeEmbeddings


def test_chromadb_keeps_document_metadata_after_insert_and_retrieval(tmp_path):
    docs = [
        Document(
            page_content="Alpha beta gamma",
            metadata={
                "document_id": "doc-123",
                "filename": "report.pdf",
                "chunk_id": "doc-123-chunk-001",
                "page": 1,
            },
        ),
        Document(
            page_content="Delta epsilon zeta",
            metadata={
                "document_id": "doc-123",
                "filename": "report.pdf",
                "chunk_id": "doc-123-chunk-002",
                "page": 2,
            },
        ),
    ]

    db = Chroma.from_documents(
        docs,
        FakeEmbeddings(size=3),
        persist_directory=str(tmp_path / "chromadb"),
    )

    results = db.similarity_search("Alpha", k=5)
    assert results

    matching_results = [
        item for item in results if item.metadata.get("document_id") == "doc-123"
    ]
    assert matching_results

    metadata = matching_results[0].metadata
    assert metadata["document_id"] == "doc-123"
    assert metadata["filename"] == "report.pdf"
    assert "chunk_id" in metadata
    assert "page" in metadata or "location" in metadata
    assert "location" not in metadata or metadata.get("location") in (None, "")

    stored = db.get(include=["metadatas", "documents"])
    assert stored["metadatas"]
    assert any(item.get("document_id") == "doc-123" for item in stored["metadatas"])
    assert all("chunk_id" in item for item in stored["metadatas"])
