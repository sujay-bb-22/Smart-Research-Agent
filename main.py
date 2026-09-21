import json
import os
import shutil
import time
import uuid
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import FakeEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from pydantic import BaseModel

from ingest import (
    DocumentParsingError,
    UnsupportedDocumentError,
    get_pdf_chunks,
    validate_upload_file,
)

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
db = None
retriever = None
embeddings = None
llm = None

os.makedirs("data", exist_ok=True)
os.makedirs("db", exist_ok=True)


def get_llm():
    global llm

    if llm is not None:
        return llm

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not configured.")

    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="openai/gpt-oss-20b",
    )
    return llm


class ChatMessage(BaseModel):
    role: str
    content: str


class QueryRequest(BaseModel):
    question: str
    history: Optional[List[ChatMessage]] = None


class DeleteRequest(BaseModel):
    filename: str


DOCUMENT_REGISTRY_PATH = os.path.join("data", "documents.json")


def generate_document_id(filename: str | None = None) -> str:
    return str(uuid.uuid4())


def _call_ingestion_with_metadata(file_path: str, document_id: str, filename: str):
    ingestion_func = get_pdf_chunks
    try:
        import inspect
        params = inspect.signature(ingestion_func).parameters
        accepts_document_id = "document_id" in params
        accepts_filename = "filename" in params
    except (TypeError, ValueError):
        accepts_document_id = False
        accepts_filename = False

    if accepts_document_id and accepts_filename:
        return ingestion_func(file_path, document_id=document_id, filename=filename)
    if accepts_document_id:
        return ingestion_func(file_path, document_id=document_id)
    return ingestion_func(file_path)


def _normalize_documents_for_chroma(docs):
    normalized = []
    for doc in docs or []:
        if isinstance(doc, Document):
            normalized.append(doc)
            continue

        page_content = getattr(doc, "page_content", None)
        metadata = getattr(doc, "metadata", {}) or {}
        if page_content is None:
            raise TypeError("Chunk is missing page_content and cannot be indexed.")

        normalized.append(Document(page_content=page_content, metadata=dict(metadata)))

    return normalized


def _load_document_registry():
    if not os.path.exists(DOCUMENT_REGISTRY_PATH):
        return {}

    try:
        with open(DOCUMENT_REGISTRY_PATH, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _save_document_registry(registry):
    os.makedirs("data", exist_ok=True)
    with open(DOCUMENT_REGISTRY_PATH, "w", encoding="utf-8") as handle:
        json.dump(registry, handle, indent=2, sort_keys=True)


def _register_document(document_id: str, filename: str, physical_path: str):
    registry = _load_document_registry()
    registry[document_id] = {
        "document_id": document_id,
        "filename": filename,
        "path": physical_path,
        "uploaded_at": time.time(),
    }
    _save_document_registry(registry)
    return registry


def _find_document_record_by_filename(filename: str):
    if not filename:
        return None

    registry = _load_document_registry()
    for record in registry.values():
        if record.get("filename") == filename:
            return record
    return None


def _remove_document_registry_entry(document_id: str):
    registry = _load_document_registry()
    registry.pop(document_id, None)
    _save_document_registry(registry)


def load_db():
    global db, retriever, embeddings

    try:
        if embeddings is None:
            google_api_key = os.getenv("GOOGLE_API_KEY")
            if google_api_key:
                embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
            else:
                embeddings = FakeEmbeddings(size=3)

        db = Chroma(persist_directory="db", embedding_function=embeddings)
        if isinstance(db, Chroma):
            db.documents = []
        retriever = db.as_retriever(search_kwargs={"k": 3})
    except Exception as exc:
        db = None
        retriever = None
        raise RuntimeError(f"Database initialization failed: {exc}") from exc


@app.get("/")
def home():
    return {"message": "Smart Research Assistant API running"}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(None)):
    if file is None or file.filename is None or not file.filename.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No file selected. Supported formats: PDF, DOCX.",
        )

    is_valid, validation_message = validate_upload_file(file.filename, file.content_type)
    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=validation_message,
        )

    document_id = generate_document_id(file.filename)
    extension = os.path.splitext(file.filename)[1].lower()
    safe_storage_name = f"{document_id}{extension}"
    file_path = os.path.join("data", safe_storage_name)

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except OSError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to save uploaded file.",
        ) from exc

    try:
        docs = _call_ingestion_with_metadata(file_path, document_id, file.filename)
    except UnsupportedDocumentError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except DocumentParsingError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to process uploaded file.",
        ) from exc

    _register_document(document_id, file.filename, file_path)

    try:
        global db, retriever, embeddings

        if embeddings is None:
            load_db()

        normalized_docs = _normalize_documents_for_chroma(docs)

        if db is not None:
            db.add_documents(normalized_docs)
            if isinstance(db, Chroma):
                prior_documents = list(getattr(db, "documents", []))
                db.documents = prior_documents + list(normalized_docs)
        else:
            db = Chroma.from_documents(normalized_docs, embeddings, persist_directory="db")
            if isinstance(db, Chroma):
                db.documents = list(normalized_docs)
            retriever = db.as_retriever(search_kwargs={"k": 3})
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Document indexing failed.",
        ) from exc

    return {"message": "Document uploaded and processed successfully"}


@app.get("/files")
def list_files():
    """List uploaded files currently stored in the data directory."""
    try:
        files = []
        registry = _load_document_registry()

        if registry:
            for record in registry.values():
                file_path = record.get("path") or os.path.join("data", os.path.basename(record.get("filename", "")))
                if not os.path.exists(file_path):
                    continue
                files.append({
                    "document_id": record.get("document_id"),
                    "name": record.get("filename", os.path.basename(file_path)),
                    "size": os.path.getsize(file_path),
                    "uploaded_at": record.get("uploaded_at", os.path.getmtime(file_path)),
                })

        if os.path.exists("data"):
            for filename in os.listdir("data"):
                if filename.lower().endswith((".pdf", ".docx")) and filename != "documents.json":
                    if any(entry.get("path") == os.path.join("data", filename) for entry in registry.values()):
                        continue
                    file_path = os.path.join("data", filename)
                    files.append({
                        "name": filename,
                        "size": os.path.getsize(file_path),
                        "uploaded_at": os.path.getmtime(file_path),
                    })

        return {"files": files}
    except OSError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to list files.",
        ) from exc


@app.post("/delete_file")
def delete_file(request: DeleteRequest):
    global db

    if not request.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing filename.",
        )

    record = _find_document_record_by_filename(request.filename)
    file_path = None
    if record is not None:
        file_path = record.get("path")
    else:
        file_path = os.path.join("data", request.filename)

    if file_path is None or not os.path.exists(file_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{request.filename}' not found.",
        )

    try:
        if db is not None:
            doc_id = None
            if record is not None:
                doc_id = record.get("document_id")
            if doc_id is not None:
                db.delete(where={"document_id": doc_id})
            else:
                db.delete(where={"source": file_path})

        if record is not None:
            _remove_document_registry_entry(record["document_id"])

        os.remove(file_path)
        return {"message": f"Successfully deleted {request.filename}"}
    except OSError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unable to delete '{request.filename}'.",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unexpected deletion failure.",
        ) from exc


@app.post("/clear")
def clear_db():
    global db, retriever

    try:
        if db is not None:
            db.delete_collection()
            db = None
            retriever = None

        if os.path.exists("data"):
            for filename in os.listdir("data"):
                if filename == "documents.json":
                    continue
                os.remove(os.path.join("data", filename))

        if os.path.exists(DOCUMENT_REGISTRY_PATH):
            os.remove(DOCUMENT_REGISTRY_PATH)

        return {"message": "Database and files cleared successfully"}
    except OSError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to clear files.",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unexpected database clear failure.",
        ) from exc


@app.post("/ask")
async def ask_question(request: QueryRequest):
    global retriever

    query = (request.question or "").strip()
    if not query:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question is empty.",
        )

    if retriever is None:
        try:
            load_db()
        except RuntimeError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Document retrieval is unavailable.",
            ) from exc

    if retriever is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No document uploaded yet.",
        )

    if llm is None:
        try:
            get_llm()
        except RuntimeError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="LLM is unavailable.",
            ) from exc

    try:
        docs = retriever.invoke(query)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Retrieval failed before streaming started.",
        ) from exc

    sources = []
    for doc in docs:
        source_location = doc.metadata.get("page")
        if source_location is None:
            source_location = doc.metadata.get("location", "unknown")
        sources.append({
            "page": source_location,
            "content": doc.page_content[:200],
        })

    if not docs:
        async def empty_gen():
            yield f"data: {json.dumps({'answer': 'No relevant information found.', 'sources': []})}\n\n"
        return StreamingResponse(empty_gen(), media_type="text/event-stream")

    history = (request.history or [])[-10:]
    history_str = "\n".join([f"{m.role.capitalize()}: {m.content}" for m in history])
    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
    You are a helpful research assistant. Answer the question based ONLY on the provided context.
    If the answer is not in the context, say you don't know based on the document.
    Use the chat history below for context when answering follow-up questions.

    Context Information:
    ---------------------
    {context}
    ---------------------

    Chat History:
    {history_str}

    Current Question: {query}

    Detailed Answer:
    (After your answer, provide 3 suggested follow-up questions in this EXACT format: SUGGESTIONS: ["Question 1", "Question 2", "Question 3"])
    """

    async def event_generator():
        yield f"data: {json.dumps({'sources': sources})}\n\n"
        full_response = ""

        try:
            async for chunk in llm.astream(prompt):
                if chunk.content:
                    full_response += chunk.content
                    yield f"data: {json.dumps({'content': chunk.content})}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"
            return

        if "SUGGESTIONS:" in full_response:
            try:
                suggestions_str = full_response.split("SUGGESTIONS:", 1)[1].strip()
                suggestions = json.loads(suggestions_str)
                yield f"data: {json.dumps({'suggestions': suggestions})}\n\n"
            except Exception:
                pass

        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
 