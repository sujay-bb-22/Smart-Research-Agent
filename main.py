from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from pydantic import BaseModel
from dotenv import load_dotenv
import json
import os
import shutil
from typing import List, Optional

from ingest import get_pdf_chunks, validate_upload_file

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

os.makedirs("data", exist_ok=True)
os.makedirs("db", exist_ok=True)

llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.1-8b-instant",
)


class ChatMessage(BaseModel):
    role: str
    content: str


class QueryRequest(BaseModel):
    question: str
    history: Optional[List[ChatMessage]] = None


class DeleteRequest(BaseModel):
    filename: str


def load_db():
    global db, retriever, embeddings

    try:
        if embeddings is None:
            embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

        db = Chroma(persist_directory="db", embedding_function=embeddings)
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

    file_path = os.path.join("data", file.filename)

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except OSError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to save uploaded file.",
        ) from exc

    docs = get_pdf_chunks(file_path)
    if docs is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unable to read the uploaded file.",
        )

    try:
        global db, retriever, embeddings

        if embeddings is None:
            load_db()

        if db is not None:
            db.add_documents(docs)
        else:
            db = Chroma.from_documents(docs, embeddings, persist_directory="db")
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
        if os.path.exists("data"):
            for filename in os.listdir("data"):
                if filename.lower().endswith((".pdf", ".docx")):
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

    file_path = os.path.join("data", request.filename)
    if not os.path.exists(file_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{request.filename}' not found.",
        )

    try:
        if db is not None:
            db.delete(where={"source": file_path})

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
                os.remove(os.path.join("data", filename))

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

    try:
        docs = retriever.invoke(query)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Retrieval failed before streaming started.",
        ) from exc

    sources = []
    for doc in docs:
        sources.append({
            "page": doc.metadata.get("page", "unknown"),
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
