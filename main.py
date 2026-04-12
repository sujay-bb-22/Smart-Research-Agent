from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
import shutil
import json
import asyncio
from typing import List, Optional

from ingest import get_pdf_chunks

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, replace with frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🔹 Global variables
db = None
retriever = None

# 🔹 Ensure folders exist
os.makedirs("data", exist_ok=True)
os.makedirs("db", exist_ok=True)

# 🔹 Load embeddings
embeddings = None

# 🔹 Load LLM
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.1-8b-instant"
)

# 🔹 Request format
class ChatMessage(BaseModel):
    role: str
    content: str

class QueryRequest(BaseModel):
    question: str
    history: Optional[List[ChatMessage]] = []

class DeleteRequest(BaseModel):
    filename: str


# 🔹 Helper: Load DB safely
def load_db():
    global db, retriever, embeddings

    if embeddings is None:
        print("🔄 Loading Google Gemini embeddings...")
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    try:
        print("🔄 Loading vector DB...")
        db = Chroma(
            persist_directory="db",
            embedding_function=embeddings
        )

        retriever = db.as_retriever(search_kwargs={"k": 3})

        print("✅ DB loaded successfully")

    except Exception as e:
        print("⚠️ DB not ready yet:", e)
        db = None
        retriever = None


@app.get("/")
def home():
    return {"message": "Smart Research Assistant API running"}


# 🚀 Upload + ingest
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        file_path = os.path.join("data", file.filename)

        # 🔹 Save file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        print(f"📄 Saved file: {file.filename}")

        # 🔹 Get chunks
        docs = get_pdf_chunks(file_path)
        
        if docs is None:
            return {"error": "Failed to process PDF"}

        global db, retriever, embeddings
        
        if embeddings is None:
            load_db()
        
        # 🔹 Append or Create database
        if db is not None:
            db.add_documents(docs)
            print("📦 Added new documents to existing DB")
        else:
            db = Chroma.from_documents(docs, embeddings, persist_directory="db")
            retriever = db.as_retriever(search_kwargs={"k": 3})
            print("📦 Initialized new DB with documents")

        return {"message": "PDF uploaded and processed successfully"}

    except Exception as e:
        print("❌ Upload error:", e)
        return {"error": str(e)}

@app.get("/files")
def list_files():
    """List all uploaded PDF files and their basic metadata."""
    files = []
    if os.path.exists("data"):
        for filename in os.listdir("data"):
            if filename.endswith(".pdf"):
                file_path = os.path.join("data", filename)
                files.append({
                    "name": filename,
                    "size": os.path.getsize(file_path),
                    "uploaded_at": os.path.getmtime(file_path)
                })
    return {"files": files}


@app.post("/delete_file")
def delete_file(request: DeleteRequest):
    """Delete a specific file from storage and the vector database."""
    global db, retriever
    try:
        filename = request.filename
        file_path = os.path.join("data", filename)
        
        # 1. Delete chunks from Vector DB (using source metadata)
        if db is not None:
            # Important: Filename must match what was saved in metadata during ingest
            db.delete(where={"source": file_path})
            print(f"🗑️ Removed {filename} chunks from Chroma DB")
            
        # 2. Delete the physical file
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"📄 Deleted file: {filename}")
            
        return {"message": f"Successfully deleted {filename}"}
    except Exception as e:
        print("❌ Deletion error:", e)
        return {"error": str(e)}


@app.post("/clear")
def clear_db():
    global db, retriever
    try:
        if db is not None:
            db.delete_collection()
            db = None
            retriever = None
            print("🧹 Database cleared")
        
        # Also clean up saved PDFs in data folder
        if os.path.exists("data"):
            for filename in os.listdir("data"):
                os.remove(os.path.join("data", filename))
                
        return {"message": "Database and files cleared successfully"}
    except Exception as e:
        print("❌ Error clearing database:", e)
        return {"error": str(e)}


@app.post("/ask")
async def ask_question(request: QueryRequest):
    global retriever

    if retriever is None:
        load_db()

    if retriever is None:
        return {"error": "No document uploaded yet"}

    query = request.question.strip()
    if not query:
        return {"error": "Question is empty"}

    try:
        # 🔹 Retrieve documents
        docs = retriever.invoke(query)
        sources = []
        for doc in docs:
            sources.append({
                "page": doc.metadata.get("page", "unknown"),
                "content": doc.page_content[:200]
            })

        if not docs:
            async def empty_gen():
                yield f"data: {json.dumps({'answer': 'No relevant information found.', 'sources': []})}\n\n"
            return StreamingResponse(empty_gen(), media_type="text/event-stream")

        # 🔹 Format context and history for the prompt
        context = "\n\n".join([doc.page_content for doc in docs])
        history = request.history[-10:] if request.history else []
        history_str = "\n".join([f"{m.role.capitalize()}: {m.content}" for m in history])

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
            # Send initial sources
            yield f"data: {json.dumps({'sources': sources})}\n\n"
            
            full_response = ""
            suggestions_sent = False
            
            # Stream the LLM response
            async for chunk in llm.astream(prompt):
                if chunk.content:
                    full_response += chunk.content
                    
                    # Only stream content if we haven't hit the SUGGESTIONS marker
                    if "SUGGESTIONS:" in full_response:
                        if not suggestions_sent:
                            # Stream everything before the marker
                            content_before = full_response.split("SUGGESTIONS:")[0]
                            # We can't easily "un-stream" previous chunks, so we just stop streaming new ones
                            # and wait to parse at the end.
                            pass 
                    else:
                        yield f"data: {json.dumps({'content': chunk.content})}\n\n"
            
            # Extract suggestions at the end
            if "SUGGESTIONS:" in full_response:
                try:
                    parts = full_response.split("SUGGESTIONS:")
                    main_content = parts[0].strip()
                    suggestions_str = parts[1].strip()
                    suggestions = json.loads(suggestions_str)
                    yield f"data: {json.dumps({'suggestions': suggestions})}\n\n"
                except Exception as e:
                    print("Error parsing suggestions:", e)
            
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    except Exception as e:
        print("❌ Query error:", e)
        async def err_gen():
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        return StreamingResponse(err_gen(), media_type="text/event-stream")