from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
import shutil

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
class QueryRequest(BaseModel):
    question: str


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
def ask_question(request: QueryRequest):
    global retriever

    if retriever is None:
        load_db()

    # 🔴 FIX 1: Handle no DB
    if retriever is None:
        return {"error": "No document uploaded yet"}

    query = request.question.strip()

    # 🔴 FIX 2: Empty question check
    if not query:
        return {"error": "Question is empty"}

    try:
        # 🔹 Retrieve documents
        docs = retriever.invoke(query)

        if not docs:
            return {"answer": "No relevant information found.", "sources": []}

        context = "\n\n".join([doc.page_content for doc in docs])

        prompt = f"""
        Answer the question based ONLY on the context below.

        Context:
        {context}

        Question:
        {query}

        Answer:
        """

        response = llm.invoke(prompt)

        # 🔹 Prepare sources
        sources = []
        for doc in docs:
            sources.append({
                "page": doc.metadata.get("page", "unknown"),
                "content": doc.page_content[:200]
            })

        return {
            "answer": response.content,
            "sources": sources
        }

    except Exception as e:
        print("❌ Query error:", e)
        return {"error": str(e)}