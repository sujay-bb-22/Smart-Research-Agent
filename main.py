from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI()

# Load embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# Load DB
db = Chroma(
    persist_directory="db",
    embedding_function=embeddings
)

retriever = db.as_retriever(search_kwargs={"k": 3})

# Load LLM
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.1-8b-instant"
)

# Request format
class QueryRequest(BaseModel):
    question: str


@app.get("/")
def home():
    return {"message": "Smart Research Assistant API running"}


@app.post("/ask")
def ask_question(request: QueryRequest):
    query = request.question

    # Retrieve documents
    docs = retriever.invoke(query)

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

    # Prepare sources
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