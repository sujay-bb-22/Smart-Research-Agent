from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

from config import settings

load_dotenv()

# Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name=settings.embedding_model
)

# Load DB
db = Chroma(
    persist_directory=settings.vector_store_path,
    embedding_function=embeddings
)

retriever = db.as_retriever(search_kwargs={"k": settings.default_k})

# Groq LLM
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name=settings.model
)

# Input
query = input("Enter your question: ")

# Retrieve docs
docs = retriever.invoke(query)

# Build context
context = "\n\n".join([doc.page_content for doc in docs])

# Prompt
prompt = f"""
Answer the question based ONLY on the context below.

Context:
{context}

Question:
{query}

Answer:
"""

# LLM response
response = llm.invoke(prompt)

print("\n📌 Answer:\n")
print(response.content)

# 🔥 ADD CITATIONS
print("\n📚 Sources:\n")

for i, doc in enumerate(docs):
    page = doc.metadata.get("page", "unknown")
    print(f"Source {i+1} (Page {page}):")
    print(doc.page_content[:200], "...\n")