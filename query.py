from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

# Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# Load DB
db = Chroma(
    persist_directory="db",
    embedding_function=embeddings
)

retriever = db.as_retriever(search_kwargs={"k": 3})

# Groq LLM
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.1-8b-instant"
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