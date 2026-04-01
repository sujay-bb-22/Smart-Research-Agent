import os
import shutil

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


def get_pdf_chunks(pdf_path):
    try:
        print(f"📄 Starting ingestion for: {pdf_path}")

        # 🔹 Step 1: Load PDF
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        if not documents:
            raise ValueError("No content extracted from PDF")

        print(f"📑 Loaded {len(documents)} pages")

        # 🔹 Step 2: Split into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        docs = splitter.split_documents(documents)

        print(f"✂️ Split into {len(docs)} chunks")

        return docs

    except Exception as e:
        print(f"❌ Error during ingestion: {e}")
        return None