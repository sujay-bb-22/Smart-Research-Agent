# Smart Research Assistant

> An AI-powered document research assistant that allows users to upload PDF documents and ask context-aware questions using Retrieval-Augmented Generation (RAG).

**Live Demo:** https://smart-research-agent-psi.vercel.app/

---

## Project Overview

Smart Research Assistant is a full-stack AI-powered web application designed to make researching PDF documents faster and more interactive.

Users can upload PDF documents and ask questions based on their content. The application uses a Retrieval-Augmented Generation (RAG) pipeline to process the documents, retrieve relevant information, and generate context-based answers.

The system extracts text from uploaded PDFs, splits the content into smaller chunks, converts the chunks into vector embeddings, and stores them in a Chroma vector database. When a user asks a question, the application retrieves the most relevant document chunks and sends them to a Large Language Model to generate an answer based on the uploaded content.

---

## Features

* Upload PDF documents
* Extract text from uploaded PDFs
* Split documents into smaller text chunks
* Generate vector embeddings for document content
* Store embeddings in ChromaDB
* Ask context-aware questions about uploaded documents
* Retrieve relevant document chunks using semantic similarity search
* Generate AI-powered answers based on document context
* Stream AI responses in real time
* Display source snippets and page references
* Maintain conversation history for follow-up questions
* Generate suggested follow-up questions
* View uploaded documents
* Delete individual documents
* Clear all uploaded documents and vector data
* Export conversations as Markdown
* Export conversations as PDF reports

---

## Architecture / Workflow

The application follows a full-stack architecture with a Next.js frontend and a FastAPI backend.

```text
┌─────────────────────────────────────────────────┐
│                 Next.js Frontend                │
│                                                 │
│  PDF Upload │ Chat │ Sources │ Report Export    │
└───────────────────────┬─────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────┐
│              Next.js API Routes                 │
│                                                 │
│  /api/upload                                    │
│  /api/ask                                       │
│  /api/files                                     │
│  /api/delete_file                               │
│  /api/clear                                     │
└───────────────────────┬─────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────┐
│                FastAPI Backend                  │
│                                                 │
│  /upload │ /ask │ /files                       │
│  /delete_file │ /clear                         │
└───────────┬───────────────────────┬─────────────┘
            │                       │
            ▼                       ▼
     ┌──────────────┐       ┌──────────────┐
     │ PDF Ingestion│       │ RAG Pipeline │
     │ PyMuPDF      │       │ Retriever    │
     └──────┬───────┘       └──────┬───────┘
            │                      │
            ▼                      ▼
     ┌──────────────┐       ┌──────────────┐
     │ Text Splitter│       │ Groq LLM     │
     └──────┬───────┘       │ Llama 3.1    │
            │               └──────────────┘
            ▼
     ┌──────────────┐
     │ Gemini       │
     │ Embeddings   │
     └──────┬───────┘
            │
            ▼
     ┌──────────────┐
     │ ChromaDB     │
     └──────────────┘
```
---

## Tech Stack

### Frontend

* Next.js
* React
* TypeScript
* Tailwind CSS
* React Markdown
* Remark GFM
* Axios
* Lucide React
* jsPDF
* html2canvas

### Backend

* Python
* FastAPI
* Uvicorn
* Pydantic
* Python Dotenv

### AI and RAG

* LangChain
* LangChain Community
* LangChain Groq
* LangChain Google GenAI
* Google Gemini Embeddings
* Groq API
* Llama 3.1 8B Instant
* ChromaDB

### Document Processing

* PyMuPDF
* Recursive Character Text Splitter

---

### Frontend Setup

Navigate to the frontend directory:

```bash
cd research-frontend
```

Install the dependencies:

```bash
npm install
```

Create a `.env.local` file:

```env
NEXT_PUBLIC_API_URL=http://127.0.0.1:8000
```

---

## How to Run the Backend and Frontend

### Run the Backend

From the project root directory, activate the virtual environment and run:

```bash
uvicorn main:app --reload
```

The FastAPI backend will run at:

```text
http://127.0.0.1:8000
```

The API documentation will be available at:

```text
http://127.0.0.1:8000/docs
```

### Run the Frontend

Open another terminal, navigate to the frontend directory, and run:

```bash
cd research-frontend
npm run dev
```

The frontend will run at:

```text
http://localhost:3000
```

Open this URL in your browser to use the application.

---

## Deployment

The Smart Research Assistant frontend is deployed on Vercel.

### Live Application

[**https://smart-research-agent-psi.vercel.app/**](https://smart-research-agent-psi.vercel.app/)

The frontend communicates with the FastAPI backend through the configured API URL.

For production deployment, set the frontend environment variable to your deployed backend URL:

```env
NEXT_PUBLIC_API_URL=https://your-backend-url.com
```
