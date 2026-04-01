"use client";

import { useState } from "react";

type Source = {
  page: number;
  content: string;
};

type ResponseType = {
  answer: string;
  sources: Source[];
};

export default function Home() {
  const [question, setQuestion] = useState("");
  const [response, setResponse] = useState<ResponseType | null>(null);
  const [loading, setLoading] = useState(false);

  const askQuestion = async () => {
    if (!question.trim()) return;

    try {
      setLoading(true);

      // ✅ FIX: use proxy instead of direct backend call
      const res = await fetch("/api/ask", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          question: question,
        }),
      });

      if (!res.ok) {
        throw new Error("Backend error");
      }

      const data: ResponseType = await res.json();
      setResponse(data);

    } catch (error) {
      console.error(error);
      alert("Error connecting to backend");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-10">
      <h1 className="text-2xl font-bold mb-4">
        Smart Research Assistant
      </h1>

      <input
        className="border p-2 w-96"
        value={question}
        onChange={(e) => setQuestion(e.target.value)}
        placeholder="Ask something..."
      />

      <button
        onClick={askQuestion}
        className="ml-2 px-4 py-2 bg-blue-500 text-white"
      >
        Ask
      </button>

      {loading && (
        <p className="mt-4 text-blue-500">
          Thinking... (first request may take a few seconds)
        </p>
      )}

      {response && (
        <div className="mt-6">
          <h2 className="text-xl font-semibold">Answer</h2>
          <p>{response.answer}</p>

          <h3 className="mt-4 font-semibold">Sources</h3>
          {response.sources.map((s, i) => (
            <div key={i} className="mb-2">
              <b>Page {s.page}:</b> {s.content}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}