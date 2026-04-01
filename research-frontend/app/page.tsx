"use client";

import { useState } from "react";

type Source = {
  page: number;
  content: string;
};

type ResponseType = {
  answer?: string;
  sources?: Source[];
  error?: string;
};

export default function Home() {
  const [question, setQuestion] = useState("");
  const [response, setResponse] = useState<ResponseType | null>(null);
  const [loading, setLoading] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [uploadStatus, setUploadStatus] = useState("");

  // 🔹 Ask question
  const askQuestion = async () => {
    if (!question.trim()) return;

    try {
      setLoading(true);

      const res = await fetch("/api/ask", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ question }),
      });

      const data = await res.json();
      setResponse(data);
    } catch (error) {
      console.error(error);
      alert("Error fetching answer");
    } finally {
      setLoading(false);
    }
  };

  // 🔹 Upload PDF
  const uploadPDF = async () => {
    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);

    try {
      setUploadStatus("Uploading...");

      const res = await fetch("/api/upload", {
        method: "POST",
        body: formData,
      });

      const data = await res.json();

      if (res.ok) {
        setUploadStatus("✅ Uploaded successfully!");
      } else {
        setUploadStatus("❌ Upload failed");
      }
    } catch (error) {
      console.error(error);
      setUploadStatus("❌ Error uploading file");
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 text-gray-900 p-8 md:p-12 font-sans selection:bg-blue-200">
      <div className="max-w-5xl mx-auto">
        <header className="mb-10 text-center md:text-left">
          <h1 className="text-4xl md:text-5xl font-extrabold tracking-tight text-gray-900 mb-3">
            Smart Research Assistant <span className="text-blue-600">🚀</span>
          </h1>
          <p className="text-gray-500 text-lg md:text-xl max-w-2xl">Upload your documents and ask intelligent questions to instantly extract insights.</p>
        </header>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          <div className="lg:col-span-1 space-y-6">
            {/* 🔹 Upload Section */}
            <div className="bg-white p-6 rounded-2xl shadow-sm border border-gray-100 transition-all hover:shadow-md">
              <h2 className="text-lg font-bold text-gray-800 mb-4 flex items-center">
                <svg className="w-5 h-5 mr-2 text-blue-500" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" /></svg>
                Document Upload
              </h2>

              <div className="mb-5">
                <label className="block text-sm font-medium text-gray-700 mb-2">Select PDF File</label>
                <div className="relative">
                  <input
                    type="file"
                    onChange={(e) => setFile(e.target.files?.[0] || null)}
                    className="block w-full text-sm text-gray-500 file:mr-4 file:py-2.5 file:px-4 file:rounded-xl file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100 transition-colors cursor-pointer border border-gray-200 rounded-xl"
                  />
                </div>
              </div>

              <button
                onClick={uploadPDF}
                disabled={!file}
                className={`w-full font-medium py-3 rounded-xl transition-all shadow-sm flex justify-center items-center ${file ? 'bg-gradient-to-r from-blue-600 to-blue-500 text-white hover:from-blue-700 hover:to-blue-600 hover:shadow-md transform hover:-translate-y-0.5' : 'bg-gray-100 text-gray-400 cursor-not-allowed'}`}
              >
                <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" /></svg>
                Upload Document
              </button>

              {uploadStatus && (
                <div className={`mt-4 text-sm font-medium p-3 rounded-xl flex items-center ${uploadStatus.includes('success') ? 'bg-green-50 text-green-700 border border-green-100' : uploadStatus.includes('Error') || uploadStatus.includes('failed') ? 'bg-red-50 text-red-700 border border-red-100' : 'bg-blue-50 text-blue-700 border border-blue-100'}`}>
                  {uploadStatus.includes('success') && <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" /></svg>}
                  {uploadStatus}
                </div>
              )}
            </div>
          </div>

          <div className="lg:col-span-2 space-y-6">
            {/* 🔹 Question Section */}
            <div className="bg-white p-6 md:p-8 rounded-2xl shadow-sm border border-gray-100 transition-all hover:shadow-md">
              <h2 className="text-lg font-bold text-gray-800 mb-4 flex items-center">
                <svg className="w-5 h-5 mr-2 text-purple-500" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8.228 9c.549-1.165 2.03-2 3.772-2 2.21 0 4 1.343 4 3 0 1.4-1.278 2.575-3.006 2.907-.542.104-.994.54-.994 1.093m0 3h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>
                Ask Questions
              </h2>

              <div className="relative">
                <input
                  className="w-full bg-gray-50 border border-gray-200 text-gray-900 rounded-xl p-4 pr-32 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent transition-all shadow-sm"
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                  placeholder="What would you like to know from the document?"
                  onKeyDown={(e) => e.key === 'Enter' && askQuestion()}
                />
                <div className="absolute right-2 top-2">
                  <button
                    onClick={askQuestion}
                    disabled={loading || !question.trim()}
                    className={`px-6 py-2 rounded-lg font-medium transition-all flex items-center ${loading || !question.trim() ? 'bg-gray-200 text-gray-400 cursor-not-allowed' : 'bg-purple-600 text-white hover:bg-purple-700 shadow-sm hover:shadow transform hover:-translate-y-0.5'}`}
                  >
                    {loading ? (
                      <>
                        <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>
                        Thinking
                      </>
                    ) : 'Ask'}
                  </button>
                </div>
              </div>
            </div>

            {/* 🔹 Answer Section */}
            {response && (
              <div className="bg-white p-6 md:p-8 rounded-2xl shadow-sm border border-gray-100 animate-fade-in-up">
                {response.error ? (
                  <>
                    <div className="flex items-center mb-4 text-red-600">
                      <svg className="w-8 h-8 mr-3" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>
                      <h2 className="text-xl font-bold tracking-tight">Oops! Something went wrong</h2>
                    </div>
                    <div className="p-5 bg-red-50 border border-red-100 rounded-xl text-red-700 font-medium">
                      {response.error}
                    </div>
                  </>
                ) : (
                  <>
                    <div className="flex items-center mb-5 pb-3 border-b border-gray-100">
                      <div className="w-10 h-10 rounded-full bg-blue-100 flex items-center justify-center mr-4">
                        <svg className="w-6 h-6 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" /></svg>
                      </div>
                      <h2 className="text-2xl font-bold text-gray-800 tracking-tight">Answer</h2>
                    </div>
                    
                    <div className="prose prose-blue max-w-none text-gray-700 mb-8 bg-blue-50/50 p-6 rounded-2xl border border-blue-100/50">
                      <p className="leading-relaxed text-lg">{response.answer}</p>
                    </div>

                    {response.sources && response.sources.length > 0 && (
                      <div>
                        <h3 className="text-sm font-bold text-gray-500 uppercase tracking-wider mb-4 flex items-center">
                          <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" /></svg>
                          Sources Consulted
                        </h3>

                        <div className="grid grid-cols-1 gap-4 max-h-72 overflow-y-auto pr-2">
                          {response.sources.map((s, i) => (
                            <div key={i} className="bg-white border border-gray-200 rounded-xl p-5 text-sm hover:border-blue-300 transition-colors shadow-sm group">
                              <div className="mb-2">
                                <span className="inline-block bg-blue-100 text-blue-800 text-xs font-bold px-3 py-1 rounded-full group-hover:bg-blue-200 transition-colors">
                                  Page {s.page}
                                </span>
                              </div>
                              <span className="text-gray-600 leading-relaxed italic border-l-2 border-gray-200 pl-3 block">"{s.content}..."</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}