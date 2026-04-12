"use client";

import { useState, useRef, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { 
  FileText, 
  Trash2, 
  Send, 
  Download, 
  Copy, 
  Check, 
  Plus, 
  Library, 
  Database, 
  BookOpen, 
  Cpu, 
  MessageSquare, 
  Share2,
  ChevronRight
} from "lucide-react";
import { jsPDF } from "jspdf";
import html2canvas from "html2canvas";

type Source = {
  page: number;
  content: string;
};

type Message = {
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  suggestions?: string[];
  error?: string;
};

export default function Home() {
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [uploadStatus, setUploadStatus] = useState("");
  const [clearing, setClearing] = useState(false);
  const [files, setFiles] = useState<{name: string, size: number, uploaded_at: number}[]>([]);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // 🔹 Fetch files
  const fetchFiles = async () => {
    try {
      const res = await fetch("/api/files");
      if (res.ok) {
        const data = await res.json();
        setFiles(data);
      }
    } catch (err) {
      console.error("Fetch files error:", err);
    }
  };

  // 🔹 Export Report functions
  const copyToClipboard = (text: string, id: string) => {
    navigator.clipboard.writeText(text);
    setUploadStatus(`📋 Copied!`);
    setTimeout(() => setUploadStatus(""), 2000);
  };

  const exportToMarkdown = () => {
    const reportHeader = `# Research Report\nGenerated on: ${new Date().toLocaleString()}\n\n---\n\n`;
    const chatContent = messages.map(m => 
      `### ${m.role === 'user' ? 'User' : 'Assistant'}\n${m.content}\n\n${m.sources?.length ? `**Sources:**\n${m.sources.map(s => `- Page ${s.page}: ${s.content}`).join('\n')}\n` : ''}`
    ).join('\n---\n\n');
    
    const blob = new Blob([reportHeader + chatContent], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `research_report_${new Date().getTime()}.md`;
    a.click();
    setUploadStatus("✅ Markdown Report Downloaded");
  };

  const exportToPDF = async () => {
    const element = document.getElementById('chat-container');
    if (!element) return;
    
    setLoading(true);
    setUploadStatus("⚡ Generating PDF Report...");
    
    try {
      const canvas = await html2canvas(element, { scale: 2, useCORS: true });
      const imgData = canvas.toDataURL('image/png');
      const pdf = new jsPDF('p', 'mm', 'a4');
      const imgProps = pdf.getImageProperties(imgData);
      const pdfWidth = pdf.internal.pageSize.getWidth();
      const pdfHeight = (imgProps.height * pdfWidth) / imgProps.width;
      
      pdf.addImage(imgData, 'PNG', 0, 0, pdfWidth, pdfHeight);
      pdf.save(`research_report_${new Date().getTime()}.pdf`);
      setUploadStatus("✅ PDF Report Saved");
    } catch (err) {
      console.error("PDF Export error:", err);
      setUploadStatus("❌ PDF Generation Failed");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchFiles();
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // 🔹 Format File Size
  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  // 🔹 Delete specific file
  const deleteFile = async (filename: string) => {
    try {
      const res = await fetch("/api/delete_file", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ filename })
      });
      if (res.ok) {
        setUploadStatus(`🗑️ Deleted ${filename}`);
        fetchFiles();
      }
    } catch (err) {
      console.error("Delete error:", err);
    }
  };

  // 🔹 Ask question
  const askQuestion = async (overrideQuestion?: string) => {
    const queryText = overrideQuestion || question;
    if (!queryText.trim()) return;

    const userMessage: Message = { role: "user", content: queryText };
    setMessages((prev) => [...prev, userMessage]);
    setQuestion("");
    setLoading(true);

    try {
      const res = await fetch("/api/ask", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ 
          question: userMessage.content,
          history: messages.map(m => ({ role: m.role, content: m.content }))
        }),
      });

      if (!res.ok) throw new Error("Failed to fetch");

      const reader = res.body?.getReader();
      const decoder = new TextDecoder();
      
      // Add an empty assistant message that we'll update
      setMessages((prev) => [...prev, { role: "assistant", content: "", sources: [] }]);

      while (reader) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });
        const lines = chunk.split("\n");

        for (const line of lines) {
          if (line.startsWith("data: ")) {
            const dataStr = line.slice(6).trim();
            if (!dataStr || dataStr === "[DONE]") continue;

            try {
              const data = JSON.parse(dataStr);
              setMessages((prev) => {
                const newMessages = [...prev];
                const lastMsg = newMessages[newMessages.length - 1];
                
                if (data.sources) lastMsg.sources = data.sources;
                if (data.suggestions) lastMsg.suggestions = data.suggestions;
                if (data.content) lastMsg.content += data.content;
                if (data.answer) lastMsg.content = data.answer;
                if (data.error) lastMsg.error = data.error;
                
                return newMessages;
              });
            } catch (e) {
              console.error("Error parsing stream chunk:", e);
            }
          }
        }
      }
    } catch (error) {
      console.error(error);
      setMessages((prev) => [...prev, {
        role: "assistant",
        content: "",
        error: "Error fetching answer. Please try again."
      }]);
    } finally {
      setLoading(false);
    }
  };

  // 🔹 Clear Database
  const clearDatabase = async () => {
    try {
      setClearing(true);
      const res = await fetch("/api/clear", { method: "POST" });
      if (res.ok) {
        setUploadStatus("🧹 Database cleared!");
        setMessages([]);
        fetchFiles();
      }
    } catch (err) {
      console.error(err);
    } finally {
      setClearing(false);
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
        fetchFiles();
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
                <Database className="w-5 h-5 mr-2 text-blue-500" />
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
                <Plus className="w-5 h-5 mr-2" />
                Upload Document
              </button>

              <button
                onClick={clearDatabase}
                disabled={clearing}
                className="w-full mt-3 font-medium py-3 rounded-xl transition-all shadow-sm flex justify-center items-center bg-red-50 text-red-600 hover:bg-red-100 border border-red-100"
              >
                <Trash2 className="w-5 h-5 mr-2" />
                {clearing ? 'Clearing...' : 'Clear Documents'}
              </button>

                {uploadStatus && (
                  <div className={`mt-4 text-sm font-medium p-3 rounded-xl flex items-center ${uploadStatus.includes('success') || uploadStatus.includes('✅') ? 'bg-green-50 text-green-700 border border-green-100' : uploadStatus.includes('Error') || uploadStatus.includes('failed') || uploadStatus.includes('❌') ? 'bg-red-50 text-red-700 border border-red-100' : 'bg-blue-50 text-blue-700 border border-blue-100'}`}>
                    { (uploadStatus.includes('success') || uploadStatus.includes('✅')) && <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" /></svg>}
                    {uploadStatus}
                  </div>
                )}
              </div>

              {/* 🔹 Document Library */}
              <div className="bg-white p-6 rounded-2xl shadow-sm border border-gray-100 transition-all hover:shadow-md">
                <h2 className="text-lg font-bold text-gray-800 mb-4 flex items-center">
                  <Library className="w-5 h-5 mr-2 text-green-500" />
                  Document Library
                </h2>
                
                {files.length === 0 ? (
                  <p className="text-gray-400 text-sm italic py-4 text-center">No documents uploaded yet.</p>
                ) : (
                  <div className="space-y-3">
                    {files.map((f, i) => (
                      <div key={i} className="flex items-center justify-between p-3 bg-gray-50 rounded-xl border border-gray-100 group hover:border-blue-200 transition-all">
                        <div className="flex-1 min-w-0 pr-2">
                          <p className="text-sm font-semibold text-gray-700 truncate" title={f.name}>{f.name}</p>
                          <p className="text-xs text-gray-400">{formatFileSize(f.size)} • {new Date(f.uploaded_at * 1000).toLocaleDateString()}</p>
                        </div>
                        <button 
                          onClick={() => deleteFile(f.name)}
                          className="p-2 text-gray-400 hover:text-red-500 hover:bg-red-50 rounded-lg transition-all"
                          title="Delete document"
                        >
                          <Trash2 className="w-4 h-4" />
                        </button>
                      </div>
                    ))}
                  </div>
                )}
              </div>
          </div>

          <div className="lg:col-span-2 space-y-6">

            {/* 🔹 Chat Section */}
            <div id="chat-container" className="space-y-6 max-h-[600px] overflow-y-auto pr-2 scroll-smooth">
              {messages.length > 0 && (
                <div className="flex justify-end gap-2 mb-4">
                  <button onClick={exportToMarkdown} className="flex items-center text-xs font-bold bg-white text-gray-600 px-3 py-2 rounded-lg border border-gray-200 hover:bg-gray-50 shadow-sm transition-all"><Download className="w-3.5 h-3.5 mr-1.5" /> MD</button>
                  <button onClick={exportToPDF} className="flex items-center text-xs font-bold bg-white text-gray-600 px-3 py-2 rounded-lg border border-gray-200 hover:bg-gray-50 shadow-sm transition-all"><Download className="w-3.5 h-3.5 mr-1.5" /> PDF</button>
                </div>
              )}
              {messages.map((msg, idx) => (
                <div key={idx} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                  {msg.role === 'user' ? (
                    <div className="bg-purple-600 text-white p-5 rounded-2xl rounded-tr-none max-w-[85%] shadow-sm">
                      <p className="leading-relaxed">{msg.content}</p>
                    </div>
                  ) : (
                    <div className="bg-white p-6 md:p-8 rounded-2xl shadow-sm border border-gray-100 max-w-full lg:max-w-[95%]">
                      {msg.error ? (
                         <>
                           <div className="flex items-center mb-4 text-red-600">
                             <svg className="w-8 h-8 mr-3" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>
                             <h2 className="text-xl font-bold tracking-tight">Oops! Something went wrong</h2>
                           </div>
                           <div className="p-5 bg-red-50 border border-red-100 rounded-xl text-red-700 font-medium">
                             {msg.error}
                           </div>
                         </>
                      ) : (
                        <>
                          <div className="flex items-center justify-between mb-5 pb-3 border-b border-gray-100">
                            <div className="flex items-center">
                              <div className="w-10 h-10 rounded-full bg-blue-100 flex items-center justify-center mr-4">
                                <Cpu className="w-6 h-6 text-blue-600" />
                              </div>
                              <h2 className="text-xl font-bold text-gray-800 tracking-tight">Answer</h2>
                            </div>
                            <button onClick={() => copyToClipboard(msg.content, `ans-${idx}`)} className="text-gray-400 hover:text-blue-600 transition-colors p-1.5 rounded-lg hover:bg-blue-50" title="Copy Answer"><Copy className="w-4 h-4" /></button>
                          </div>
                          
                          <div className="prose prose-blue max-w-none text-gray-800 mb-8 overflow-x-auto">
                             <ReactMarkdown remarkPlugins={[remarkGfm]}>
                                {msg.content}
                             </ReactMarkdown>
                          </div>

                          {msg.sources && msg.sources.length > 0 && (
                            <div className="mt-6 bg-gray-50 p-5 rounded-xl border border-gray-100">
                              <h3 className="text-sm font-bold text-gray-400 uppercase tracking-widest mb-4 flex items-center">
                                <BookOpen className="w-4 h-4 mr-2" />
                                Sources Consulted
                              </h3>

                              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 max-h-72 overflow-y-auto pr-2">
                                {msg.sources.map((s, i) => (
                                  <div key={i} className="bg-white border border-gray-200 rounded-xl p-4 text-sm hover:border-blue-300 transition-all shadow-sm group relative">
                                    <div className="flex justify-between items-center mb-2">
                                      <span className="inline-block bg-blue-50 text-blue-700 text-xs font-bold px-2 py-1 rounded-lg">
                                        Page {s.page}
                                      </span>
                                      <button onClick={() => copyToClipboard(s.content, `src-${idx}-${i}`)} className="text-gray-300 hover:text-blue-500 transition-colors opacity-0 group-hover:opacity-100"><Copy className="w-3 h-3" /></button>
                                    </div>
                                    <span className="text-gray-600 leading-relaxed italic border-l-2 border-gray-100 pl-3 block line-clamp-4 group-hover:text-gray-900 transition-colors">"{s.content}..."</span>
                                  </div>
                                ))}
                              </div>
                            </div>
                          )}

                          {msg.suggestions && msg.suggestions.length > 0 && (
                            <div className="mt-6">
                              <p className="text-xs font-bold text-gray-400 uppercase tracking-widest mb-3 flex items-center">
                                <svg className="w-3.5 h-3.5 mr-1.5 text-purple-400" fill="currentColor" viewBox="0 0 20 20"><path d="M11 3a1 1 0 10-2 0v1a1 1 0 102 0V3zM15.657 5.757a1 1 0 00-1.414-1.414l-.707.707a1 1 0 001.414 1.414l.707-.707zM18 10a1 1 0 01-1 1h-1a1 1 0 110-2h1a1 1 0 011 1zM5.05 6.464A1 1 0 106.464 5.05l-.707-.707a1 1 0 00-1.414 1.414l.707.707zM5 10a1 1 0 01-1 1H3a1 1 0 110-2h1a1 1 0 011 1zM8 16v-1a1 1 0 112 0v1a1 1 0 11-2 0zM13.464 15.05a1 1 0 010 1.414l-.707.707a1 1 0 01-1.414-1.414l.707-.707a1 1 0 011.414 0zM17 14a1 1 0 011 1v1a1 1 0 11-2 0v-1a1 1 0 011-1z" /></svg>
                                Suggested Follow-ups
                              </p>
                              <div className="flex flex-wrap gap-2">
                                {msg.suggestions.map((s, j) => (
                                  <button
                                    key={j}
                                    onClick={() => askQuestion(s)}
                                    className="text-sm bg-purple-50 text-purple-700 px-4 py-2 rounded-xl border border-purple-100 hover:bg-purple-100 hover:border-purple-200 transition-all text-left animate-in fade-in slide-in-from-bottom-2 duration-300"
                                    style={{ animationDelay: `${j * 100}ms` }}
                                  >
                                    {s}
                                  </button>
                                ))}
                              </div>
                            </div>
                          )}
                        </>
                      )}
                    </div>
                  )}
                </div>
              ))}
              <div ref={messagesEndRef} />
            </div>

            {/* 🔹 Question Section (Relocated) */}
            <div className="bg-white p-6 md:p-8 rounded-2xl shadow-lg border border-gray-100 transition-all hover:shadow-xl relative overflow-hidden group">
              <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-purple-500 via-blue-500 to-purple-500 opacity-70"></div>
              
              <h2 className="text-lg font-bold text-gray-800 mb-4 flex items-center">
                <MessageSquare className="w-5 h-5 mr-2 text-purple-600 animate-pulse" />
                Ask a follow-up question
              </h2>

              <div className="relative">
                <input
                  className="w-full bg-gray-50 border border-gray-200 text-gray-900 rounded-xl p-4 pr-32 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent transition-all shadow-inner group-hover:bg-white"
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                  placeholder="Ask anything about the document..."
                  onKeyDown={(e) => e.key === 'Enter' && askQuestion()}
                />
                <div className="absolute right-2 top-2">
                  <button
                    onClick={() => askQuestion()}
                    disabled={loading || !question.trim()}
                    className={`px-8 py-2.5 rounded-lg font-bold transition-all flex items-center shadow-md ${loading || !question.trim() ? 'bg-gray-100 text-gray-400 cursor-not-allowed shadow-none' : 'bg-gradient-to-r from-purple-600 to-blue-600 text-white hover:from-purple-700 hover:to-blue-700 transform hover:-translate-y-0.5 active:translate-y-0 active:shadow-inner'}`}
                  >
                    {loading ? (
                      <>
                        <Cpu className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" />
                        Thinking
                      </>
                    ) : (
                      <>
                        Send <ChevronRight className="w-4 h-4 ml-1" />
                      </>
                    )}
                  </button>
                </div>
              </div>
              <p className="mt-3 text-xs text-gray-400 text-center italic">Tip: Press Enter to quickly send your question</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}