"use client";

import { useState } from "react";
import { BookOpen, ChevronDown, Copy } from "lucide-react";

export type Citation = {
  page: number | string;
  content: string;
  document_id?: string;
  filename?: string;
  chunk_id?: string;
  score?: number;
};

type SourceCitationsProps = {
  citations: Citation[];
  onCopy: (content: string) => void;
};

export function SourceCitations({ citations, onCopy }: SourceCitationsProps) {
  const [expanded, setExpanded] = useState<number | null>(null);

  return (
    <section className="mt-6 bg-gray-50 p-5 rounded-xl border border-gray-100" aria-label="Sources consulted">
      <h3 className="text-sm font-bold text-gray-500 uppercase tracking-widest mb-4 flex items-center">
        <BookOpen className="w-4 h-4 mr-2" />
        Sources Consulted
      </h3>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3 max-h-80 overflow-y-auto pr-1">
        {citations.map((citation, index) => {
          const isExpanded = expanded === index;
          const location = typeof citation.page === "number" ? `Page ${citation.page}` : citation.page;
          return (
            <article key={`${citation.document_id ?? "source"}-${citation.chunk_id ?? index}`} className="bg-white border border-gray-200 rounded-xl p-4 text-sm">
              <div className="flex items-start justify-between gap-3">
                <button
                  type="button"
                  onClick={() => setExpanded(isExpanded ? null : index)}
                  className="flex min-w-0 items-center gap-2 text-left text-blue-700 hover:text-blue-900 focus:outline-none focus:ring-2 focus:ring-blue-500 rounded"
                  aria-expanded={isExpanded}
                >
                  <span className="font-bold">{location}</span>
                  <ChevronDown className={`w-4 h-4 shrink-0 transition-transform ${isExpanded ? "rotate-180" : ""}`} />
                </button>
                <button
                  type="button"
                  onClick={() => onCopy(citation.content)}
                  className="p-1 text-gray-400 hover:text-blue-600 focus:outline-none focus:ring-2 focus:ring-blue-500 rounded"
                  aria-label={`Copy evidence from ${location}`}
                  title="Copy evidence"
                >
                  <Copy className="w-3.5 h-3.5" />
                </button>
              </div>
              {citation.filename && <p className="mt-2 truncate text-xs text-gray-500" title={citation.filename}>{citation.filename}</p>}
              <p className={`mt-2 leading-relaxed text-gray-700 ${isExpanded ? "" : "line-clamp-4"}`}>
                {citation.content}
              </p>
              {isExpanded && citation.chunk_id && <p className="mt-3 text-xs text-gray-400">Chunk {citation.chunk_id}</p>}
            </article>
          );
        })}
      </div>
    </section>
  );
}
