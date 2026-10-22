"use client";

import { useEffect } from "react";

export default function GlobalError({ error, reset }: { error: Error & { digest?: string }; reset: () => void }) {
  useEffect(() => {
    console.error("Unhandled frontend error", { message: error.message, digest: error.digest });
  }, [error]);

  return (
    <main className="min-h-screen grid place-items-center bg-gray-50 p-6 text-gray-900">
      <section className="max-w-md bg-white border border-gray-200 rounded-xl p-6 shadow-sm">
        <h1 className="text-xl font-bold">Something interrupted the research workspace</h1>
        <p className="mt-2 text-sm text-gray-600">Your documents remain unchanged. Reload the workspace to continue.</p>
        <button type="button" onClick={reset} className="mt-5 rounded-lg bg-blue-600 px-4 py-2 text-sm font-semibold text-white hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500">Try again</button>
      </section>
    </main>
  );
}
