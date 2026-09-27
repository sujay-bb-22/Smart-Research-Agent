export type StreamEvent = {
  event?: "answer_delta" | "sources" | "suggestions" | "error" | "done";
  answer?: string;
  content?: string;
  sources?: unknown[];
  suggestions?: string[];
  error?: string;
};

/**
 * Buffers an SSE response so JSON that spans network chunks is parsed only
 * after the blank-line event delimiter has arrived.
 */
export async function readSseStream(
  response: Response,
  onEvent: (event: StreamEvent) => void,
  signal?: AbortSignal,
) {
  if (!response.body) throw new Error("The answer stream was unavailable.");

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  const consume = (block: string) => {
    const data = block
      .split(/\r?\n/)
      .filter((line) => line.startsWith("data:"))
      .map((line) => line.slice(5).trimStart())
      .join("\n")
      .trim();

    if (!data) return;
    if (data === "[DONE]") {
      onEvent({ event: "done" });
      return;
    }

    try {
      onEvent(JSON.parse(data) as StreamEvent);
    } catch {
      onEvent({ event: "error", error: "The server sent an invalid response event." });
    }
  };

  try {
    while (true) {
      if (signal?.aborted) {
        await reader.cancel();
        throw new DOMException("Request cancelled", "AbortError");
      }
      const { done, value } = await reader.read();
      buffer += decoder.decode(value || new Uint8Array(), { stream: !done });

      let divider = buffer.search(/\r?\n\r?\n/);
      while (divider >= 0) {
        consume(buffer.slice(0, divider));
        buffer = buffer.slice(divider).replace(/^\r?\n\r?\n/, "");
        divider = buffer.search(/\r?\n\r?\n/);
      }
      if (done) break;
    }
    if (buffer.trim()) consume(buffer);
  } finally {
    reader.releaseLock();
  }
}
