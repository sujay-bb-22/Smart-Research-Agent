export type PersistedMessage = {
  role: "user" | "assistant";
  content: string;
  sources?: unknown[];
  suggestions?: string[];
  error?: string;
};

const STORAGE_KEY = "smart-research-agent:conversation:v1";

export function loadConversation(): PersistedMessage[] {
  if (typeof window === "undefined") return [];
  try {
    const stored = JSON.parse(window.localStorage.getItem(STORAGE_KEY) || "[]");
    return Array.isArray(stored) ? stored : [];
  } catch {
    return [];
  }
}

export function saveConversation(messages: PersistedMessage[]) {
  if (typeof window === "undefined") return;
  window.localStorage.setItem(STORAGE_KEY, JSON.stringify(messages));
}

export function clearConversation() {
  if (typeof window !== "undefined") window.localStorage.removeItem(STORAGE_KEY);
}
