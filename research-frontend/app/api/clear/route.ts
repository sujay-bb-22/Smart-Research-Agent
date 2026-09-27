import { backendError, backendHeaders } from "../../lib/backend";

export async function POST(req: Request) {
    try {
        const BACKEND_URL = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";
        const res = await fetch(`${BACKEND_URL}/clear`, {
            method: "POST",
            headers: backendHeaders(req),
        });

        if (!res.ok) {
            return backendError(res, "Unable to clear documents.");
        }

        const data = await res.json();
        return Response.json(data);
    } catch (error) {
        console.error("Clear proxy failed", error);
        return Response.json({ error: { code: "CLEAR_PROXY_ERROR", message: "Unable to clear documents." } }, { status: 502 });
    }
}
