import { backendError, backendHeaders } from "../../lib/backend";

export async function GET(req: Request) {
    try {
        const BACKEND_URL = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";
        const res = await fetch(`${BACKEND_URL}/files`, {
            cache: "no-store",
            headers: backendHeaders(req),
        });

        if (!res.ok) {
            return backendError(res, "Unable to load documents.");
        }

        const data = await res.json();
        return Response.json(data);
    } catch (error) {
        console.error("Files proxy failed", error);
        return Response.json({ error: { code: "FILES_PROXY_ERROR", message: "Unable to load documents." } }, { status: 502 });
    }
}
