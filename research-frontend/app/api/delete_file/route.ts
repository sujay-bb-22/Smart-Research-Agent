import { backendError, backendHeaders } from "../../lib/backend";

export async function POST(req: Request) {
    try {
        const body = await req.json();
        const BACKEND_URL = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

        const res = await fetch(`${BACKEND_URL}/delete_file`, {
            method: "POST",
            headers: backendHeaders(req, "application/json"),
            body: JSON.stringify(body),
        });

        if (!res.ok) {
            return backendError(res, "Unable to delete the document.");
        }

        const data = await res.json();
        return Response.json(data);
    } catch (error) {
        console.error("Delete proxy failed", error);
        return Response.json({ error: { code: "DELETE_PROXY_ERROR", message: "Unable to delete the document." } }, { status: 502 });
    }
}
