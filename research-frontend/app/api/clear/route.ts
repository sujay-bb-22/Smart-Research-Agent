import { backendError, backendHeaders, backendUrlOrError } from "../../lib/backend";

export async function POST(req: Request) {
    const backend = backendUrlOrError();
    if (backend.response) return backend.response;
    try {
        const res = await fetch(`${backend.url}/clear`, {
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
