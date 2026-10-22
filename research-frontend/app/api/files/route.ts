import { backendError, backendHeaders, backendUrlOrError } from "../../lib/backend";

export async function GET(req: Request) {
    const backend = backendUrlOrError();
    if (backend.response) return backend.response;
    try {
        const res = await fetch(`${backend.url}/files`, {
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
