import { backendError, backendHeaders, backendUrlOrError } from "../../lib/backend";

export async function POST(req: Request) {
    const backend = backendUrlOrError();
    if (backend.response) return backend.response;
    try {
        const body = await req.json();

        const res = await fetch(`${backend.url}/delete_file`, {
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
