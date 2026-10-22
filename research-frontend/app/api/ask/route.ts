import { backendError, backendHeaders, backendUrlOrError } from "../../lib/backend";

export async function POST(req: Request) {
    const backend = backendUrlOrError();
    if (backend.response) return backend.response;
    try {
        const body = await req.json();

        const res = await fetch(`${backend.url}/ask`, {
            method: "POST",
            headers: backendHeaders(req, "application/json"),
            body: JSON.stringify(body),
        });

        if (!res.ok) {
            return backendError(res, "Unable to retrieve an answer.");
        }

        // Return the raw body stream from FastAPI to the browser
        return new Response(res.body, {
            headers: {
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        });

    } catch (error) {
        console.error("❌ Proxy crash:", error);
        return Response.json({ error: "Ask proxy failed" }, { status: 500 });
    }
}
