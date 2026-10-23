import { backendHeaders, backendUrlOrError } from "../../lib/backend";

export async function POST(req: Request) {
    const backend = backendUrlOrError();
    if (backend.response) return backend.response;
    try {
        const formData = await req.formData();

        const res = await fetch(`${backend.url}/upload`, {
            method: "POST",
            body: formData,
            headers: backendHeaders(req),
        });

        const text = await res.text();
        // Parse the backend response safely.
        try {
            const data = JSON.parse(text);
            if (!res.ok) return Response.json(data, { status: res.status });
            return Response.json(data, { status: res.status });
        } catch {
            return Response.json(
                { error: { code: "INVALID_BACKEND_RESPONSE", message: "Upload service returned an invalid response." } },
                { status: res.ok ? 502 : res.status }
            );
        }

    } catch (error) {
        console.error("Upload proxy failed", error);

        return Response.json(
            { error: "Upload proxy failed" },
            { status: 500 }
        );
    }
}
