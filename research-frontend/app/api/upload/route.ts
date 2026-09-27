import { backendHeaders } from "../../lib/backend";

export async function POST(req: Request) {
    try {
        const formData = await req.formData();

        const BACKEND_URL = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";
        const res = await fetch(`${BACKEND_URL}/upload`, {
            method: "POST",
            body: formData,
            headers: backendHeaders(req),
        });

        const text = await res.text();

        // 🔥 Try parse JSON safely
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
