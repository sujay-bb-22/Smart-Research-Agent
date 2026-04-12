export async function POST(req: Request) {
    try {
        const body = await req.json();
        const BACKEND_URL = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

        const res = await fetch(`${BACKEND_URL}/ask`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify(body),
        });

        if (!res.ok) {
            const errorText = await res.text();
            return Response.json({ error: "Backend error", detail: errorText }, { status: res.status });
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
