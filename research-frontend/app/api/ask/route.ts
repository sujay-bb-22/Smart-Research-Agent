export async function POST(req: Request) {
    try {
        const body = await req.json();

        const res = await fetch("http://127.0.0.1:8000/ask", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify(body),
        });

        // ✅ Handle backend errors
        if (!res.ok) {
            return Response.json(
                { error: "Backend error" },
                { status: res.status }
            );
        }

        const data = await res.json();

        return Response.json(data);

    } catch (error) {
        console.error("Proxy error:", error);

        return Response.json(
            { error: "Proxy failed to connect to backend" },
            { status: 500 }
        );
    }
}