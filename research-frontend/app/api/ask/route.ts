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

        // ALWAYS read as text first to handle non-JSON responses from FastAPI
        const text = await res.text();

        // Try parse JSON safely
        try {
            const data = JSON.parse(text);
            return Response.json(data, { status: res.status });
        } catch (err) {
            console.error("❌ Not JSON:", text);

            return Response.json(
                { error: "Invalid backend response", raw: text },
                { status: 500 }
            );
        }

    } catch (error) {
        console.error("❌ Proxy crash:", error);

        return Response.json(
            { error: "Ask proxy failed" },
            { status: 500 }
        );
    }
}
