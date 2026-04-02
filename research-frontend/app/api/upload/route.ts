export async function POST(req: Request) {
    try {
        const formData = await req.formData();

        const BACKEND_URL = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";
        const res = await fetch(`${BACKEND_URL}/upload`, {
            method: "POST",
            body: formData,
        });

        // 🔥 ALWAYS read as text first
        const text = await res.text();

        console.log("Backend response:", text);

        // 🔥 Try parse JSON safely
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
            { error: "Upload proxy failed" },
            { status: 500 }
        );
    }
}
