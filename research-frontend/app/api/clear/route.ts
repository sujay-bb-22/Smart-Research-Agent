export async function POST(req: Request) {
    try {
        const BACKEND_URL = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";
        const res = await fetch(`${BACKEND_URL}/clear`, {
            method: "POST",
        });

        if (!res.ok) {
            const errorText = await res.text();
            return Response.json({ error: "Failed to clear documents", detail: errorText }, { status: res.status });
        }

        const data = await res.json();
        return Response.json(data);
    } catch (error) {
        console.error("❌ Clear proxy failed:", error);
        return Response.json({ error: "Clear proxy failed" }, { status: 500 });
    }
}
