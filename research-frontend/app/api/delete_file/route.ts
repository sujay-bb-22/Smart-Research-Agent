export async function POST(req: Request) {
    try {
        const body = await req.json();
        const BACKEND_URL = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

        const res = await fetch(`${BACKEND_URL}/delete_file`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify(body),
        });

        if (!res.ok) {
            const errorText = await res.text();
            return Response.json({ error: "Failed to delete file", detail: errorText }, { status: res.status });
        }

        const data = await res.json();
        return Response.json(data);
    } catch (error) {
        console.error("❌ Delete proxy failed:", error);
        return Response.json({ error: "Delete proxy failed" }, { status: 500 });
    }
}
