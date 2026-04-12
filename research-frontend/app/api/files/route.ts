export async function GET() {
    try {
        const BACKEND_URL = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";
        const res = await fetch(`${BACKEND_URL}/files`, {
            cache: 'no-store' // Ensure we always get the latest file list
        });

        if (!res.ok) {
            return Response.json({ error: "Failed to fetch files from backend" }, { status: res.status });
        }

        const data = await res.json();
        return Response.json(data);
    } catch (error) {
        console.error("❌ Files proxy failed:", error);
        return Response.json({ error: "Files proxy failed" }, { status: 500 });
    }
}
