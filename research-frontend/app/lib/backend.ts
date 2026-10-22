const FORWARDED_HEADERS = [
  "authorization",
  "x-api-key",
  "x-user-id",
  "x-workspace-id",
  "x-request-id",
];

export function backendHeaders(request: Request, contentType?: string) {
  const headers = new Headers();
  for (const name of FORWARDED_HEADERS) {
    const value = request.headers.get(name);
    if (value) headers.set(name, value);
  }
  if (contentType) headers.set("content-type", contentType);
  return headers;
}

export async function backendError(response: Response, fallback: string) {
  const body = await response.json().catch(() => null);
  return Response.json(body || { error: { code: "BACKEND_ERROR", message: fallback } }, {
    status: response.status || 502,
  });
}
