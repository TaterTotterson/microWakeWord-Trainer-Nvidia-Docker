export type JsonRecord = Record<string, any>;

export async function request<T = JsonRecord>(path: string, options: RequestInit = {}): Promise<T> {
  const response = await fetch(path, {
    credentials: "same-origin",
    ...options,
    headers: {
      Accept: "application/json",
      ...(options.headers || {}),
    },
  });
  const contentType = response.headers.get("content-type") || "";
  const body = contentType.includes("application/json")
    ? await response.json()
    : await response.text();
  if (!response.ok) {
    const message = typeof body === "object" && body
      ? body.error || body.detail || body.message
      : body;
    throw new Error(String(message || `Request failed (${response.status})`));
  }
  return body as T;
}

export function getJson<T = JsonRecord>(path: string): Promise<T> {
  return request<T>(path);
}

export function postJson<T = JsonRecord>(path: string, body: unknown = {}): Promise<T> {
  return request<T>(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export function putJson<T = JsonRecord>(path: string, body: unknown): Promise<T> {
  return request<T>(path, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}
