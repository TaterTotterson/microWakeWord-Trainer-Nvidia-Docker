export type JsonRecord = Record<string, any>;
const pendingGets = new Map<string, Promise<unknown>>();

export async function request<T = JsonRecord>(path: string, options: RequestInit = {}): Promise<T> {
  const response = await fetch(path, {
    credentials: "same-origin",
    cache: "no-store",
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
  const existing = pendingGets.get(path);
  if (existing) return existing as Promise<T>;
  const load = async () => {
    const controller = new AbortController();
    const timeout = window.setTimeout(() => controller.abort(), path === "/api/data" ? 120000 : 20000);
    try {
      return await request<T>(path, { signal: controller.signal });
    } finally {
      window.clearTimeout(timeout);
      pendingGets.delete(path);
    }
  };
  const promise = load();
  pendingGets.set(path, promise);
  return promise;
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
