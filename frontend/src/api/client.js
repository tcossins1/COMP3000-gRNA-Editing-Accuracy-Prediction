const DEFAULT_BASE = "http://127.0.0.1:8000";

export async function requestJson(path, { baseUrl, method = "GET", body } = {}) {
  const urlBase = baseUrl || import.meta.env.VITE_API_BASE || DEFAULT_BASE;
  const url = `${urlBase}${path}`;

  const opts = {
    method,
    headers: { "Content-Type": "application/json" },
  };

  if (body !== undefined) opts.body = JSON.stringify(body);

  let res;
  try {
    res = await fetch(url, opts);
  } catch {
    throw new Error("Network error: could not reach the API.");
  }

  let data = null;
  try {
    data = await res.json();
  } catch {
    data = null;
  }

  if (!res.ok) {
    const msg = data?.detail || data?.message || `Request failed (${res.status}).`;
    throw new Error(msg);
  }

  return data;
}