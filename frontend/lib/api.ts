import type { ModelInfo, PredictionResponse } from "./types";

export class ApiError extends Error {
  constructor(message: string, public code: string) {
    super(message);
  }
}

const BACKEND_DOWN =
  "The inference backend is not reachable. Start it with `uv run uvicorn app.main:app " +
  "--port 8000` in backend/.";

async function parseError(res: Response): Promise<ApiError> {
  try {
    const body = await res.json();
    if (body?.error?.message) return new ApiError(body.error.message, body.error.code);
  } catch {
    /* non-JSON error (e.g. proxy failure) */
  }
  if (res.status >= 500) return new ApiError(BACKEND_DOWN, "BACKEND_UNAVAILABLE");
  return new ApiError(`Request failed (HTTP ${res.status}).`, "HTTP_ERROR");
}

async function request<T>(input: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(input, init);
  } catch {
    throw new ApiError(BACKEND_DOWN, "BACKEND_UNAVAILABLE");
  }
  if (!res.ok) throw await parseError(res);
  return res.json() as Promise<T>;
}

export function getModelInfo(): Promise<ModelInfo> {
  return request<ModelInfo>("/api/v1/model", { cache: "no-store" });
}

export function predict(file: File): Promise<PredictionResponse> {
  const form = new FormData();
  form.append("file", file);
  return request<PredictionResponse>("/api/v1/predictions", { method: "POST", body: form });
}
