export const jsonHeaders = { "Content-Type": "application/json" };

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly body: string
  ) {
    super(message);
    this.name = "ApiError";
  }
}

export async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, init);
  if (!response.ok) {
    const body = await response.text();
    throw new ApiError(body || response.statusText, response.status, body);
  }
  return response.json() as Promise<T>;
}
