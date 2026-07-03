import { describe, expect, test } from "bun:test";
import { ApiError, request } from "./http";

describe("request", () => {
  test("returns typed JSON response data", async () => {
    const originalFetch = globalThis.fetch;
    globalThis.fetch = async () => new Response(JSON.stringify({ ok: true }), { status: 200 });

    try {
      const result = await request<{ ok: boolean }>("/api/test");

      expect(result).toEqual({ ok: true });
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  test("throws ApiError with status and response text for failed requests", async () => {
    const originalFetch = globalThis.fetch;
    globalThis.fetch = async () => new Response("nope", { status: 403, statusText: "Forbidden" });

    try {
      await expect(request("/api/test")).rejects.toMatchObject({
        name: "ApiError",
        status: 403,
        body: "nope"
      } satisfies Partial<ApiError>);
    } finally {
      globalThis.fetch = originalFetch;
    }
  });
});
