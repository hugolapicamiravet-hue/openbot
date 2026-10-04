import { describe, expect, spyOn, test } from "bun:test";
import { runtimeModelForEnvironment } from "../src/copilot";
import { chatCompletionsUrl, createModelCompleter } from "../src/routing/model";

/**
 * Where the deployment's own model calls go.
 *
 * This existed as `${base}/v1/chat/completions` and the documented value of `OPENAI_BASE_URL`
 * already ends in `/v1`, so every deployment behind a gateway got `/v1/v1/chat/completions` and a
 * 404. Both callers treat a throw as "not sure": the intent router silently routed everything to
 * the default coworker, and tool selection would have silently offered every tool. Neither said
 * anything, which is why the case is pinned here rather than left to a comment.
 */
describe("chatCompletionsUrl", () => {
  test("unset falls back to the public API, with its version", () => {
    expect(chatCompletionsUrl({})).toBe(
      "https://api.openai.com/v1/chat/completions",
    );
  });

  test("a gateway documented with /v1 is not given a second one", () => {
    expect(
      chatCompletionsUrl({ OPENAI_BASE_URL: "https://gateway.internal/v1" }),
    ).toBe("https://gateway.internal/v1/chat/completions");
  });

  test("a trailing slash is not a different URL", () => {
    expect(
      chatCompletionsUrl({ OPENAI_BASE_URL: "https://gateway.internal/v1/" }),
    ).toBe("https://gateway.internal/v1/chat/completions");
  });

  test("a host with no version gets one, which is what a bare origin means", () => {
    expect(
      chatCompletionsUrl({ OPENAI_BASE_URL: "http://localhost:4010" }),
    ).toBe("http://localhost:4010/v1/chat/completions");
  });

  test("a version other than 1 is still a version", () => {
    expect(chatCompletionsUrl({ OPENAI_BASE_URL: "https://x.test/v2" })).toBe(
      "https://x.test/v2/chat/completions",
    );
  });

  test("whitespace and empty are the same as unset", () => {
    expect(chatCompletionsUrl({ OPENAI_BASE_URL: "   " })).toBe(
      "https://api.openai.com/v1/chat/completions",
    );
    expect(chatCompletionsUrl({ OPENAI_BASE_URL: "" })).toBe(
      "https://api.openai.com/v1/chat/completions",
    );
  });

  test("a path that merely contains v1 is not a version segment", () => {
    // `/v1beta` is Google's, and it is not the segment this is looking for.
    expect(
      chatCompletionsUrl({ OPENAI_BASE_URL: "https://x.test/v1beta" }),
    ).toBe("https://x.test/v1beta/v1/chat/completions");
  });
});

test("local router reuses Ollama context and thinking settings without a paid/key fallback", async () => {
  const model = runtimeModelForEnvironment(
    { provider: "openai", defaultModel: "qwen3:14b" },
    {
      OPENAI_BASE_URL: "http://localhost:11434/v1/",
      OPENBOT_OLLAMA_CONTEXT_LENGTH: "8192",
      OPENBOT_OLLAMA_THINK: "false",
    },
  );
  let calls = 0;
  const fetchSpy = spyOn(globalThis, "fetch").mockImplementation(
    async (url, init) => {
      calls++;
      expect(String(url)).toBe("http://localhost:11434/api/chat");
      const body = JSON.parse(String(init?.body));
      expect(body.options).toMatchObject({ num_ctx: 8192 });
      expect(body.think).toBe(false);
      expect(body.format).toBe("json");
      expect(init?.signal).toBeInstanceOf(AbortSignal);
      return Response.json({
        model: "qwen3:14b",
        created_at: "2026-10-04T00:00:00Z",
        message: { role: "assistant", content: '{"bot":"general"}' },
        done: true,
        done_reason: "stop",
      });
    },
  );
  const complete = createModelCompleter({
    model,
    resolveApiKey: async () => {
      throw new Error("The local router must not resolve paid credentials.");
    },
  });
  try {
    expect(await complete("Choose a Bot.")).toBe('{"bot":"general"}');
    await expect(complete("Cancelled", AbortSignal.abort())).rejects.toThrow();
    expect(calls).toBe(1);
  } finally {
    fetchSpy.mockRestore();
  }
});
