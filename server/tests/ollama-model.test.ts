import { describe, expect, spyOn, test } from "bun:test";
import { ollamaModelForEnvironment } from "../src/ollama-model";

const model = { provider: "openai", defaultModel: "qwen3:14b" };
const environment = {
  OPENAI_BASE_URL: "http://localhost:11434/v1",
  OPENBOT_OLLAMA_CONTEXT_LENGTH: "8192",
  OPENBOT_OLLAMA_THINK: "false",
  OPENBOT_OLLAMA_KEEP_ALIVE: "30m",
};

describe("native Ollama model", () => {
  test("preserves tool-call history and returns native function calls", async () => {
    let sent: Record<string, unknown> = {};
    const fetchSpy = spyOn(globalThis, "fetch").mockImplementation(
      async (_url, init) => {
        sent = JSON.parse(String(init?.body));
        return Response.json({
          model: "qwen3:14b",
          created_at: "2026-10-04T00:00:00Z",
          message: {
            role: "assistant",
            content: "",
            tool_calls: [
              { function: { name: "test_tool", arguments: { choice: "x" } } },
            ],
          },
          done: true,
          done_reason: "stop",
        });
      },
    );
    try {
      const native = ollamaModelForEnvironment(model, environment);
      if (!native) throw new Error("Native model was not created.");
      const result = await native.doGenerate({
        prompt: [
          { role: "user", content: [{ type: "text", text: "Use the tool." }] },
          {
            role: "assistant",
            content: [
              {
                type: "tool-call",
                toolCallId: "call-1",
                toolName: "test_tool",
                input: { choice: "x" },
              },
            ],
          },
          {
            role: "tool",
            content: [
              {
                type: "tool-result",
                toolCallId: "call-1",
                toolName: "test_tool",
                output: { type: "text", value: "completed" },
              },
            ],
          },
        ],
      });
      expect(sent.messages).toMatchObject([
        { role: "user", content: "Use the tool." },
        {
          role: "assistant",
          tool_calls: [
            { function: { name: "test_tool", arguments: { choice: "x" } } },
          ],
        },
        { role: "tool", content: "completed" },
      ]);
      expect(result.content).toContainEqual(
        expect.objectContaining({
          type: "tool-call",
          toolName: "test_tool",
          input: '{"choice":"x"}',
        }),
      );
    } finally {
      fetchSpy.mockRestore();
    }
  });

  test("decodes text across fragmented native NDJSON chunks", async () => {
    const native = ollamaModelForEnvironment(model, environment);
    if (!native) throw new Error("Native model was not created.");
    const base = { model: "qwen3:14b", created_at: "2026-10-04T00:00:00Z" };
    const encoded = new TextEncoder().encode(
      `${[
        {
          ...base,
          message: { role: "assistant", content: "Hello " },
          done: false,
        },
        {
          ...base,
          message: { role: "assistant", content: "world" },
          done: false,
        },
        {
          ...base,
          message: { role: "assistant", content: "" },
          done: true,
          done_reason: "stop",
          prompt_eval_count: 12,
          eval_count: 2,
        },
      ]
        .map((line) => JSON.stringify(line))
        .join("\n")}\n`,
    );
    const fetchSpy = spyOn(globalThis, "fetch").mockImplementation(
      async () =>
        new Response(
          new ReadableStream({
            start(controller) {
              for (let i = 0; i < encoded.length; i += 17)
                controller.enqueue(encoded.slice(i, i + 17));
              controller.close();
            },
          }),
          { headers: { "Content-Type": "application/x-ndjson" } },
        ),
    );
    try {
      const result = await native.doStream({
        prompt: [{ role: "system", content: "Hello" }],
      });
      let answer = "",
        finished = false;
      for await (const event of result.stream) {
        if (event.type === "text-delta") answer += event.delta;
        if (event.type === "finish") finished = true;
      }
      expect(answer).toBe("Hello world");
      expect(finished).toBe(true);
    } finally {
      fetchSpy.mockRestore();
    }
  });

  test("accepts another Ollama model without intercepting non-Ollama configuration", () => {
    expect(
      ollamaModelForEnvironment(
        { ...model, defaultModel: "llama3.2" },
        environment,
      )?.modelId,
    ).toBe("llama3.2");
    expect(
      ollamaModelForEnvironment(
        { provider: "anthropic", defaultModel: "claude" },
        {},
      ),
    ).toBeUndefined();
    expect(() =>
      ollamaModelForEnvironment(model, {
        ...environment,
        OPENBOT_OLLAMA_THINK: "maybe",
      }),
    ).toThrow("true or false");
    expect(() =>
      ollamaModelForEnvironment(model, {
        ...environment,
        OPENAI_BASE_URL: "https://example.test",
      }),
    ).toThrow("ending in /v1");
    expect(() =>
      ollamaModelForEnvironment(model, {
        ...environment,
        OPENAI_BASE_URL: "https://example.test/v1?key=fixture",
      }),
    ).toThrow("must not contain credentials");
    expect(() =>
      ollamaModelForEnvironment(model, {
        ...environment,
        OPENAI_BASE_URL: "secret-invalid-url",
      }),
    ).toThrow("valid OPENAI_BASE_URL");
  });

  test("propagates cancellation and HTTP errors instead of falling back to a paid provider", async () => {
    const native = ollamaModelForEnvironment(model, environment);
    if (!native) throw new Error("Native model was not created.");
    const aborted = new DOMException("Cancelled", "AbortError");
    const fetchSpy = spyOn(globalThis, "fetch").mockRejectedValue(aborted);
    const prompt = [{ role: "system" as const, content: "Hello" }];
    try {
      await expect(native.doGenerate({ prompt })).rejects.toBe(aborted);
      fetchSpy.mockImplementation(
        async () =>
          new Response(JSON.stringify({ error: "Model unavailable" }), {
            status: 503,
            headers: { "Content-Type": "application/json" },
          }),
      );
      await expect(native.doStream({ prompt })).rejects.toThrow();
      expect(fetchSpy).toHaveBeenCalledTimes(2);
    } finally {
      fetchSpy.mockRestore();
    }
  });

  test("requires explicit opt-in and refuses invalid settings", () => {
    expect(ollamaModelForEnvironment(model, {})).toBeUndefined();
    expect(() =>
      ollamaModelForEnvironment(model, {
        ...environment,
        OPENBOT_OLLAMA_KEEP_ALIVE: "forever",
      }),
    ).toThrow("positive duration");
    for (const value of ["0", "-1", "1.5", "invalid"]) {
      expect(() =>
        ollamaModelForEnvironment(model, {
          ...environment,
          OPENBOT_OLLAMA_CONTEXT_LENGTH: value,
        }),
      ).toThrow("positive integer");
    }
    expect(() =>
      ollamaModelForEnvironment(
        { ...model, provider: "anthropic" },
        environment,
      ),
    ).toThrow("OpenAI-compatible provider");
  });

  test("sends the complete conversation, tools and context to the local native API", async () => {
    let requestUrl = "";
    let body: Record<string, unknown> = {};
    let requestSignal: AbortSignal | null | undefined;
    const controller = new AbortController();
    const fetchSpy = spyOn(globalThis, "fetch").mockImplementation(
      async (url, init) => {
        requestUrl = String(url);
        requestSignal = init?.signal;
        body = JSON.parse(String(init?.body));
        return new Response(
          JSON.stringify({
            model: "qwen3:14b",
            created_at: "2026-10-03T00:00:00Z",
            message: {
              role: "assistant",
              content: "OpenBot local model works",
            },
            done: true,
            done_reason: "stop",
            prompt_eval_count: 7103,
            eval_count: 7,
          }) + (body.stream ? "\n" : ""),
          { headers: { "Content-Type": "application/json" } },
        );
      },
    );
    try {
      const native = ollamaModelForEnvironment(model, environment);
      if (!native) throw new Error("Native model was not created.");
      await native.doGenerate({
        abortSignal: controller.signal,
        providerOptions: { ollama: { options: { seed: 42, num_ctx: 4096 } } },
        prompt: [
          { role: "system", content: "Keep the standing instructions." },
          {
            role: "user",
            content: [
              {
                type: "text",
                text: "Reply with exactly: OpenBot local model works",
              },
            ],
          },
        ],
        tools: [
          {
            type: "function",
            name: "test_tool",
            inputSchema: { type: "object", properties: {} },
          },
        ],
      });
      expect(requestUrl).toBe("http://localhost:11434/api/chat");
      expect(body.model).toBe("qwen3:14b");
      expect(body.options).toMatchObject({ num_ctx: 8192 });
      expect(body.think).toBe(false);
      expect(body.keep_alive).toBe("30m");
      expect(body.options).toMatchObject({ seed: 42, num_ctx: 8192 });
      expect(requestSignal).toBe(controller.signal);
      expect(body.messages).toEqual([
        { role: "system", content: "Keep the standing instructions." },
        {
          role: "user",
          content: "Reply with exactly: OpenBot local model works",
        },
      ]);
      expect(body.tools).toHaveLength(1);
      const streamed = await native.doStream({
        prompt: [{ role: "user", content: [{ type: "text", text: "Hello" }] }],
      });
      let answer = "";
      for await (const event of streamed.stream) {
        if (event.type === "text-delta") answer += event.delta;
      }
      expect(answer).toBe("OpenBot local model works");
      expect(body.options).toMatchObject({ num_ctx: 8192 });
      expect(body.think).toBe(false);
      expect(body.keep_alive).toBe("30m");
      expect(body.stream).toBe(true);
    } finally {
      fetchSpy.mockRestore();
    }
  });
});
