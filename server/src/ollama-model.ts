import { createOllama } from "ollama-ai-provider-v2";

/** Opt in to Ollama's native API so context size is sent with each request. */
export function ollamaModelForEnvironment(
  model: { provider: string; defaultModel: string },
  environment: Record<string, string | undefined> = process.env,
) {
  const configured = environment.OPENBOT_OLLAMA_CONTEXT_LENGTH?.trim();
  if (!configured) return undefined;
  const contextLength = Number(configured);
  if (!Number.isSafeInteger(contextLength) || contextLength < 1) {
    throw new Error(
      "OPENBOT_OLLAMA_CONTEXT_LENGTH must be a positive integer.",
    );
  }
  if (model.provider !== "openai" || !environment.OPENAI_BASE_URL?.trim()) {
    throw new Error(
      "Ollama requires the OpenAI-compatible provider and OPENAI_BASE_URL.",
    );
  }
  let base: URL;
  try {
    base = new URL(environment.OPENAI_BASE_URL.trim());
  } catch {
    throw new Error("Ollama requires a valid OPENAI_BASE_URL.");
  }
  if (!/^https?:$/.test(base.protocol) || !/\/v1\/?$/.test(base.pathname)) {
    throw new Error(
      "Ollama requires an HTTP(S) OPENAI_BASE_URL ending in /v1.",
    );
  }
  if (base.username || base.password || base.search || base.hash) {
    throw new Error(
      "Ollama OPENAI_BASE_URL must not contain credentials, a query or a fragment.",
    );
  }
  base.pathname = base.pathname.replace(/\/v1\/?$/, "/api");
  const thinking = environment.OPENBOT_OLLAMA_THINK?.trim();
  if (thinking && thinking !== "true" && thinking !== "false") {
    throw new Error("OPENBOT_OLLAMA_THINK must be true or false.");
  }
  const keepAlive = environment.OPENBOT_OLLAMA_KEEP_ALIVE?.trim();
  if (keepAlive && !/^[1-9]\d*(?:ms|s|m|h)$/.test(keepAlive)) {
    throw new Error(
      "OPENBOT_OLLAMA_KEEP_ALIVE must be a positive duration such as 30m.",
    );
  }
  const native = createOllama({
    baseURL: base.toString().replace(/\/$/, ""),
    // Provider v3.6 does not expose keep_alive in its options schema. Use its supported fetch seam.
    ...(keepAlive
      ? {
          fetch: Object.assign(
            (input: RequestInfo | URL, init?: RequestInit) => {
              if (typeof init?.body !== "string") return fetch(input, init);
              return fetch(input, {
                ...init,
                body: JSON.stringify({
                  ...JSON.parse(init.body),
                  keep_alive: keepAlive,
                }),
              });
            },
            { preconnect: fetch.preconnect },
          ),
        }
      : {}),
  }).chat(model.defaultModel);
  const withOptions = (options: Parameters<typeof native.doGenerate>[0]) => ({
    ...options,
    providerOptions: {
      ...options.providerOptions,
      ollama: {
        ...options.providerOptions?.ollama,
        options: {
          ...((options.providerOptions?.ollama?.options as
            | Record<string, unknown>
            | undefined) ?? {}),
          num_ctx: contextLength,
        },
        ...(thinking ? { think: thinking === "true" } : {}),
      },
    },
  });
  return {
    specificationVersion: native.specificationVersion,
    provider: native.provider,
    modelId: native.modelId,
    supportedUrls: native.supportedUrls,
    doGenerate: (options: Parameters<typeof native.doGenerate>[0]) =>
      native.doGenerate(withOptions(options)),
    doStream: (options: Parameters<typeof native.doStream>[0]) =>
      native.doStream(withOptions(options)),
  };
}
