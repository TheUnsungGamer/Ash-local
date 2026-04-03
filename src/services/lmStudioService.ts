const LM_STUDIO_BASE_URL = "/lmstudio";

const REQUEST_START_TIMEOUT_MS = 30_000;
const HEALTH_CHECK_TIMEOUT_MS = 5_000;
const STREAM_IDLE_TIMEOUT_MS = 20_000;

export type LmStudioRequestMessage = {
  role: "system" | "user" | "assistant";
  content: string;
};

interface LmStudioModelRecord {
  id: string;
}

interface LmStudioModelsResponse {
  data: LmStudioModelRecord[];
}

interface StreamChatCompletionParams {
  model: string;
  messages: LmStudioRequestMessage[];
  temperature?: number;
  maxTokens?: number;
}

function createTimeoutError(message: string): Error {
  const error = new Error(message);
  error.name = "AbortError";
  return error;
}

async function fetchWithTimeout(
  input: RequestInfo | URL,
  init?: RequestInit,
  timeoutMs = 10_000
): Promise<Response> {
  const controller = new AbortController();
  const timeoutId = window.setTimeout(() => controller.abort(), timeoutMs);

  try {
    return await fetch(input, {
      ...init,
      signal: controller.signal,
    });
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") {
      throw createTimeoutError("Request timed out.");
    }

    throw error;
  } finally {
    window.clearTimeout(timeoutId);
  }
}

function wrapStreamWithIdleTimeout(
  sourceStream: ReadableStream<Uint8Array>,
  idleTimeoutMs = STREAM_IDLE_TIMEOUT_MS
): ReadableStream<Uint8Array> {
  return new ReadableStream<Uint8Array>({
    start(controller) {
      const reader = sourceStream.getReader();
      let timeoutId: number | null = null;
      let isClosed = false;

      const clearIdleTimer = () => {
        if (timeoutId !== null) {
          window.clearTimeout(timeoutId);
          timeoutId = null;
        }
      };

      const startIdleTimer = () => {
        clearIdleTimer();

        timeoutId = window.setTimeout(async () => {
          if (isClosed) {
            return;
          }

          isClosed = true;

          try {
            await reader.cancel("LM Studio stream idle timeout.");
          } catch {
            // Ignore cancellation errors.
          }

          controller.error(
            new Error(
              `LM Studio stream stalled for more than ${idleTimeoutMs}ms.`
            )
          );
        }, idleTimeoutMs);
      };

      const pump = async (): Promise<void> => {
        startIdleTimer();

        try {
          while (!isClosed) {
            const result = await reader.read();
            clearIdleTimer();

            if (result.done) {
              isClosed = true;
              controller.close();
              break;
            }

            if (result.value) {
              controller.enqueue(result.value);
            }

            startIdleTimer();
          }
        } catch (error) {
          if (!isClosed) {
            isClosed = true;
            controller.error(
              error instanceof Error
                ? error
                : new Error("LM Studio stream failed.")
            );
          }
        } finally {
          clearIdleTimer();
          reader.releaseLock();
        }
      };

      void pump();
    },

    async cancel(reason) {
      try {
        await sourceStream.cancel(reason);
      } catch {
        // Ignore cancel errors.
      }
    },
  });
}

export async function checkLmStudioHealth(): Promise<void> {
  let response: Response;

  try {
    response = await fetchWithTimeout(
      `${LM_STUDIO_BASE_URL}/models`,
      {
        method: "GET",
      },
      HEALTH_CHECK_TIMEOUT_MS
    );
  } catch (error) {
    if (error instanceof Error) {
      throw new Error(`LM Studio health check failed: ${error.message}`);
    }

    throw new Error("LM Studio health check failed.");
  }

  if (!response.ok) {
    const errorText = await response.text().catch(() => "");
    throw new Error(
      `LM Studio health check failed: ${response.status}${
        errorText ? ` - ${errorText}` : ""
      }`
    );
  }
}

export async function fetchAvailableModels(): Promise<string[]> {
  let response: Response;

  try {
    response = await fetchWithTimeout(`${LM_STUDIO_BASE_URL}/models`);
  } catch {
    throw new Error("Could not reach LM Studio model server.");
  }

  if (!response.ok) {
    const errorText = await response.text().catch(() => "");
    throw new Error(
      `Failed to load models: ${response.status}${
        errorText ? ` - ${errorText}` : ""
      }`
    );
  }

  const data: LmStudioModelsResponse = await response.json();
  return data.data.map((model) => model.id);
}

export async function streamChatCompletion({
  model,
  messages,
  temperature = 0.7,
  maxTokens = 500,
}: StreamChatCompletionParams): Promise<ReadableStream<Uint8Array>> {
  let response: Response;

  try {
    response = await fetchWithTimeout(
      `${LM_STUDIO_BASE_URL}/chat/completions`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          model,
          messages: messages.map((message) => ({
            role: message.role,
            content: message.content,
          })),
          temperature,
          max_tokens: maxTokens,
          stream: true,
        }),
      },
      REQUEST_START_TIMEOUT_MS
    );
  } catch (error) {
    console.error("LM Studio fetch error:", error);

    if (error instanceof Error && error.name === "AbortError") {
      throw new Error("LM Studio did not begin responding in time.");
    }

    throw new Error("Could not connect to LM Studio chat server.");
  }

  if (!response.ok) {
    const errorText = await response.text().catch(() => "");
    console.error("LM Studio error response:", errorText);
    throw new Error(
      `Chat request failed: ${response.status}${
        errorText ? ` - ${errorText}` : ""
      }`
    );
  }

  if (!response.body) {
    throw new Error("LM Studio response did not include a readable stream.");
  }

  return wrapStreamWithIdleTimeout(response.body, STREAM_IDLE_TIMEOUT_MS);
}