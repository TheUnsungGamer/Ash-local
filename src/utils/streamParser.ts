export interface ParsedStreamChunk {
  tokenText: string;
  isDone: boolean;
}

export interface LmStudioStreamParser {
  push: (chunk: string) => ParsedStreamChunk[];
  flush: () => ParsedStreamChunk[];
  reset: () => void;
}

function extractTokenText(parsedData: unknown): string {
  if (!parsedData || typeof parsedData !== "object") {
    return "";
  }

  const record = parsedData as {
    choices?: Array<{
      delta?: { content?: unknown };
      text?: unknown;
      message?: { content?: unknown };
    }>;
  };

  const firstChoice = record.choices?.[0];

  if (!firstChoice) {
    return "";
  }

  const deltaContent = firstChoice.delta?.content;
  if (typeof deltaContent === "string") {
    return deltaContent;
  }

  const textContent = firstChoice.text;
  if (typeof textContent === "string") {
    return textContent;
  }

  const messageContent = firstChoice.message?.content;
  if (typeof messageContent === "string") {
    return messageContent;
  }

  return "";
}

function parseDataLine(line: string): ParsedStreamChunk | null {
  const trimmed = line.trim();

  if (!trimmed || !trimmed.startsWith("data:")) {
    return null;
  }

  const jsonPayload = trimmed.slice(5).trim();

  if (!jsonPayload) {
    return null;
  }

  if (jsonPayload === "[DONE]") {
    return {
      tokenText: "",
      isDone: true,
    };
  }

  let parsedData: unknown;

  try {
    parsedData = JSON.parse(jsonPayload);
  } catch {
    throw new Error("INCOMPLETE_JSON");
  }

  if (
    parsedData &&
    typeof parsedData === "object" &&
    "error" in parsedData
  ) {
    const errorRecord = parsedData as {
      error?: unknown;
      message?: unknown;
    };

    const directMessage =
      typeof errorRecord.message === "string" ? errorRecord.message : null;

    const nestedErrorMessage =
      errorRecord.error &&
      typeof errorRecord.error === "object" &&
      "message" in errorRecord.error &&
      typeof (errorRecord.error as { message?: unknown }).message === "string"
        ? (errorRecord.error as { message: string }).message
        : null;

    throw new Error(
      directMessage ??
        nestedErrorMessage ??
        "LM Studio returned a stream error."
    );
  }

  const tokenText = extractTokenText(parsedData);

  return {
    tokenText,
    isDone: false,
  };
}

export function createLmStudioStreamParser(): LmStudioStreamParser {
  let buffer = "";

  const processBuffer = (flushPartialLine: boolean): ParsedStreamChunk[] => {
    const parsedChunks: ParsedStreamChunk[] = [];

    if (!buffer) {
      return parsedChunks;
    }

    const normalized = buffer.replace(/\r\n/g, "\n");
    const lines = normalized.split("\n");

    if (!flushPartialLine) {
      buffer = lines.pop() ?? "";
    } else {
      buffer = "";
    }

    for (const rawLine of lines) {
      try {
        const parsed = parseDataLine(rawLine);
        if (parsed) {
          parsedChunks.push(parsed);
        }
      } catch (error) {
        if (
          error instanceof Error &&
          error.message === "INCOMPLETE_JSON" &&
          !flushPartialLine
        ) {
          buffer = `${rawLine}\n${buffer}`;
          break;
        }

        throw error;
      }
    }

    if (flushPartialLine && buffer.trim()) {
      try {
        const parsed = parseDataLine(buffer);
        if (parsed) {
          parsedChunks.push(parsed);
        }
      } catch (error) {
        if (
          !(error instanceof Error && error.message === "INCOMPLETE_JSON")
        ) {
          throw error;
        }
      } finally {
        buffer = "";
      }
    }

    return parsedChunks;
  };

  return {
    push(chunk: string): ParsedStreamChunk[] {
      if (!chunk) {
        return [];
      }

      buffer += chunk;
      return processBuffer(false);
    },

    flush(): ParsedStreamChunk[] {
      if (!buffer.trim()) {
        buffer = "";
        return [];
      }

      return processBuffer(true);
    },

    reset(): void {
      buffer = "";
    },
  };
}