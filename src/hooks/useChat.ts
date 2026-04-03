import {
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";
import type { Dispatch, SetStateAction } from "react";
import type { ChatMessage } from "../types/chat";
import {
  checkLmStudioHealth,
  fetchAvailableModels,
  streamChatCompletion,
} from "../services/lmStudioService";
import type { LmStudioRequestMessage } from "../services/lmStudioService";
import { createLmStudioStreamParser } from "../utils/streamParser";
import { VOICE_PROFILES } from "../config/voiceProfiles";

interface UseChatOptions {
  initialMessages?: ChatMessage[];
  onAssistantMessageComplete?: (message: ChatMessage) => void | Promise<void>;
}

interface UseChatResult {
  messages: ChatMessage[];
  availableModels: string[];
  selectedModel: string;
  isStreaming: boolean;
  errorMessage: string | null;
  setSelectedModel: (model: string) => void;
  sendMessage: (text: string) => Promise<void>;
  setMessages: Dispatch<SetStateAction<ChatMessage[]>>;
}

function createMessage(role: ChatMessage["role"], content: string): ChatMessage {
  return {
    id: crypto.randomUUID(),
    role,
    content,
    createdAt: Date.now(),
  };
}

async function retryAsync<T>(
  fn: () => Promise<T>,
  retries = 3,
  delayMs = 1000
): Promise<T> {
  let lastError: unknown;

  for (let attempt = 0; attempt < retries; attempt++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error;

      if (attempt < retries - 1) {
        await new Promise((res) => setTimeout(res, delayMs * (attempt + 1)));
      }
    }
  }

  throw lastError;
}

export function useChat({
  initialMessages = [],
  onAssistantMessageComplete,
}: UseChatOptions = {}): UseChatResult {
  const [messages, setMessages] = useState<ChatMessage[]>(initialMessages);
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const [selectedModel, setSelectedModel] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const messagesRef = useRef<ChatMessage[]>(initialMessages);

  useEffect(() => {
    messagesRef.current = messages;
  }, [messages]);

  useEffect(() => {
    async function loadModels() {
      try {
        const modelIds = await fetchAvailableModels();
        setAvailableModels(modelIds);

        if (modelIds.length > 0) {
          setSelectedModel(modelIds[0]);
        }
      } catch (error) {
        setErrorMessage(
          error instanceof Error
            ? error.message
            : "Failed to load available models."
        );
      }
    }

    void loadModels();
  }, []);

  const sendMessage = useCallback(
    async (text: string) => {
      const trimmedText = text.trim();

      if (!trimmedText || isStreaming || !selectedModel) {
        return;
      }

      setErrorMessage(null);
      setIsStreaming(true);

      const userMessage = createMessage("user", trimmedText);
      const assistantMessageId = crypto.randomUUID();
      const assistantCreatedAt = Date.now();

      const currentConversation: LmStudioRequestMessage[] = messagesRef.current
        .filter((message) => message.role === "user" || message.role === "assistant")
        .map((message) => ({
          role: message.role,
          content: message.content,
        }));

      const nextConversation: LmStudioRequestMessage[] = [
  ...currentConversation,
  {
    role: "user",
    content: `${VOICE_PROFILES.VERITY.systemPrompt}\n\nUser: ${userMessage.content}`,
  },
];

      setMessages((previousMessages) => [
        ...previousMessages,
        userMessage,
        {
          id: assistantMessageId,
          role: "assistant",
          content: "",
          createdAt: assistantCreatedAt,
        },
      ]);

      try {
        await retryAsync(() => checkLmStudioHealth(), 2, 500);

        const responseStream = await retryAsync(
          () =>
            streamChatCompletion({
              model: selectedModel,
              messages: nextConversation,
            }),
          3,
          1000
        );

        const reader = responseStream.getReader();
const decoder = new TextDecoder("utf-8");
const parser = createLmStudioStreamParser();

let fullAssistantText = "";

// 🔥 Heartbeat
let heartbeatTimeout: number | null = null;

const restartHeartbeat = () => {
  if (heartbeatTimeout !== null) {
    window.clearTimeout(heartbeatTimeout);
  }

  heartbeatTimeout = window.setTimeout(() => {
    if (onAssistantMessageComplete) {
      void onAssistantMessageComplete({
        id: assistantMessageId,
        role: "assistant",
        content: "Connection Failed",
        createdAt: assistantCreatedAt,
      });
    }
  }, 5000);
};

restartHeartbeat();
        try {
          while (true) {
            const { value, done } = await reader.read();

            if (done) {
              break;
            }

            if (!value || value.byteLength === 0) {
              continue;
            }

            const decodedChunk = decoder.decode(value, { stream: true });
            const parsedEntries = parser.push(decodedChunk);

            for (const entry of parsedEntries) {
              if (entry.isDone) {
                continue;
              }

              if (entry.tokenText.length > 0) {
                restartHeartbeat();
                fullAssistantText += entry.tokenText;

                setMessages((previousMessages) =>
                  previousMessages.map((message) =>
                    message.id === assistantMessageId
                      ? { ...message, content: fullAssistantText }
                      : message
                  )
                );
              }
            }
          }

          const trailingEntries = parser.flush();

          for (const entry of trailingEntries) {
            if (entry.isDone) {
              continue;
            }

            if (entry.tokenText.length > 0) {
              fullAssistantText += entry.tokenText;

              setMessages((previousMessages) =>
                previousMessages.map((message) =>
                  message.id === assistantMessageId
                    ? { ...message, content: fullAssistantText }
                    : message
                )
              );
            }
          }
        }   finally {
             if (heartbeatTimeout !== null) {
              window.clearTimeout(heartbeatTimeout);
    }

             parser.reset();
             reader.releaseLock();
                    }

        const finalizedContent = fullAssistantText.trim();

        if (finalizedContent.length === 0) {
          throw new Error("LM Studio returned an empty streamed response.");
        }

        const completedAssistantMessage: ChatMessage = {
          id: assistantMessageId,
          role: "assistant",
          content: finalizedContent,
          createdAt: assistantCreatedAt,
        };

        setMessages((previousMessages) =>
          previousMessages.map((message) =>
            message.id === assistantMessageId ? completedAssistantMessage : message
          )
        );

        if (onAssistantMessageComplete) {
          await onAssistantMessageComplete(completedAssistantMessage);
        }
      } catch (error) {
        setErrorMessage(
          error instanceof Error ? error.message : "Failed to send message."
        );

        setMessages((previousMessages) =>
          previousMessages.filter((message) => message.id !== assistantMessageId)
        );
      } finally {
        setIsStreaming(false);
      }
    },
    [isStreaming, onAssistantMessageComplete, selectedModel]
  );

  return {
    messages,
    availableModels,
    selectedModel,
    isStreaming,
    errorMessage,
    setSelectedModel,
    sendMessage,
    setMessages,
  };
}