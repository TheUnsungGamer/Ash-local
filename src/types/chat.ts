export type MessageRole = "user" | "assistant" | "system";

export interface ChatMessage {
  id: string;
  role: MessageRole;
  content: string;
  createdAt: number;
}

export interface ChatState {
  messages: ChatMessage[];
  isStreaming: boolean;
  errorMessage: string | null;
  selectedModel: string;
}

export interface SendMessageParams {
  text: string;
}