import { ChatWindow } from "./ChatWindow";
import { ChatInput } from "./ChatInput";
import { MapPanel } from "../system/MapPanel";
import type { ChatMessage } from "../../types/chat";

interface ChatSectionProps {
  messages: ChatMessage[];
  isStreaming: boolean;
  draftMessage: string;
  onDraftChange: (value: string) => void;
  onSubmit: () => void;
  onCopyMessage?: (message: ChatMessage) => void;
  onSpeakMessage?: (message: ChatMessage) => void;
}

export function ChatSection({
  messages,
  isStreaming,
  draftMessage,
  onDraftChange,
  onSubmit,
  onCopyMessage,
  onSpeakMessage,
}: ChatSectionProps) {
  return (
    <section className="stack-md" aria-label="Chat section">
      <ChatWindow
        messages={messages}
        isStreaming={isStreaming}
        onCopyMessage={onCopyMessage}
        onSpeakMessage={onSpeakMessage}
      />

      <MapPanel />

      <ChatInput
        value={draftMessage}
        onChange={onDraftChange}
        onSubmit={onSubmit}
        disabled={isStreaming}
      />
    </section>
  );
}