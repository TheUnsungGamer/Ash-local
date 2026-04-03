import { MessageItem } from "./MessageItem";
import type { ChatMessage } from "../../types/chat";
import { Panel } from "../layout/Panel";

interface ChatWindowProps {
  messages: ChatMessage[];
  isStreaming: boolean;
  onCopyMessage?: (message: ChatMessage) => void;
  onSpeakMessage?: (message: ChatMessage) => void;
}

export function ChatWindow({
  messages,
  isStreaming,
  onCopyMessage,
  onSpeakMessage,
}: ChatWindowProps) {
  return (
    <Panel title="Console">
      <section
        aria-label="Chat messages"
        className="stack-md"
        style={{ minHeight: "320px" }}
      >
        {messages.length === 0 ? (
          <p style={{ margin: 0, opacity: 0.7 }}>No messages yet.</p>
        ) : (
          messages.map((message) => (
            <MessageItem
              key={message.id}
              message={message}
              onCopy={onCopyMessage}
              onSpeak={onSpeakMessage}
            />
          ))
        )}

        {isStreaming && (
          <p style={{ margin: 0, opacity: 0.8 }}>
            Assistant is responding...
          </p>
        )}
      </section>
    </Panel>
  );
}