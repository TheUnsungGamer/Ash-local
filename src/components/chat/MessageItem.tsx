import type { ChatMessage } from "../../types/chat";

interface MessageItemProps {
  message: ChatMessage;
  onCopy?: (message: ChatMessage) => void;
  onSpeak?: (message: ChatMessage) => void;
}

export function MessageItem({
  message,
  onCopy,
  onSpeak,
}: MessageItemProps) {
  const isAssistantMessage = message.role === "assistant";

  return (
    <article className={`message-item message-item--${message.role}`}>
      <header className="message-header">
        <span className="message-role">{message.role}</span>

        {isAssistantMessage && (
          <div className="message-actions">
            {onCopy && (
              <button
                type="button"
                className="control-button message-action-button"
                onClick={() => onCopy(message)}
                aria-label="Copy assistant message"
              >
                Copy
              </button>
            )}

            {onSpeak && (
              <button
                type="button"
                className="control-button message-action-button"
                onClick={() => onSpeak(message)}
                aria-label="Speak assistant message"
              >
                Speak
              </button>
            )}
          </div>
        )}
      </header>

      <p className="message-content" style={{ whiteSpace: "pre-wrap" }}>
        {message.content}
      </p>
    </article>
  );
}