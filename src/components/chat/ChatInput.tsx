import type { KeyboardEvent } from "react";
import { Panel } from "../layout/Panel";
import { theme } from "../../theme/theme";

interface ChatInputProps {
  value: string;
  onChange: (nextValue: string) => void;
  onSubmit: () => void | Promise<void>;
  disabled?: boolean;
}

export function ChatInput({
  value,
  onChange,
  onSubmit,
  disabled = false,
}: ChatInputProps) {
  function handleKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      void onSubmit();
    }
  }

  return (
    <Panel title="Input">
      <section
        aria-label="Message input"
        style={{
          display: "grid",
          gap: theme.spacing.sm,
        }}
      >
        <label>
          <div style={{ marginBottom: theme.spacing.xs }}>Message</div>
          <textarea
            value={value}
            onChange={(event) => onChange(event.target.value)}
            onKeyDown={handleKeyDown}
            rows={5}
            placeholder="Type a message..."
            disabled={disabled}
            className="field-textarea"
          />
        </label>

        <button
          type="button"
          onClick={() => void onSubmit()}
          disabled={disabled}
          className="control-button"
        >
          Send
        </button>
      </section>
    </Panel>
  );
}