import type { AudioState } from "../../types/audio";
import { theme } from "../../theme/theme";
import { Panel } from "../layout/Panel";

interface AudioControlsProps {
  audioState: AudioState;
  onToggleVoice: () => void;
  onToggleMode: () => void;
  onPause: () => void;
  onResume: () => void;
  onStop: () => void;
}

export function AudioControls({
  audioState,
  onToggleVoice,
  onToggleMode,
  onPause,
  onResume,
  onStop,
}: AudioControlsProps) {
  return (
    <Panel title="Audio">
      <section
        aria-label="Audio controls"
        style={{
          display: "grid",
          gap: theme.spacing.md,
        }}
      >
        <div style={{ display: "flex", gap: theme.spacing.sm, flexWrap: "wrap" }}>
          <button type="button" onClick={onToggleVoice} className="control-button">
            Voice: {audioState.isVoiceEnabled ? "ON" : "OFF"}
          </button>

          <button type="button" onClick={onToggleMode} className="control-button">
            Mode: {audioState.voiceMode === "tech_priest" ? "TECH-PRIEST" : "NORMAL"}
          </button>

          <button type="button" onClick={onPause} className="control-button">
            Pause
          </button>

          <button type="button" onClick={onResume} className="control-button">
            Resume
          </button>

          <button type="button" onClick={onStop} className="control-button">
            Stop
          </button>
        </div>

        <section aria-label="Audio status" className="audio-status-box stack-sm">
          <div>Status: {audioState.status}</div>
          <div>Voice enabled: {String(audioState.isVoiceEnabled)}</div>
          <div>Mode: {audioState.voiceMode}</div>
          <div>Active message ID: {audioState.activeMessageId ?? "none"}</div>
          <div>Audio error: {audioState.errorMessage ?? "none"}</div>
        </section>
      </section>
    </Panel>
  );
}