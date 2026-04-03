import { ModelSelector } from "../system/ModelSelector";
import { AudioControls } from "./AudioControls";
import { StatsPanel } from "../system/StatsPanel";
import { LogPanel } from "../system/LogPanel";
import { ErrorPanel } from "../system/ErrorPanel";
import type { AudioState } from "../../types/audio";

interface SidebarProps {
  availableModels: string[];
  selectedModel: string;
  isStreaming: boolean;
  errorMessage?: string;
  logs: string[];
  audioState: AudioState;
  onSelectModel: (model: string) => void;
  onToggleVoice: () => void;
  onToggleMode: () => void;
  onPauseAudio: () => void;
  onResumeAudio: () => void;
  onStopAudio: () => void;
  messageCount: number;
}

export function Sidebar({
  availableModels,
  selectedModel,
  isStreaming,
  errorMessage,
  logs,
  audioState,
  onSelectModel,
  onToggleVoice,
  onToggleMode,
  onPauseAudio,
  onResumeAudio,
  onStopAudio,
  messageCount,
}: SidebarProps) {
  return (
    <section className="stack-md" aria-label="Sidebar controls">
      <ModelSelector
        availableModels={availableModels}
        selectedModel={selectedModel}
        onChange={onSelectModel}
        disabled={isStreaming}
      />

      <AudioControls
        audioState={audioState}
        onToggleVoice={onToggleVoice}
        onToggleMode={onToggleMode}
        onPause={onPauseAudio}
        onResume={onResumeAudio}
        onStop={onStopAudio}
      />

      <StatsPanel
        stats={{
          messageCount,
          isStreaming,
          selectedModel,
          audioStatus: audioState.status,
          voiceMode: audioState.voiceMode,
        }}
      />

      <LogPanel logs={logs} />

      {errorMessage && <ErrorPanel message={errorMessage} />}
    </section>
  );
}