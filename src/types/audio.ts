export type AudioStatus =
  | "idle"
  | "playing"
  | "paused"
  | "stopped"
  | "interrupted"
  | "error";

export type VoiceMode = "normal" | "tech_priest";

export interface AudioState {
  status: AudioStatus;
  activeText: string | null;
  activeMessageId: string | null;
  isVoiceEnabled: boolean;
  voiceMode: VoiceMode;
  errorMessage: string | null;
}

export interface AudioController {
  audioState: AudioState;
  speak: (text: string, messageId?: string) => Promise<void>;
  pause: () => void;
  resume: () => void;
  stop: () => void;
  interrupt: () => void;
  toggleVoice: () => void;
  toggleMode: () => void;
}