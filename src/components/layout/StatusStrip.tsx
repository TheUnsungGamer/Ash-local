interface StatusStripProps {
  selectedModel: string;
  audioStatus: string;
  voiceMode: string;
}

export function StatusStrip({
  selectedModel,
  audioStatus,
  voiceMode,
}: StatusStripProps) {
  return (
    <section className="status-strip" aria-label="System status strip">
      <span>SYS: ONLINE</span>
      <span>MODEL: {selectedModel || "NONE"}</span>
      <span>AUDIO: {audioStatus.toUpperCase()}</span>
      <span>VOICE: {voiceMode.toUpperCase()}</span>
      <span>LOCAL: TRUE</span>
    </section>
  );
}