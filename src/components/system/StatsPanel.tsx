import { Panel } from "../layout/Panel";

interface StatsPanelProps {
  stats: {
    messageCount: number;
    isStreaming: boolean;
    selectedModel: string;
    audioStatus: string;
    voiceMode: string;
  };
}

export function StatsPanel({ stats }: StatsPanelProps) {
  return (
    <Panel title="Live Stats">
      <section
        aria-label="System stats"
        className="stack-sm stats-panel"
      >
        <StatRow label="Messages" value={String(stats.messageCount)} />
        <StatRow label="Streaming" value={stats.isStreaming ? "YES" : "NO"} />
        <StatRow label="Audio" value={stats.audioStatus.toUpperCase()} />
        <StatRow label="Voice Mode" value={stats.voiceMode.toUpperCase()} />
        <StatRow label="Model" value={stats.selectedModel || "NONE"} />
      </section>
    </Panel>
  );
}

interface StatRowProps {
  label: string;
  value: string;
}

function StatRow({ label, value }: StatRowProps) {
  return (
    <div className="stat-row">
      <span className="stat-label">{label}</span>
      <span className="stat-value">{value}</span>
    </div>
  );
}