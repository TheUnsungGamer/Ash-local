import { Panel } from "../layout/Panel";

interface LogPanelProps {
  logs: string[];
}

export function LogPanel({ logs }: LogPanelProps) {
  return (
    <Panel title="Log Stream">
      <section
        aria-label="System log stream"
        className="stack-sm log-stream"
      >
        {logs.length === 0 ? (
          <p style={{ margin: 0, opacity: 0.7 }}>No log entries yet.</p>
        ) : (
          logs.map((logLine, index) => (
            <p
              key={`${index}-${logLine}`}
              className="log-line"
            >
              {logLine}
            </p>
          ))
        )}
      </section>
    </Panel>
  );
}