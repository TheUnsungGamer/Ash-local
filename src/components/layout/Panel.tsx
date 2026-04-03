import type { ReactNode } from "react";

interface PanelProps {
  title: string;
  children: ReactNode;
}

export function Panel({ title, children }: PanelProps) {
  return (
    <section className="panel-wrapper">
      <div className="panel-accent" />
      <div className="panel">
        <header className="panel-header">
          <span>{title}</span>
          <span className="panel-sys-tag">SYS</span>
        </header>

        <div className="panel-body">{children}</div>
      </div>
    </section>
  );
}