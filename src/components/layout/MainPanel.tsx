import type { ReactNode } from "react";

interface MainPanelProps {
  children: ReactNode;
}

export function MainPanel({ children }: MainPanelProps) {
  return <section className="main-panel">{children}</section>;
}