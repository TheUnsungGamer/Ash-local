interface ErrorPanelProps {
  message: string;
}

export function ErrorPanel({ message }: ErrorPanelProps) {
  return (
    <section aria-label="Chat error" className="error-panel">
      {message}
    </section>
  );
}