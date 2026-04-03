interface AppHeaderProps {
  title: string;
  subtitle?: string;
}

export function AppHeader({ title, subtitle }: AppHeaderProps) {
  return (
    <header className="app-header">
      <h1 className="app-title">{title}</h1>
      {subtitle && <p className="app-subtitle">{subtitle}</p>}
    </header>
  );
}