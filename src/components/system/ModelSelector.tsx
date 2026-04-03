import { Panel } from "../layout/Panel";
import { theme } from "../../theme/theme";

interface ModelSelectorProps {
  availableModels: string[];
  selectedModel: string;
  onChange: (modelName: string) => void;
  disabled?: boolean;
}

export function ModelSelector({
  availableModels,
  selectedModel,
  onChange,
  disabled = false,
}: ModelSelectorProps) {
  return (
    <Panel title="Model">
      <section
        aria-label="Model selection"
        style={{
          display: "grid",
          gap: theme.spacing.xs,
        }}
      >
        <label>
          <div style={{ marginBottom: theme.spacing.xs }}>Active model</div>
          <select
            value={selectedModel}
            onChange={(event) => onChange(event.target.value)}
            disabled={disabled}
            className="field-select"
          >
            {availableModels.length === 0 ? (
              <option value="">Loading models...</option>
            ) : (
              availableModels.map((modelName) => (
                <option key={modelName} value={modelName}>
                  {modelName}
                </option>
              ))
            )}
          </select>
        </label>
      </section>
    </Panel>
  );
}