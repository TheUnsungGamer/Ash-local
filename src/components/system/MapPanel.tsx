import { Panel } from "../layout/Panel";

export function MapPanel() {
  return (
    <Panel title="Map">
      <section aria-label="Map panel" className="map-panel">
        <div className="map-panel__glow" />

        <div className="map-crosshair">
          <div className="map-crosshair__vertical-top" />
          <div className="map-crosshair__vertical-bottom" />
          <div className="map-crosshair__horizontal-left" />
          <div className="map-crosshair__horizontal-right" />
        </div>

        <div className="map-panel__label">
          <div>Map Module Standby</div>
          <div className="map-panel__sublabel">Local Overlay Pending</div>
        </div>
      </section>
    </Panel>
  );
}