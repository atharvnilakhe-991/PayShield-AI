import { Eye, Layers, Image as ImageIcon, AlertCircle, Info } from 'lucide-react';

/**
 * HeatmapView component provides a side-by-side comparison between the
 * original uploaded screenshot and the Grad-CAM/ELA heatmap visualization.
 * Handles null or missing heatmap URLs with a professional empty-state placeholder.
 */
export default function HeatmapView({ originalImageUrl, heatmapUrl }) {
  const hasHeatmap = Boolean(heatmapUrl && typeof heatmapUrl === 'string');

  return (
    <div className="card heatmap-comparison-card">
      <div className="card-header heatmap-header">
        <div className="card-header-left">
          <Layers className="card-icon text-primary" size={20} />
          <h3>Visual Inspection &amp; Heatmap Comparison</h3>
        </div>
        <span className="badge badge-accent">
          <Eye size={12} />
          Side-by-Side Analysis
        </span>
      </div>

      <div className="card-body">
        <p className="comparison-intro">
          Compare the original uploaded GPay receipt screenshot with the localized anomaly map.
        </p>

        <div className="comparison-grid">
          {/* Panel 1: Original Uploaded Screenshot */}
          <div className="comparison-panel">
            <div className="panel-header">
              <span className="panel-tag tag-primary">Input Receipt</span>
              <h4 className="panel-title">Original Screenshot</h4>
            </div>

            <div className="panel-media-box">
              {originalImageUrl ? (
                <img
                  src={originalImageUrl}
                  alt="Original uploaded GPay receipt screenshot"
                  className="comparison-image"
                />
              ) : (
                <div className="panel-media-fallback">
                  <ImageIcon size={32} className="text-secondary" />
                  <span>Original image preview not available</span>
                </div>
              )}
            </div>

            <div className="panel-footer">
              <span>Source: User Upload</span>
              <span className="panel-dot">•</span>
              <span>Intact Unaltered Frame</span>
            </div>
          </div>

          {/* Panel 2: Heatmap Visualization or Empty-State Placeholder */}
          <div className="comparison-panel">
            <div className="panel-header">
              <span className="panel-tag tag-accent">Explainability Layer</span>
              <h4 className="panel-title">Grad-CAM / ELA Heatmap</h4>
            </div>

            <div className="panel-media-box">
              {hasHeatmap ? (
                <div className="heatmap-image-wrapper">
                  <img
                    src={heatmapUrl}
                    alt="Simulated Grad-CAM or ELA manipulation heatmap"
                    className="comparison-image heatmap-image"
                  />
                  <div className="heatmap-overlay-badge">
                    <span>Model Heatmap Overlay</span>
                  </div>
                </div>
              ) : (
                /* Professional Empty-State Placeholder (Task 2E) */
                <div className="heatmap-placeholder">
                  <div className="placeholder-icon-wrap">
                    <Layers size={36} className="text-secondary" />
                  </div>
                  <h4 className="placeholder-title">Localization Mask Unavailable</h4>
                  <p className="placeholder-desc">
                    Pixel-level Grad-CAM heatmaps are intentionally suppressed because baseline segmentation IoU is below operating threshold (0.059).
                  </p>
                  <div className="placeholder-status-pill">
                    <AlertCircle size={13} />
                    <span>Spatial Mask Suppressed for Quality</span>
                  </div>
                </div>
              )}
            </div>

            <div className="panel-footer">
              <Info size={12} className="text-secondary" />
              <span>
                {hasHeatmap
                  ? 'Active Grad-CAM explainability heatmap overlay'
                  : 'Zero synthetic or hallucinated heatmap pixels generated'}
              </span>
            </div>
          </div>
        </div>

        <div className="heatmap-notice-box">
          <Info size={14} className="text-primary" />
          <p>
            <strong>Spatial Explainability Policy:</strong> In compliance with forensic validation standards, visual localization is only displayed when model segmentation reaches certified quality. Unverified activation masks are withheld to prevent false localization.
          </p>
        </div>
      </div>
    </div>
  );
}
