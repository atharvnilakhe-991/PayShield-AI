import { CheckCircle2, Crosshair, MapPin, AlertCircle } from 'lucide-react';

/**
 * SuspiciousRegionsCard displays localized regions of interest detected
 * during tamper inspection, or an honest empty/unavailable state if none are found.
 */
export default function SuspiciousRegionsCard({ regions, manipulationDetected, isLive = true }) {
  const regionList = Array.isArray(regions) ? regions : [];
  const hasSuspiciousRegions = regionList.length > 0;

  return (
    <div className="card suspicious-regions-card">
      <div className="card-header suspicious-card-header">
        <div className="card-header-left">
          <Crosshair className="card-icon text-accent" size={20} />
          <h3>Suspicious Region Analysis</h3>
        </div>
        <span
          className={`badge ${
            hasSuspiciousRegions
              ? 'badge-warning'
              : manipulationDetected
              ? 'badge-warning'
              : 'badge-accent'
          }`}
        >
          {hasSuspiciousRegions
            ? `${regionList.length} Region${regionList.length > 1 ? 's' : ''} Flagged`
            : manipulationDetected
            ? 'Localization Unavailable'
            : 'Integrity Verified'}
        </span>
      </div>

      <div className="card-body">
        {hasSuspiciousRegions ? (
          <div className="regions-content">
            <p className="regions-intro">
              {isLive
                ? 'Localized inconsistencies identified by semantic consistency & layout inspection:'
                : 'Localized anomalies identified by artifact detection filters:'}
            </p>

            <div className="regions-list">
              {regionList.map((region, index) => {
                const scorePercent = Math.round((region.score || 0) * 100);
                const hasPixels = region.x !== null && region.x !== undefined && region.y !== null && region.y !== undefined;
                const hasNormalized = Boolean(region.box_normalized);
                const hasDimensions = region.width !== null && region.width !== undefined && region.height !== null && region.height !== undefined;

                return (
                  <div key={index} className="region-item">
                    <div className="region-header-row">
                      <div className="region-label-wrap">
                        <span className="region-index-pill">#{index + 1}</span>
                        <h4 className="region-title">{region.label || 'Unlabeled Region'}</h4>
                      </div>
                      <div className="region-score-badge">
                        <span className="score-label">Anomaly Score:</span>
                        <span className="score-value">{scorePercent}%</span>
                      </div>
                    </div>

                    <div className="region-coords-grid">
                      <div className="coord-box">
                        <span className="coord-label">Top-Left (X, Y)</span>
                        <span className="coord-val">
                          <MapPin size={12} className="text-primary" />
                          {hasPixels
                            ? `X: ${region.x}px, Y: ${region.y}px`
                            : hasNormalized
                            ? `X: ${Math.round(region.box_normalized.xmin * 100)}%, Y: ${Math.round(region.box_normalized.ymin * 100)}%`
                            : 'Coordinates not localized'}
                        </span>
                      </div>

                      <div className="coord-box">
                        <span className="coord-label">Dimensions (W × H)</span>
                        <span className="coord-val">
                          {hasDimensions
                            ? `${region.width}px × ${region.height}px`
                            : hasNormalized
                            ? `${Math.round(region.box_normalized.width * 100)}% × ${Math.round(region.box_normalized.height * 100)}%`
                            : 'Dimensions unavailable'}
                        </span>
                      </div>

                      <div className="coord-box">
                        <span className="coord-label">Area</span>
                        <span className="coord-val">
                          {hasDimensions && region.width && region.height
                            ? `${(region.width * region.height).toLocaleString()} px²`
                            : hasNormalized
                            ? `${(region.box_normalized.width * region.box_normalized.height * 100).toFixed(1)}% frame`
                            : 'N/A'}
                        </span>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>

            <p className="regions-disclaimer">
              {isLive
                ? '* Localized regions reflect verified OCR and semantic layout evidence. Pixel-level CNN mask segmentation remains suppressed for certified quality.'
                : '* Simulated regions for offline demonstration.'}
            </p>
          </div>
        ) : manipulationDetected ? (
          <div className="regions-empty-state">
            <div className="empty-icon-wrap" style={{ background: 'rgba(245, 158, 11, 0.12)' }}>
              <AlertCircle size={36} style={{ color: '#f59e0b' }} />
            </div>
            <h4>Localization Unavailable</h4>
            <p>
              Global tampering signals were flagged by CNN/ensemble models, but spatial bounding box
              localization is suppressed to comply with quality safeguards (test IoU 0.059 below operating threshold).
            </p>
          </div>
        ) : (
          <div className="regions-empty-state">
            <div className="empty-icon-wrap">
              <CheckCircle2 size={36} className="text-success" />
            </div>
            <h4>No Suspicious Regions Detected</h4>
            <p>
              The structural analysis did not identify any localized font splicing,
              compression anomalies, or manipulated pixel clusters in this screenshot.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
