import { CheckCircle2, AlertOctagon, HelpCircle, RefreshCw, Sparkles } from 'lucide-react';
import { VERDICTS } from '../services/apiContract';

/**
 * MockResultPreview renders a compact, lightweight summary of the mock
 * analysis result for Phase 2 without building the complete Phase 3 dashboard.
 */
export default function MockResultPreview({ result, onReset }) {
  if (!result) return null;

  const { verdict, confidence, message, request_id } = result;
  const confidencePercent = Math.round((confidence || 0) * 100);

  // Determine semantic color theme based on verdict
  let statusClass = 'verdict-suspicious';
  let VerdictIcon = HelpCircle;
  let verdictTitle = 'Suspicious Receipt';

  if (verdict === VERDICTS.REAL) {
    statusClass = 'verdict-real';
    VerdictIcon = CheckCircle2;
    verdictTitle = 'Authentic Receipt';
  } else if (verdict === VERDICTS.FAKE) {
    statusClass = 'verdict-fake';
    VerdictIcon = AlertOctagon;
    verdictTitle = 'Manipulated / Fake Receipt';
  }

  return (
    <div className={`card mock-result-card ${statusClass}`} aria-live="polite">
      <div className="mock-result-header">
        <div className="mock-result-badge">
          <Sparkles size={14} />
          <span>Simulated Demo Result (Phase 2 Mock)</span>
        </div>
        <span className="mock-request-id">Request ID: {request_id}</span>
      </div>

      <div className="mock-result-body">
        <div className="verdict-banner">
          <div className="verdict-icon-box">
            <VerdictIcon size={32} />
          </div>

          <div className="verdict-info">
            <div className="verdict-row">
              <h3 className="verdict-heading">{verdictTitle}</h3>
              <span className="verdict-tag">{verdict}</span>
            </div>
            <p className="verdict-message">{message}</p>
          </div>

          <div className="confidence-box">
            <span className="confidence-label">Confidence</span>
            <span className="confidence-value">{confidencePercent}%</span>
          </div>
        </div>

        <div className="mock-result-footer">
          <p className="mock-disclaimer">
            * Note: This is simulated demo output adhering to the confirmed CBL-04 API contract. Full heatmap overlays and OCR extraction panels will be unlocked in Phase 3.
          </p>

          <button
            type="button"
            className="btn btn-secondary btn-sm"
            onClick={onReset}
          >
            <RefreshCw size={14} />
            Test Another Screenshot
          </button>
        </div>
      </div>
    </div>
  );
}

