import { Shield, CheckCircle2, AlertOctagon, HelpCircle, Info } from 'lucide-react';

/**
 * ConfidenceScore component displays the uncalibrated decision score or
 * indeterminate ambiguity status with strict semantic guardrails.
 */
export default function ConfidenceScore({
  confidence = null,
  decisionScore = null,
  uncertaintyScore = null,
  verdict = null,
  scoreType = null,
  _pipelineStatus = null,
}) {
  const normVerdict = (verdict || '').toUpperCase();
  const isGenuine = normVerdict === 'GENUINE' || normVerdict === 'REAL';
  const isEdited = normVerdict === 'EDITED' || normVerdict === 'FAKE';
  const isInconclusive = !isGenuine && !isEdited;

  // Semantic color and status determination
  let themeClass = 'confidence-suspicious';
  let StatusIcon = HelpCircle;
  let statusText = 'Insufficient or ambiguous evidence';

  if (isGenuine) {
    themeClass = 'confidence-real';
    StatusIcon = CheckCircle2;
    statusText = 'Authentic Pattern Alignment (Uncalibrated)';
  } else if (isEdited) {
    themeClass = 'confidence-fake';
    StatusIcon = AlertOctagon;
    statusText = 'Tamper Pattern Alignment (Uncalibrated)';
  }

  // Resolve effective uncalibrated decision score for decisive verdicts
  const effectiveScore = decisionScore !== null && decisionScore !== undefined
    ? Number(decisionScore)
    : (confidence !== null && confidence !== undefined ? Number(confidence) : null);

  const scorePercent = effectiveScore !== null
    ? Math.round(Math.min(Math.max(effectiveScore, 0), 1) * 100)
    : null;

  return (
    <div className={`card confidence-card ${themeClass}`}>
      <div className="confidence-header">
        <div className="confidence-title-group">
          <Shield size={18} className="confidence-icon" />
          <h4 className="confidence-title">
            {isInconclusive
              ? 'Evidence Ambiguity & Decision Status'
              : 'Decision Score (Uncalibrated)'}
          </h4>
        </div>
        <span className="confidence-badge">
          <StatusIcon size={13} />
          {statusText}
        </span>
      </div>

      <div className="confidence-body">
        {isInconclusive ? (
          /* INCONCLUSIVE: Suppress percentage meter and progress bar */
          <div className="confidence-inconclusive-content">
            <div className="confidence-display-row">
              <span className="confidence-numeric" style={{ fontSize: '1.35rem', color: '#f59e0b' }}>
                Indeterminate
              </span>
              <span className="confidence-raw-value">
                Status: <code>Insufficient or ambiguous evidence</code>
                {uncertaintyScore !== null && uncertaintyScore !== undefined && (
                  <span style={{ display: 'block', marginTop: '0.35rem', fontSize: '0.8rem', color: '#94a3b8' }}>
                    Boundary Ambiguity Index: <code>{Number(uncertaintyScore).toFixed(4)}</code> (1.0 = threshold boundary)
                  </span>
                )}
                {effectiveScore !== null && (
                  <span style={{ display: 'block', marginTop: '0.2rem', fontSize: '0.8rem', color: '#94a3b8' }}>
                    Uncalibrated Decision Score: <code>{effectiveScore.toFixed(4)}</code>
                  </span>
                )}
              </span>
            </div>

            <div
              className="inconclusive-notice-box"
              style={{
                background: 'rgba(245, 158, 11, 0.08)',
                border: '1px solid rgba(245, 158, 11, 0.25)',
                borderRadius: '6px',
                padding: '0.85rem',
                marginTop: '0.85rem',
                display: 'flex',
                gap: '0.6rem',
                alignItems: 'flex-start',
              }}
            >
              <Info size={18} style={{ color: '#f59e0b', flexShrink: 0, marginTop: '2px' }} />
              <p style={{ margin: 0, fontSize: '0.85rem', lineHeight: '1.45', color: '#cbd5e1' }}>
                <strong>Indeterminate Verdict:</strong> Multi-branch perception signals fall within the decision boundary margin or exhibit inter-branch conflict. Confidence percentage and progress meters are suppressed because the evidence is insufficient to confirm or refute tampering. Manual review is required.
              </p>
            </div>

            <p className="confidence-disclaimer" style={{ marginTop: '0.85rem' }}>
              * Uncalibrated experimental assessment. Does not verify bank-side ledger settlement, account balance, or transaction execution.
            </p>
          </div>
        ) : (
          /* DECISIVE VERDICTS (GENUINE / EDITED): Display uncalibrated score with explicit disclaimers */
          <>
            <div className="confidence-display-row">
              <span className="confidence-numeric">
                {effectiveScore !== null ? effectiveScore.toFixed(4) : 'Not available'}
              </span>
              <span className="confidence-raw-value">
                Score Scale: <code>[0.0000 - 1.0000]</code>
                {scoreType && (
                  <span className="score-type-pill" style={{ marginLeft: '0.5rem', fontSize: '0.75rem', color: '#64748b' }}>
                    ({scoreType})
                  </span>
                )}
              </span>
            </div>

            {/* Visual Clamped Horizontal Progress Meter for Decisive Verdicts Only */}
            {scorePercent !== null && (
              <div
                className="confidence-meter-track"
                role="progressbar"
                aria-valuenow={scorePercent}
                aria-valuemin={0}
                aria-valuemax={100}
                aria-label="Model uncalibrated decision score meter"
              >
                <div
                  className="confidence-meter-fill"
                  style={{ width: `${scorePercent}%` }}
                />
              </div>
            )}

            <div className="confidence-scale-markers">
              <span>0.0 (Clean Pattern)</span>
              <span>0.50 Threshold</span>
              <span>1.0 (Tamper Pattern)</span>
            </div>

            <p className="confidence-disclaimer">
              * Decision score reflects uncalibrated heuristic multi-branch signals, not a statistical confidence or probability. Does not verify bank-side ledger settlement.
            </p>
          </>
        )}
      </div>
    </div>
  );
}
