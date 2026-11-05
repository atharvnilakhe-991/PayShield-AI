import { useEffect, useRef } from 'react';
import {
  CheckCircle2,
  AlertOctagon,
  HelpCircle,
  RefreshCw,
  Sparkles,
  ArrowLeft,
  FileCheck2,
  FileText
} from 'lucide-react';
import ConfidenceScore from './ConfidenceScore';
import OCRDataCard from './OCRDataCard';
import SuspiciousRegionsCard from './SuspiciousRegionsCard';
import HeatmapView from './HeatmapView';

/**
 * ResultsDashboard: CBL-04 Live Google Pay Tamper Detection Dashboard
 * Truthfully displays real model verdict, uncalibrated decision score,
 * live OCR metadata, multi-branch evidence breakdown, and spatial safeguards.
 */
export default function ResultsDashboard({
  result,
  imagePreviewUrl,
  selectedFile,
  onReset,
  onChangeScreenshot,
  isLiveMode = true,
}) {
  const dashboardRef = useRef(null);

  // Smooth scroll into view when results dashboard mounts
  useEffect(() => {
    if (dashboardRef.current) {
      dashboardRef.current.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  }, []);

  if (!result) return null;

  const {
    verdict,
    confidence,
    decision_score,
    uncertainty_score,
    message,
    request_id,
    ocr_data,
    manipulation_analysis,
    branch_scores,
    cnn_score,
    semantic_score,
    stacker_score,
    reason_codes,
    warnings,
    pipeline_status,
    processing_time_ms,
    component_status,
    disclaimer,
    score_type,
    uploadedFileName,
  } = result;

  // Truly live if pipeline_status starts with phase/live or if in live mode
  const isLive = Boolean(
    isLiveMode ||
    (pipeline_status && (pipeline_status.startsWith('phase') || pipeline_status.startsWith('live')))
  );

  // Determine semantic color theme based on canonical or legacy verdict
  const normVerdict = (verdict || '').toUpperCase();
  const isGenuine = normVerdict === 'GENUINE' || normVerdict === 'REAL';
  const isEdited = normVerdict === 'EDITED' || normVerdict === 'FAKE';
  const isInconclusive = normVerdict === 'INCONCLUSIVE' || (!isGenuine && !isEdited);

  const effectiveDecisionScore = decision_score ?? (confidence !== null && confidence !== undefined ? confidence : null);

  let statusTheme = 'dashboard-suspicious';
  let VerdictIcon = HelpCircle;
  let verdictTitle = 'Inconclusive / Indeterminate Receipt (INCONCLUSIVE)';
  let verdictDescription = 'Analysis indeterminate. Text legibility is insufficient or multi-branch perception signals fall within decision boundary margin.';

  if (isGenuine) {
    statusTheme = 'dashboard-real';
    VerdictIcon = CheckCircle2;
    verdictTitle = 'Authentic Receipt Verified (GENUINE)';
    verdictDescription = 'Visual layout, fonts, and UPI alignments match authentic Google Pay dark-theme receipt patterns.';
  } else if (isEdited) {
    statusTheme = 'dashboard-fake';
    VerdictIcon = AlertOctagon;
    verdictTitle = 'Manipulated Receipt Detected (EDITED)';
    verdictDescription = 'Visual or semantic evidence indicates digital tampering, font inconsistency, or alignment anomaly.';
  }

  const effectiveCnnScore = branch_scores?.cnn_score ?? cnn_score;
  const effectiveSemanticScore = branch_scores?.semantic_score ?? semantic_score;
  const effectiveStackerScore = branch_scores?.stacker_score ?? stacker_score;
  const hasBranchScores = effectiveCnnScore !== null && effectiveCnnScore !== undefined;

  const displayFileName = selectedFile?.name || uploadedFileName || 'screenshot.jpg';

  return (
    <div ref={dashboardRef} className={`results-dashboard ${statusTheme}`}>
      {/* Top Action & Navigation Bar */}
      <div className="dashboard-action-bar">
        <button
          type="button"
          className="btn btn-secondary btn-sm"
          onClick={onChangeScreenshot || onReset}
          title="Upload or select a different image"
        >
          <ArrowLeft size={15} />
          <span>Change Screenshot</span>
        </button>

        <div className="dashboard-status-pill">
          <FileCheck2 size={15} className="text-primary" />
          <span>
            {isLive
              ? `Live Backend Analysis (${pipeline_status || 'verified'})`
              : 'Analysis Complete (Simulated)'}
          </span>
        </div>

        <button
          type="button"
          className="btn btn-primary btn-sm"
          onClick={onReset}
          title="Start fresh analysis workflow"
        >
          <RefreshCw size={15} />
          <span>Analyze Another Screenshot</span>
        </button>
      </div>

      {/* Verdict Summary Card */}
      <section className="card verdict-summary-card">
        <div className="summary-card-header">
          <div className="summary-badge">
            <Sparkles size={14} />
            <span>CBL-04 Automated Fraud Analysis</span>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '0.8rem', flexWrap: 'wrap' }}>
            <div className="target-filename-badge" style={{ display: 'flex', alignItems: 'center', gap: '0.35rem', fontSize: '0.8rem', color: '#94a3b8' }}>
              <FileText size={14} />
              <span>Target:</span>
              <code style={{ color: '#e2e8f0' }}>{displayFileName}</code>
            </div>

            {request_id && (
              <div className="request-id-badge">
                <span className="id-label">Trace ID:</span>
                <code>{request_id}</code>
              </div>
            )}
          </div>
        </div>

        <div className="verdict-banner-content">
          <div className="verdict-icon-container">
            <VerdictIcon size={44} className="verdict-main-icon" />
          </div>

          <div className="verdict-text-block">
            <div className="verdict-headline-row">
              <h2 className="verdict-headline">{verdictTitle}</h2>
              <span className={`verdict-pill pill-${(verdict || '').toLowerCase()}`}>
                {verdict || 'INCONCLUSIVE'}
              </span>
            </div>
            <p className="verdict-subtext">{verdictDescription}</p>
            {message && (
              <p className="verdict-server-message">
                <strong>Engine Note:</strong> {message}
              </p>
            )}
          </div>

          <div className="verdict-stat-tile">
            {isInconclusive ? (
              <>
                <span className="stat-label">Assessment Status</span>
                <span className="stat-value" style={{ fontSize: '1.25rem', color: '#f59e0b' }}>
                  Indeterminate
                </span>
                <span className="stat-footnote">
                  {uncertainty_score !== null && uncertainty_score !== undefined
                    ? `Ambiguity Index: ${Number(uncertainty_score).toFixed(4)}`
                    : 'Ambiguous / Low Signal'}
                </span>
              </>
            ) : (
              <>
                <span className="stat-label">Decision Score (Uncalibrated)</span>
                <span className="stat-value">
                  {effectiveDecisionScore !== null && effectiveDecisionScore !== undefined
                    ? Number(effectiveDecisionScore).toFixed(4)
                    : 'Not available'}
                </span>
                <span className="stat-footnote">
                  {isLive ? 'Uncalibrated Model Score' : 'Decision Margin'}
                </span>
              </>
            )}
          </div>
        </div>

        <div className="verdict-card-footer">
          <p className="simulated-disclaimer">
            * <strong>Notice:</strong>{' '}
            {disclaimer ||
              'Screenshot analysis performs digital artifact and layout integrity checks only. It does not verify bank-side settlement, banking ledger entries, or prove that a financial payment occurred.'}
          </p>
        </div>
      </section>

      {/* Confidence / Decision Score Visualization */}
      <section className="dashboard-section">
        <ConfidenceScore
          confidence={confidence}
          decisionScore={decision_score}
          uncertaintyScore={uncertainty_score}
          verdict={verdict}
          scoreType={score_type}
          pipelineStatus={pipeline_status}
          isLive={isLive}
        />
      </section>

      {/* Multi-Branch AI Breakdown & Provenance */}
      {hasBranchScores && (
        <section className="dashboard-section">
          <div className="card branch-scores-card">
            <div className="card-header">
              <div className="card-header-left">
                <Sparkles className="card-icon text-primary" size={20} />
                <h3>Multi-Branch AI Evidence Breakdown</h3>
              </div>
              {processing_time_ms !== undefined && (
                <span className="badge badge-accent">
                  Latency: {processing_time_ms} ms
                </span>
              )}
            </div>

            <div className="card-body">
              <p className="text-muted" style={{ fontSize: '0.85rem' }}>
                Independent perception branches evaluated prior to final ensemble decision. Scores represent uncalibrated anomaly indicators:
              </p>

              <div className="branch-scores-grid">
                <div className="branch-score-tile">
                  <div className="branch-score-header">
                    <span className="branch-name">CNN Visual Branch</span>
                    <span className="badge badge-sm badge-primary">ResNet-18</span>
                  </div>
                  <span className="branch-value">
                    {effectiveCnnScore !== undefined && effectiveCnnScore !== null
                      ? Number(effectiveCnnScore).toFixed(4)
                      : 'Not available'}
                  </span>
                  <span className="branch-note">
                    Visual anomaly score: {effectiveCnnScore !== null && effectiveCnnScore !== undefined ? Number(effectiveCnnScore).toFixed(4) : 'Not available'}
                  </span>
                </div>

                <div className="branch-score-tile">
                  <div className="branch-score-header">
                    <span className="branch-name">Semantic Consistency</span>
                    <span className="badge badge-sm badge-secondary">Phase 6 Rules</span>
                  </div>
                  <span className="branch-value">
                    {effectiveSemanticScore !== undefined && effectiveSemanticScore !== null
                      ? Number(effectiveSemanticScore).toFixed(4)
                      : 'Not available'}
                  </span>
                  <span className="branch-note">
                    Format & layout score: {effectiveSemanticScore !== null && effectiveSemanticScore !== undefined ? Number(effectiveSemanticScore).toFixed(4) : 'Not available'}
                  </span>
                </div>

                <div className="branch-score-tile">
                  <div className="branch-score-header">
                    <span className="branch-name">OOF Stacker Meta-Learner</span>
                    <span className="badge badge-sm badge-accent">L2 Logistic</span>
                  </div>
                  <span className="branch-value">
                    {effectiveStackerScore !== undefined && effectiveStackerScore !== null
                      ? Number(effectiveStackerScore).toFixed(4)
                      : 'Not available'}
                  </span>
                  <span className="branch-note">
                    Ensemble decision score: {effectiveStackerScore !== null && effectiveStackerScore !== undefined ? Number(effectiveStackerScore).toFixed(4) : 'Not available'}
                  </span>
                </div>
              </div>

              {/* Machine-Readable Reason Codes */}
              {Array.isArray(reason_codes) && reason_codes.length > 0 && (
                <div className="reason-codes-section">
                  <span className="reason-codes-label">Detected Anomaly / Integrity Codes:</span>
                  <div className="reason-codes-wrap">
                    {reason_codes.map((code, idx) => (
                      <span key={idx} className="reason-code-tag">
                        {code}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {/* Backend Operational Warnings */}
              {Array.isArray(warnings) && warnings.length > 0 && (
                <div className="warnings-box">
                  <div className="warnings-box-title">
                    <AlertOctagon size={15} />
                    <span>Pipeline Observations &amp; Guardrail Triggers ({warnings.length}):</span>
                  </div>
                  <ul className="warnings-list">
                    {warnings.map((w, idx) => (
                      <li key={idx}>{w}</li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Component Operational Status Breakdown */}
              {component_status && typeof component_status === 'object' && (
                <div style={{ marginTop: '1rem', display: 'flex', flexWrap: 'wrap', gap: '0.4rem', alignItems: 'center' }}>
                  <span style={{ fontSize: '0.75rem', fontWeight: 700, color: '#64748b' }}>Branch Engines:</span>
                  {Object.entries(component_status).map(([comp, stat]) => (
                    <span
                      key={comp}
                      className={`badge badge-sm ${stat === 'live' ? 'badge-primary' : 'badge-accent'}`}
                      style={{ textTransform: 'uppercase', fontSize: '0.7rem' }}
                    >
                      {comp}: {stat}
                    </span>
                  ))}
                </div>
              )}
            </div>
          </div>
        </section>
      )}

      {/* Side-by-Side Uploaded Screenshot vs Heatmap Comparison */}
      <section className="dashboard-section">
        <HeatmapView
          originalImageUrl={imagePreviewUrl}
          heatmapUrl={manipulation_analysis?.heatmap_url}
        />
      </section>

      {/* Two-Column Grid for OCR Details and Suspicious Regions */}
      <div className="dashboard-2col-grid">
        {/* Suspicious Regions Analysis */}
        <section className="grid-column">
          <SuspiciousRegionsCard
            regions={manipulation_analysis?.suspicious_regions}
            manipulationDetected={manipulation_analysis?.detected}
            isLive={isLive}
          />
        </section>

        {/* OCR Information Card */}
        <section className="grid-column">
          <OCRDataCard ocrData={ocr_data} isLive={isLive} />
        </section>
      </div>

      {/* Bottom Sticky Reset Bar */}
      <div className="dashboard-bottom-bar">
        <div className="bottom-meta-text">
          <span>Target File: <strong>{displayFileName}</strong></span>
          <span className="meta-divider">•</span>
          <span>
            Status:{' '}
            <strong>
              {isLive ? `Live Backend Analysis (${pipeline_status || 'active'})` : 'Offline Demonstration'}
            </strong>
          </span>
        </div>

        <div className="bottom-buttons">
          <button
            type="button"
            className="btn btn-secondary"
            onClick={onChangeScreenshot || onReset}
          >
            <ArrowLeft size={16} />
            Back to Upload
          </button>
          <button
            type="button"
            className="btn btn-primary"
            onClick={onReset}
          >
            <RefreshCw size={16} />
            Analyze Another Screenshot
          </button>
        </div>
      </div>
    </div>
  );
}
