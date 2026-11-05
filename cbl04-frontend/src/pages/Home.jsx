import { useState, useEffect } from 'react';
import { ShieldCheck, Play, AlertTriangle, Sparkles, CheckCircle2, Server, RotateCcw } from 'lucide-react';
import UploadCard from '../components/UploadCard';
import LoadingState from '../components/LoadingState';
import ResultsDashboard from '../components/ResultsDashboard';
import { validateScreenshotFile } from '../utils/fileValidation';
import { analyzeImage } from '../services/api';
import { analyzeImageMock } from '../services/mockApi';
import { VERDICTS } from '../services/apiContract';

/**
 * Home Page: CBL-04 Live Google Pay Receipt Manipulation Detection
 * Connects directly to the FastAPI backend with real OCR, CNN visual inference,
 * and semantic consistency checks.
 */
export default function Home() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [validationError, setValidationError] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisProgress, setAnalysisProgress] = useState(0);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [apiError, setApiError] = useState(null);
  const [completionNotification, setCompletionNotification] = useState(null);

  // Live FastAPI Backend is the default and standard operational mode
  const [apiMode, setApiMode] = useState('LIVE'); // 'LIVE' | 'MOCK'
  const [mockVerdictMode, setMockVerdictMode] = useState('AUTO');

  // Clean up object URLs on file change or unmount
  useEffect(() => {
    return () => {
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
      }
    };
  }, [previewUrl]);

  const handleFileSelect = (file) => {
    // Clear previous results & errors
    setValidationError(null);
    setApiError(null);
    setAnalysisResult(null);
    setCompletionNotification(null);

    // Validate the selected file
    const validation = validateScreenshotFile(file);
    if (!validation.isValid) {
      setValidationError(validation.error);
      return;
    }

    // Clean up previous preview URL if any
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }

    // Generate new object URL for instant client-side preview
    const objectUrl = URL.createObjectURL(file);
    setSelectedFile(file);
    setPreviewUrl(objectUrl);
  };

  const handleFileRemove = () => {
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
    setSelectedFile(null);
    setPreviewUrl(null);
    setValidationError(null);
    setApiError(null);
    setAnalysisResult(null);
    setCompletionNotification(null);
  };

  const handleAnalyzeClick = async () => {
    if (!selectedFile || isAnalyzing) return;

    setIsAnalyzing(true);
    setAnalysisProgress(15);
    setApiError(null);
    setAnalysisResult(null);
    setCompletionNotification(null);

    if (apiMode === 'LIVE') {
      try {
        setAnalysisProgress(30);
        const result = await analyzeImage(selectedFile, (progressEvent) => {
          if (progressEvent && progressEvent.total) {
            const uploadPct = Math.round((progressEvent.loaded * 40) / progressEvent.total);
            setAnalysisProgress(20 + uploadPct);
          }
        });

        setAnalysisProgress(90);
        // Attach uploaded file name to result for truthful provenance
        result.uploadedFileName = selectedFile.name;
        setAnalysisResult(result);
        setCompletionNotification(
          `Live backend analysis completed via FastAPI (${result.pipeline_status || 'pipeline'}).`
        );
      } catch (err) {
        // Strict guardrail: Do NOT silently fallback to mock predictions; surface truthful error
        setApiError(err);
        setAnalysisResult(null);
      } finally {
        setIsAnalyzing(false);
        setAnalysisProgress(0);
      }
    } else {
      // Explicit Mock Mode: Isolated developer-only setting
      try {
        const forced = mockVerdictMode === 'AUTO' ? null : mockVerdictMode;
        const result = await analyzeImageMock(selectedFile, {
          forcedVerdict: forced,
          onProgress: (pct) => setAnalysisProgress(pct),
        });

        result.uploadedFileName = selectedFile.name;
        setAnalysisResult(result);
        setCompletionNotification('Simulated analysis completed. Results dashboard generated.');
      } catch (err) {
        setApiError({
          message: err.message || 'An error occurred during mock analysis.',
          status: null,
          detail: null,
          isNetworkError: false,
        });
      } finally {
        setIsAnalyzing(false);
        setAnalysisProgress(0);
      }
    }
  };

  const handleReset = () => {
    handleFileRemove();
  };

  return (
    <div className="home-page-container">
      {/* Page Header */}
      <section className="page-header-section">
        <div className="page-header-badge">
          <ShieldCheck size={14} className="text-primary" />
          <span>CBL-04 Fraud Prevention</span>
        </div>
        <h2 className="page-main-heading">GPay Receipt Verification</h2>
        <p className="page-subheading">
          Upload a Google Pay dark-theme transaction screenshot to verify receipt authenticity and detect digital tampering using live dual-branch CNN and OCR inference.
        </p>
      </section>

      {/* Upload & Controls Section */}
      <div className="upload-section-wrapper">
        <UploadCard
          selectedFile={selectedFile}
          previewUrl={previewUrl}
          validationError={validationError}
          onFileSelect={handleFileSelect}
          onFileRemove={handleFileRemove}
          isAnalyzing={isAnalyzing}
          onClearError={() => setValidationError(null)}
        />

        {/* Action Controls Bar - Live Backend Default */}
        <div className="action-control-bar">
          <div className="live-status-bar" style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
            <span className="selector-label" style={{ display: 'flex', alignItems: 'center', gap: '0.35rem', fontWeight: 600 }}>
              <Server size={15} className="text-primary" />
              Backend Pipeline:
            </span>
            <span className="badge badge-primary" style={{ padding: '0.35rem 0.75rem', fontSize: '0.8rem', fontWeight: 600 }}>
              Live FastAPI Backend
            </span>
          </div>

          <div className="action-button-group">
            <button
              type="button"
              className="btn btn-primary btn-analyze"
              onClick={handleAnalyzeClick}
              disabled={!selectedFile || isAnalyzing}
              aria-label="Analyze payment screenshot"
            >
              <Play size={16} fill="currentColor" />
              {isAnalyzing ? 'Analyzing Screenshot with AI Models...' : 'Analyze Screenshot'}
            </button>
          </div>
        </div>

        {/* Developer-Only Setting (Isolated and collapsed by default, never enabled in normal workflow) */}
        <details className="dev-mock-details" style={{ marginTop: '0.75rem', padding: '0.5rem 0.8rem', border: '1px solid #334155', borderRadius: '6px', background: '#0f172a' }}>
          <summary style={{ cursor: 'pointer', color: '#64748b', fontSize: '0.75rem', fontWeight: 600 }}>
            Developer Options (Offline Mock Simulation - Testing Only)
          </summary>
          <div style={{ marginTop: '0.6rem', display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: '0.8rem' }}>
            <div className="toggle-group toggle-group-sm">
              <button
                type="button"
                className={`btn-tab ${apiMode === 'LIVE' ? 'active' : ''}`}
                onClick={() => setApiMode('LIVE')}
              >
                Live FastAPI Backend
              </button>
              <button
                type="button"
                className={`btn-tab ${apiMode === 'MOCK' ? 'active' : ''}`}
                onClick={() => setApiMode('MOCK')}
              >
                Mock Simulation (Demo)
              </button>
            </div>

            {apiMode === 'MOCK' && (
              <div className="toggle-group toggle-group-sm">
                <button
                  type="button"
                  className={`btn-tab ${mockVerdictMode === 'AUTO' ? 'active' : ''}`}
                  onClick={() => setMockVerdictMode('AUTO')}
                >
                  Auto-Cycle
                </button>
                <button
                  type="button"
                  className={`btn-tab ${mockVerdictMode === VERDICTS.FAKE ? 'active' : ''}`}
                  onClick={() => setMockVerdictMode(VERDICTS.FAKE)}
                >
                  Force Fake
                </button>
                <button
                  type="button"
                  className={`btn-tab ${mockVerdictMode === VERDICTS.REAL ? 'active' : ''}`}
                  onClick={() => setMockVerdictMode(VERDICTS.REAL)}
                >
                  Force Real
                </button>
                <button
                  type="button"
                  className={`btn-tab ${mockVerdictMode === VERDICTS.SUSPICIOUS ? 'active' : ''}`}
                  onClick={() => setMockVerdictMode(VERDICTS.SUSPICIOUS)}
                >
                  Force Suspicious
                </button>
              </div>
            )}
          </div>
        </details>

        {/* Backend API Error Banner */}
        {apiError && (
          <div className="status-message status-error" role="alert" style={{ flexDirection: 'column', alignItems: 'flex-start', gap: '0.5rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', width: '100%' }}>
              <AlertTriangle size={18} />
              <div className="error-text-container" style={{ flex: 1 }}>
                <strong>
                  {apiError.status ? `Backend Error (HTTP ${apiError.status}):` : 'Connection Error:'}
                </strong>{' '}
                {apiError.message}
              </div>
              <button
                type="button"
                className="btn-dismiss"
                onClick={() => setApiError(null)}
              >
                ×
              </button>
            </div>

            {apiError.isNetworkError && (
              <p style={{ margin: '0 0 0 1.6rem', fontSize: '0.8rem', color: '#991b1b' }}>
                Tip: Ensure the FastAPI service is running via <code>python -m uvicorn app.main:app --port 8000</code>.
              </p>
            )}

            <div style={{ marginLeft: '1.6rem' }}>
              <button
                type="button"
                className="btn btn-sm btn-secondary"
                onClick={handleAnalyzeClick}
                disabled={isAnalyzing}
              >
                <RotateCcw size={13} />
                <span>Retry Request</span>
              </button>
            </div>
          </div>
        )}

        {/* Loading State */}
        {isAnalyzing && <LoadingState progress={analysisProgress} />}

        {/* Completion Status Notification */}
        {completionNotification && !isAnalyzing && (
          <div className="status-message status-success status-toast" role="status">
            <CheckCircle2 size={16} />
            <span>{completionNotification}</span>
          </div>
        )}

        {/* Detailed Results Dashboard */}
        {analysisResult && !isAnalyzing && (
          <ResultsDashboard
            result={analysisResult}
            imagePreviewUrl={previewUrl}
            selectedFile={selectedFile}
            onReset={handleReset}
            onChangeScreenshot={handleFileRemove}
            isLiveMode={apiMode === 'LIVE'}
          />
        )}
      </div>
    </div>
  );
}
