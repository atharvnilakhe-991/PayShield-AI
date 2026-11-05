import { useState } from 'react';
import Header from './components/Header';
import Home from './pages/Home';
import { API_BASE_URL, checkBackendHealth } from './services/api';
import {
  PROPOSED_ENDPOINTS,
  IMAGE_CONSTRAINTS,
  VERDICTS,
  MOCK_API_RESPONSES
} from './services/apiContract';
import {
  Server,
  FileCode2,
  CheckCircle2,
  AlertTriangle,
  Layers
} from 'lucide-react';
import './App.css';

function App() {
  const [activeTab, setActiveTab] = useState('upload'); // 'upload' | 'contract'
  const [healthStatus, setHealthStatus] = useState(null);
  const [isCheckingHealth, setIsCheckingHealth] = useState(false);
  const [selectedMock, setSelectedMock] = useState('FAKE_SAMPLE');

  const handleTestConnection = async () => {
    setIsCheckingHealth(true);
    setHealthStatus(null);
    try {
      const data = await checkBackendHealth();
      setHealthStatus({
        success: true,
        message: 'Backend connection successful!',
        data,
      });
    } catch (err) {
      setHealthStatus({
        success: false,
        message: err.message,
        isNetworkError: err.isNetworkError,
      });
    } finally {
      setIsCheckingHealth(false);
    }
  };

  return (
    <div className="app-container">
      <Header activeTab={activeTab} onTabChange={setActiveTab} />

      <main className="main-content">
        {activeTab === 'upload' ? (
          /* ====================================================================
             PHASE 2: GPay Screenshot Upload & Mock Verification Workspace
             ==================================================================== */
          <Home />
        ) : (
          /* ====================================================================
             PHASE 1: API Contract Specification & FastAPI Backend Health Tester
             ==================================================================== */
          <div className="contract-view-wrapper">
            <section className="card phase-banner">
              <div className="banner-content">
                <div className="badge badge-warning">
                  <AlertTriangle size={14} />
                  Phase 1 Deliverable: Project Setup & API Contract
                </div>
                <h2>Frontend Skeleton & API Contract Ready</h2>
                <p className="text-muted">
                  Codified communication contract for Member 3 (FastAPI Backend Lead).
                  Use this screen to test backend connectivity and inspect the JSON schema.
                </p>
              </div>
            </section>

            <div className="grid-2col">
              {/* Card 1: Backend Connection & Environment */}
              <section className="card">
                <div className="card-header">
                  <Server className="card-icon text-primary" size={20} />
                  <h3>Backend API Integration Setup</h3>
                </div>

                <div className="card-body">
                  <div className="info-row">
                    <span className="label">Vite Env Base URL:</span>
                    <code className="code-badge">{API_BASE_URL}</code>
                  </div>
                  <div className="info-row">
                    <span className="label">Analysis Endpoint:</span>
                    <code className="code-badge">{PROPOSED_ENDPOINTS.ANALYZE}</code>
                  </div>
                  <div className="info-row">
                    <span className="label">Payload Type:</span>
                    <span className="text-emphasis">multipart/form-data (key: 'image')</span>
                  </div>
                  <div className="info-row">
                    <span className="label">Supported Formats:</span>
                    <span className="text-muted">{IMAGE_CONSTRAINTS.SUPPORTED_EXTENSIONS.join(', ')}</span>
                  </div>
                  <div className="info-row">
                    <span className="label">Max Upload Limit:</span>
                    <span className="text-muted">{IMAGE_CONSTRAINTS.MAX_FILE_SIZE_LABEL}</span>
                  </div>

                  <div className="health-test-box">
                    <button
                      type="button"
                      className="btn btn-primary"
                      onClick={handleTestConnection}
                      disabled={isCheckingHealth}
                    >
                      <Server size={16} />
                      {isCheckingHealth ? 'Pinging Backend...' : 'Test Backend Connection'}
                    </button>

                    {healthStatus && (
                      <div
                        className={`status-message ${
                          healthStatus.success ? 'status-success' : 'status-error'
                        }`}
                      >
                        {healthStatus.success ? (
                          <CheckCircle2 size={16} />
                        ) : (
                          <AlertTriangle size={16} />
                        )}
                        <span>{healthStatus.message}</span>
                      </div>
                    )}
                    <p className="hint-text">
                      * Note: Network errors are expected until Member 3 launches the FastAPI server.
                    </p>
                  </div>
                </div>
              </section>

              {/* Card 2: Proposed API Contract Specification */}
              <section className="card">
                <div className="card-header">
                  <FileCode2 className="card-icon text-accent" size={20} />
                  <h3>Confirmed Contract: POST {PROPOSED_ENDPOINTS.ANALYZE}</h3>
                </div>

                <div className="card-body">
                  <div className="contract-status-pill">
                    <span className="pill-dot"></span>
                    <span>STATUS: Confirmed Contract Specification</span>
                  </div>

                  <div className="verdicts-list">
                    <span className="label">Agreed Verdict Enums:</span>
                    <div className="tags-row">
                      <span className="tag tag-success">{VERDICTS.REAL}</span>
                      <span className="tag tag-danger">{VERDICTS.FAKE}</span>
                      <span className="tag tag-warning">{VERDICTS.SUSPICIOUS}</span>
                    </div>
                  </div>

                  <div className="mock-selector-header">
                    <span className="label">Inspect Schema Fixtures:</span>
                    <div className="toggle-group">
                      <button
                        type="button"
                        className={`btn-tab ${selectedMock === 'FAKE_SAMPLE' ? 'active' : ''}`}
                        onClick={() => setSelectedMock('FAKE_SAMPLE')}
                      >
                        Fake Sample
                      </button>
                      <button
                        type="button"
                        className={`btn-tab ${selectedMock === 'REAL_SAMPLE' ? 'active' : ''}`}
                        onClick={() => setSelectedMock('REAL_SAMPLE')}
                      >
                        Real Sample
                      </button>
                      <button
                        type="button"
                        className={`btn-tab ${selectedMock === 'SUSPICIOUS_SAMPLE' ? 'active' : ''}`}
                        onClick={() => setSelectedMock('SUSPICIOUS_SAMPLE')}
                      >
                        Suspicious
                      </button>
                    </div>
                  </div>

                  <pre className="json-preview">
                    {JSON.stringify(MOCK_API_RESPONSES[selectedMock], null, 2)}
                  </pre>
                </div>
              </section>
            </div>

            {/* Hackathon Roadmap Card */}
            <section className="card roadmap-card">
              <div className="card-header">
                <Layers className="card-icon text-secondary" size={20} />
                <h3>Project Deliverables & Hackathon Roadmap</h3>
              </div>
              <div className="card-body roadmap-grid">
                <div className="roadmap-step completed">
                  <div className="step-badge">H0–H1 (Completed)</div>
                  <h4>Phase 1: Setup & API Contract</h4>
                  <ul>
                    <li>✓ React + Vite skeleton initialized</li>
                    <li>✓ Axios API service configured</li>
                    <li>✓ API contract documented & confirmed</li>
                    <li>✓ Light design system established</li>
                  </ul>
                </div>

                <div className="roadmap-step completed">
                  <div className="step-badge">H1–H3 (Completed)</div>
                  <h4>Phase 2: Upload Page & Mock API</h4>
                  <ul>
                    <li>✓ Drag-and-drop receipt uploader</li>
                    <li>✓ Client-side MIME & size validation</li>
                    <li>✓ Instant image preview with revocation</li>
                    <li>✓ 2–3s simulated inference flow</li>
                    <li>✓ Auto-cycle & force verdict controls</li>
                  </ul>
                </div>

                <div className="roadmap-step completed">
                  <div className="step-badge">H3–H5 (Current)</div>
                  <h4>Phase 3: Results Dashboard & Heatmap</h4>
                  <ul>
                    <li>✓ Verdict summary & confidence meter</li>
                    <li>✓ Side-by-side original vs heatmap inspection</li>
                    <li>✓ Suspicious regions coordinate breakdown</li>
                    <li>✓ Simulated OCR data extraction card</li>
                    <li>✓ Empty-state heatmap placeholder</li>
                  </ul>
                </div>

                <div className="roadmap-step upcoming">
                  <div className="step-badge">H9–H12</div>
                  <h4>Phase 4: Full Backend Integration</h4>
                  <p className="text-muted">
                    Live FastAPI backend integration, real CNN Grad-CAM overlay, and final hackathon presentation prep.
                  </p>
                </div>
              </div>
            </section>
          </div>
        )}
      </main>

      <footer className="app-footer">
        <p>CBL-04 Hackathon • Member 4 (Frontend Integration) &amp; Member 3 (FastAPI/ML)</p>
      </footer>
    </div>
  );
}

export default App;
