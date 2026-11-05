import { Loader2, Cpu } from 'lucide-react';

/**
 * LoadingState displays a clean, animated progress view during mock analysis.
 */
export default function LoadingState({ progress = 0 }) {
  return (
    <div className="loading-card" role="status" aria-live="polite">
      <div className="loading-spinner-wrapper">
        <Loader2 size={36} className="spinner-icon text-primary animate-spin" />
      </div>

      <div className="loading-content">
        <h4 className="loading-title">Analyzing screenshot...</h4>
        <p className="loading-subtitle">
          Running simulated analysis. ML model integration will be added later.
        </p>

        {/* Simulated Progress Bar */}
        <div className="progress-container">
          <div
            className="progress-bar-fill"
            style={{ width: `${Math.min(progress, 100)}%` }}
            aria-valuenow={progress}
            aria-valuemin={0}
            aria-valuemax={100}
          />
        </div>

        <div className="progress-meta">
          <span className="progress-step-text">
            <Cpu size={13} className="text-primary" />
            {progress < 40
              ? 'Preprocessing image & normalizing...'
              : progress < 80
              ? 'Running OCR & font consistency checks...'
              : 'Synthesizing manipulation score...'}
          </span>
          <span className="progress-pct">{progress}%</span>
        </div>
      </div>
    </div>
  );
}

