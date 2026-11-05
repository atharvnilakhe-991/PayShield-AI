import { ShieldCheck, Upload, FileCode2 } from 'lucide-react';
import { APP_INFO } from '../utils/constants';

export default function Header({ activeTab, onTabChange }) {
  return (
    <header className="app-header">
      <div className="header-brand">
        <div className="brand-icon">
          <ShieldCheck size={28} className="text-primary" />
        </div>
        <div>
          <h1 className="brand-title">{APP_INFO.TITLE}</h1>
          <p className="brand-subtitle">{APP_INFO.SUBTITLE}</p>
        </div>
      </div>

      <div className="header-nav-group">
        <nav className="header-tabs" aria-label="Main application navigation">
          <button
            type="button"
            className={`nav-tab-btn ${activeTab === 'upload' ? 'active' : ''}`}
            onClick={() => onTabChange('upload')}
          >
            <Upload size={15} />
            <span>Upload & Verify</span>
            <span className="tab-pill-badge">Phase 2</span>
          </button>
          <button
            type="button"
            className={`nav-tab-btn ${activeTab === 'contract' ? 'active' : ''}`}
            onClick={() => onTabChange('contract')}
          >
            <FileCode2 size={15} />
            <span>API Contract & Health</span>
            <span className="tab-pill-badge">Phase 1</span>
          </button>
        </nav>
      </div>
    </header>
  );
}
