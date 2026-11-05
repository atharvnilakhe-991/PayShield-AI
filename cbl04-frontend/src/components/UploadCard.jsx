import { useState, useRef } from 'react';
import { UploadCloud, FileImage, AlertCircle } from 'lucide-react';
import { IMAGE_CONSTRAINTS } from '../services/apiContract';
import ImagePreview from './ImagePreview';

/**
 * UploadCard manages the drag-and-drop dropzone, file input,
 * inline validation errors, and delegates preview rendering.
 */
export default function UploadCard({
  selectedFile,
  previewUrl,
  validationError,
  onFileSelect,
  onFileRemove,
  isAnalyzing,
  onClearError,
}) {
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef(null);

  const handleDragEnter = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (!isAnalyzing) {
      setIsDragging(true);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    if (isAnalyzing) return;

    if (e.dataTransfer?.files && e.dataTransfer.files.length > 0) {
      const droppedFile = e.dataTransfer.files[0];
      onFileSelect(droppedFile);
    }
  };

  const handleBrowseClick = () => {
    if (fileInputRef.current && !isAnalyzing) {
      fileInputRef.current.value = '';
      fileInputRef.current.click();
    }
  };

  const handleFileInputChange = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      const chosenFile = e.target.files[0];
      onFileSelect(chosenFile);
    }
  };

  const handleKeyDown = (e) => {
    if ((e.key === 'Enter' || e.key === ' ') && !selectedFile && !isAnalyzing) {
      e.preventDefault();
      handleBrowseClick();
    }
  };

  return (
    <div className="card upload-card">
      <div className="card-header">
        <FileImage className="card-icon text-primary" size={20} />
        <h3>Upload Payment Screenshot</h3>
      </div>

      <div className="card-body">
        {/* Hidden File Input */}
        <input
          ref={fileInputRef}
          type="file"
          id="screenshot-file-input"
          accept={IMAGE_CONSTRAINTS.SUPPORTED_EXTENSIONS.join(',')}
          onChange={handleFileInputChange}
          style={{ display: 'none' }}
          disabled={isAnalyzing}
          aria-label="Upload payment screenshot file"
        />

        {/* Selected File Preview or Drop Area */}
        {selectedFile && previewUrl ? (
          <ImagePreview
            file={selectedFile}
            previewUrl={previewUrl}
            onRemove={onFileRemove}
            onReplace={handleBrowseClick}
            disabled={isAnalyzing}
          />
        ) : (
          <div
            className={`dropzone ${isDragging ? 'dropzone-active' : ''}`}
            onDragEnter={handleDragEnter}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            onClick={handleBrowseClick}
            onKeyDown={handleKeyDown}
            tabIndex={0}
            role="button"
            aria-label="Drop screenshot image here or press Enter to browse"
          >
            <div className="dropzone-icon-wrapper">
              <UploadCloud size={40} className="dropzone-icon" />
            </div>

            <h4 className="dropzone-title">Upload Payment Screenshot</h4>
            <p className="dropzone-description">
              Drag and drop your screenshot here, or{' '}
              <span className="browse-link">browse files</span> from your computer.
            </p>

            <div className="dropzone-specs">
              <span>Supported: JPG, JPEG, PNG, WebP</span>
              <span className="spec-dot">•</span>
              <span>Max file size: {IMAGE_CONSTRAINTS.MAX_FILE_SIZE_LABEL}</span>
            </div>

            <button
              type="button"
              className="btn btn-secondary btn-browse"
              onClick={(e) => {
                e.stopPropagation();
                handleBrowseClick();
              }}
              disabled={isAnalyzing}
            >
              Browse Files
            </button>
          </div>
        )}

        {/* Inline Validation Error Notification */}
        {validationError && (
          <div className="status-message status-error validation-error-box" role="alert">
            <AlertCircle size={18} className="error-icon" />
            <div className="error-text-container">
              <strong>Invalid File:</strong> {validationError}
            </div>
            {onClearError && (
              <button
                type="button"
                className="btn-dismiss"
                onClick={onClearError}
                aria-label="Dismiss error"
              >
                ×
              </button>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

