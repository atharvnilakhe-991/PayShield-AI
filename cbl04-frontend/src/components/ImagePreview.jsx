import { useState } from 'react';
import { Trash2, RefreshCw, Image as ImageIcon, CheckCircle, AlertCircle } from 'lucide-react';
import { formatFileSize } from '../utils/fileValidation';

/**
 * ImagePreview component displays the currently selected screenshot,
 * its metadata, and actions to replace or remove it.
 */
export default function ImagePreview({ file, previewUrl, onRemove, onReplace, disabled }) {
  const [dimensions, setDimensions] = useState(null);
  const [loadError, setLoadError] = useState(false);

  const handleImageLoad = (e) => {
    setDimensions({
      width: e.target.naturalWidth,
      height: e.target.naturalHeight,
    });
    setLoadError(false);
  };

  const handleImageError = () => {
    setLoadError(true);
  };

  return (
    <div className="preview-container">
      <div className="preview-media-box">
        {loadError ? (
          <div className="preview-load-error">
            <AlertCircle size={32} className="text-danger" />
            <p>Failed to generate image preview. The file may be corrupt.</p>
          </div>
        ) : (
          <img
            src={previewUrl}
            alt="Uploaded GPay screenshot preview"
            className="preview-image"
            onLoad={handleImageLoad}
            onError={handleImageError}
          />
        )}
      </div>

      <div className="preview-details">
        <div className="preview-meta">
          <div className="preview-title-row">
            <ImageIcon size={18} className="text-primary" />
            <span className="preview-filename" title={file.name}>
              {file.name}
            </span>
          </div>

          <div className="preview-submeta">
            <span className="meta-pill">{formatFileSize(file.size)}</span>
            {dimensions && (
              <span className="meta-pill">
                {dimensions.width} × {dimensions.height} px
              </span>
            )}
            <span className="meta-pill meta-ready">
              <CheckCircle size={12} /> Ready for analysis
            </span>
          </div>
        </div>

        <div className="preview-actions">
          <button
            type="button"
            className="btn btn-secondary btn-sm"
            onClick={onReplace}
            disabled={disabled}
            title="Choose a different screenshot"
          >
            <RefreshCw size={14} />
            Replace
          </button>
          <button
            type="button"
            className="btn btn-outline-danger btn-sm"
            onClick={onRemove}
            disabled={disabled}
            title="Remove this screenshot"
          >
            <Trash2 size={14} />
            Remove
          </button>
        </div>
      </div>
    </div>
  );
}

