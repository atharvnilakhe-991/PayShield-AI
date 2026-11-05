import { IMAGE_CONSTRAINTS } from '../services/apiContract';

/**
 * Formats byte count into a readable string (e.g. "2.4 MB", "450 KB").
 * @param {number} bytes
 * @returns {string}
 */
export function formatFileSize(bytes) {
  if (bytes === 0) return '0 Bytes';
  if (!bytes || isNaN(bytes)) return 'Unknown size';

  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));

  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(2))} ${sizes[i]}`;
}

/**
 * Validates an uploaded screenshot file against CBL-04 rules.
 *
 * @param {File} file - File selected by user.
 * @returns {{ isValid: boolean, error: string | null }}
 */
export function validateScreenshotFile(file) {
  if (!file) {
    return {
      isValid: false,
      error: 'No file selected. Please select a payment screenshot.',
    };
  }

  // Check 1: Empty file check
  if (file.size === 0) {
    return {
      isValid: false,
      error: 'The selected file is empty (0 bytes). Please select a valid screenshot image.',
    };
  }

  // Check 2: Maximum file size
  if (file.size > IMAGE_CONSTRAINTS.MAX_FILE_SIZE_BYTES) {
    const formattedCurrentSize = formatFileSize(file.size);
    return {
      isValid: false,
      error: `File size exceeds the ${IMAGE_CONSTRAINTS.MAX_FILE_SIZE_LABEL} limit (Current: ${formattedCurrentSize}). Please upload a smaller image.`,
    };
  }

  // Check 3: MIME type & extension validation
  const fileType = file.type?.toLowerCase() || '';
  const fileName = file.name?.toLowerCase() || '';

  const hasValidMime = IMAGE_CONSTRAINTS.SUPPORTED_MIME_TYPES.some(
    (type) => fileType === type
  );

  const hasValidExtension = IMAGE_CONSTRAINTS.SUPPORTED_EXTENSIONS.some(
    (ext) => fileName.endsWith(ext)
  );

  if (!hasValidMime && !hasValidExtension) {
    const supportedList = IMAGE_CONSTRAINTS.SUPPORTED_EXTENSIONS.map((ext) => ext.toUpperCase()).join(', ');
    return {
      isValid: false,
      error: `Unsupported file format. Please upload an image in one of the supported formats: ${supportedList}.`,
    };
  }

  return {
    isValid: true,
    error: null,
  };
}

