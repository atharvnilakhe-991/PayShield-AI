import axios from 'axios';
import { PROPOSED_ENDPOINTS } from './apiContract';

/**
 * CBL-04: API Client Service
 * Configured Axios instance and communication handlers for FastAPI backend.
 *
 * NOTE: Backend URL is driven by VITE_API_BASE_URL from the environment.
 * Default local development URL is http://localhost:8000.
 */

// Resolve Base URL dynamically from Vite environment variable
export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

// Create configured Axios instance
export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 60000, // 60 seconds inference grace period for CPU EasyOCR/CNN workloads
  headers: {
    Accept: 'application/json',
  },
});

/**
 * Standardizes API errors into a consistent structure for frontend consumption.
 * @param {Error} error - Axios error or generic error
 * @returns {Object} Standardized error object { message, status, detail, isNetworkError }
 */
export function formatApiError(error) {
  if (axios.isAxiosError(error)) {
    if (error.response) {
      // Server responded with non-2xx status
      const data = error.response.data;
      const primaryMessage = data?.message || (typeof data?.detail === 'string' ? data.detail : null);
      const detail = data?.detail || data?.message || 'Server error occurred.';
      return {
        message: primaryMessage || (typeof detail === 'string' ? detail : JSON.stringify(detail)),
        status: error.response.status,
        detail,
        isNetworkError: false,
      };
    } else if (error.request) {
      // Request was made but no response received (Backend offline / CORS / Timeout)
      return {
        message: `Cannot connect to backend server at ${API_BASE_URL}. Please verify the FastAPI service is running.`,
        status: null,
        detail: error.message,
        isNetworkError: true,
      };
    }
  }

  return {
    message: error.message || 'An unexpected error occurred.',
    status: null,
    detail: null,
    isNetworkError: false,
  };
}

/**
 * Placeholder API function for image analysis.
 * Sends an uploaded receipt screenshot to the FastAPI backend.
 *
 * @param {File|Blob} imageFile - The image file to analyze
 * @param {Function} [onUploadProgress] - Optional callback for upload progress tracking
 * @returns {Promise<Object>} The analysis result adhering to the proposed API contract
 */
export async function analyzeImage(imageFile, onUploadProgress) {
  if (!imageFile) {
    throw new Error('Please provide an image file to analyze.');
  }

  const formData = new FormData();
  // Field name 'image' as defined in FastAPI backend endpoint
  formData.append('image', imageFile);

  try {
    // Note: Do not pass explicit Content-Type header so Axios & browser set multipart boundary
    const response = await apiClient.post(PROPOSED_ENDPOINTS.ANALYZE, formData, {
      onUploadProgress,
    });

    return response.data;
  } catch (error) {
    throw formatApiError(error);
  }
}

/**
 * Placeholder health check to verify backend connectivity.
 * @returns {Promise<Object>} Backend health response
 */
export async function checkBackendHealth() {
  try {
    const response = await apiClient.get(PROPOSED_ENDPOINTS.HEALTH);
    return response.data;
  } catch (error) {
    throw formatApiError(error);
  }
}

export default apiClient;

