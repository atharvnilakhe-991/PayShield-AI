/**
 * CBL-04: Mock API Service
 * Phase 2 (H1–H3) Simulation Layer for GPay Receipt Screenshot Verification
 *
 * NOTE: This service runs 100% on the client side for Phase 2 prototyping.
 * It simulates backend network latency and model inference without making
 * any real HTTP requests to the FastAPI backend.
 */

import { MOCK_API_RESPONSES, VERDICTS } from './apiContract';

// Re-export mock response fixtures strictly adhering to the confirmed API contract
export const MOCK_SAMPLES = {
  REAL_SAMPLE: { ...MOCK_API_RESPONSES.REAL_SAMPLE },
  FAKE_SAMPLE: { ...MOCK_API_RESPONSES.FAKE_SAMPLE },
  SUSPICIOUS_SAMPLE: { ...MOCK_API_RESPONSES.SUSPICIOUS_SAMPLE },
};

// Internal counter to cycle through verdicts across consecutive uploads
let sampleCycleIndex = 0;
const sampleKeys = ['FAKE_SAMPLE', 'REAL_SAMPLE', 'SUSPICIOUS_SAMPLE'];

/**
 * Simulates the POST /api/v1/analyze endpoint.
 *
 * @param {File} imageFile - The uploaded receipt image file.
 * @param {Object} [options] - Optional simulation overrides.
 * @param {string} [options.forcedVerdict] - Specific verdict to test ('REAL' | 'FAKE' | 'SUSPICIOUS').
 * @param {boolean} [options.simulateError] - If true, simulates a server failure.
 * @param {number} [options.delayMs] - Milliseconds to simulate (default: ~2200ms).
 * @param {Function} [options.onProgress] - Callback for simulated progress percentage (0 - 100).
 * @returns {Promise<Object>} The simulated analysis response matching apiContract.js.
 */
export async function analyzeImageMock(imageFile, options = {}) {
  const {
    forcedVerdict = null,
    simulateError = false,
    delayMs = 2300,
    onProgress = null,
  } = options;

  if (!imageFile) {
    throw new Error('No image file provided for analysis.');
  }

  // Simulate stepwise progress over the duration
  if (onProgress && typeof onProgress === 'function') {
    const steps = [
      { pct: 20, time: delayMs * 0.2 },
      { pct: 50, time: delayMs * 0.45 },
      { pct: 80, time: delayMs * 0.75 },
      { pct: 100, time: delayMs * 0.95 },
    ];

    steps.forEach(({ pct, time }) => {
      setTimeout(() => {
        onProgress(pct);
      }, time);
    });
  }

  // Simulate network & inference delay
  await new Promise((resolve) => setTimeout(resolve, delayMs));

  // Simulate optional server-side error for testing error recovery
  if (simulateError) {
    const error = new Error('Simulated analysis error: OCR pipeline timed out.');
    error.status = 500;
    error.detail = 'Simulated OCR pipeline timeout during image text extraction.';
    throw error;
  }

  // Select mock fixture based on forced verdict or cyclic rotation
  let chosenKey;
  if (forcedVerdict && VERDICTS[forcedVerdict]) {
    chosenKey = Object.keys(MOCK_SAMPLES).find(
      (key) => MOCK_SAMPLES[key].verdict === forcedVerdict
    ) || 'FAKE_SAMPLE';
  } else {
    chosenKey = sampleKeys[sampleCycleIndex % sampleKeys.length];
    sampleCycleIndex += 1;
  }

  const baseSample = MOCK_SAMPLES[chosenKey];

  // Deep clone to prevent mutations and inject a unique synthetic request_id
  const response = JSON.parse(JSON.stringify(baseSample));
  const timestampSuffix = Math.floor(1000 + Math.random() * 9000);
  response.request_id = `sim-${Date.now().toString().slice(-4)}-${timestampSuffix}`;

  return response;
}

