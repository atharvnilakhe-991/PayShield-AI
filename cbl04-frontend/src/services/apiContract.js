/**
 * CBL-04: Fake/Manipulated Google Pay (GPay) Receipt Screenshot Detection
 * Frontend-Backend API Contract Specification
 *
 * ⚠️ STATUS: PROPOSED CONTRACT (Awaiting confirmation from Member 3 / Backend Lead)
 * Author: Member 4 (Frontend Development & Integration)
 * Target Backend: FastAPI (Python)
 * Proposed Version: v1
 * Date: 2026-11-03
 *
 * NOTE FOR MEMBER 3:
 * This file serves as both our machine-usable constants/mock definitions
 * and our living interface documentation for frontend and backend coordination.
 * Any proposed schema revisions should be discussed and updated here.
 */

// ============================================================================
// 1. ENDPOINT DEFINITIONS
// ============================================================================
export const PROPOSED_ENDPOINTS = {
  /**
   * Primary image inspection endpoint
   * Method: POST
   * Content-Type: multipart/form-data
   */
  ANALYZE: '/api/v1/analyze',

  /**
   * Backend health check endpoint
   * Method: GET
   * Content-Type: application/json
   */
  HEALTH: '/api/v1/health',
};

// ============================================================================
// 2. IMAGE UPLOAD CONSTRAINTS (CONFIGURABLE PLACEHOLDERS)
// ============================================================================
export const IMAGE_CONSTRAINTS = {
  /**
   * Accepted MIME types for GPay screenshot uploads
   */
  SUPPORTED_MIME_TYPES: [
    'image/jpeg',
    'image/png',
    'image/webp'
  ],

  /**
   * Accepted file extensions
   */
  SUPPORTED_EXTENSIONS: ['.jpg', '.jpeg', '.png', '.webp'],

  /**
   * Maximum upload size placeholder in bytes (Default: 10 MB).
   * TODO: Coordinate with Member 3 based on server memory and CNN inference constraints.
   */
  MAX_FILE_SIZE_BYTES: 10 * 1024 * 1024, // 10 MB placeholder

  /**
   * Max upload size human-readable label
   */
  MAX_FILE_SIZE_LABEL: '10 MB',
};

// ============================================================================
// 3. VERDICT DEFINITIONS & CONFIDENCE SPECIFICATION
// ============================================================================
export const VERDICTS = {
  GENUINE: 'GENUINE',
  EDITED: 'EDITED',
  INCONCLUSIVE: 'INCONCLUSIVE',
  REAL: 'REAL',
  FAKE: 'FAKE',
  SUSPICIOUS: 'SUSPICIOUS',
};

export const VERDICT_DESCRIPTIONS = {
  [VERDICTS.GENUINE]: {
    label: 'Authentic Receipt Verified',
    description: 'Visual layout, fonts, and UPI alignments match genuine Google Pay receipt signatures.',
    colorScheme: 'green',
  },
  [VERDICTS.EDITED]: {
    label: 'Manipulated / Edited Receipt Detected',
    description: 'Definite evidence of tampering, splicing, fabricated fonts, or unnatural digital artifacts detected.',
    colorScheme: 'red',
  },
  [VERDICTS.INCONCLUSIVE]: {
    label: 'Inconclusive / Indeterminate Receipt',
    description: 'Low-confidence artifacts, anomalous compression, or unreadable fields detected. Manual review recommended.',
    colorScheme: 'amber',
  },
  [VERDICTS.REAL]: {
    label: 'Authentic Receipt',
    description: 'No significant visual, structural, or font manipulation detected. Matches genuine GPay receipt signatures.',
    colorScheme: 'green',
  },
  [VERDICTS.FAKE]: {
    label: 'Manipulated / Fake Receipt',
    description: 'Definite evidence of tampering, splicing, fabricated fonts, or unnatural digital artifacts detected.',
    colorScheme: 'red',
  },
  [VERDICTS.SUSPICIOUS]: {
    label: 'Suspicious / Inconclusive',
    description: 'Low-confidence artifacts, anomalous compression, or unreadable fields detected. Manual review recommended.',
    colorScheme: 'amber',
  },
};

export const CONFIDENCE_SPEC = {
  MIN: 0.0,
  MAX: 1.0,
  TYPE: 'float (normalized probability score 0.00 to 1.00)',
  RECOMMENDED_THRESHOLD_DECISION: {
    HIGH_CONFIDENCE: 0.85,
    MODERATE_CONFIDENCE: 0.60,
  },
};

// ============================================================================
// 4. COORDINATE CONVENTION & BOUNDING BOXES
// ============================================================================
/**
 * Coordinate Convention for suspicious_regions:
 * - Origin (0,0): Top-Left corner of the uploaded image.
 * - Units: Pixels relative to the original uploaded image's natural dimensions (width x height).
 *   (Discussion point for Member 3: Alternatively, normalized floats [0.0 - 1.0] could be used.
 *    Pixel coordinates are currently proposed for direct canvas/SVG overlay alignment).
 *
 * Properties per region:
 * - label (string): Short descriptive identifier (e.g. "Amount field", "Timestamp", "Transaction ID")
 * - x (number): X-coordinate of the region's top-left bounding box
 * - y (number): Y-coordinate of the region's top-left bounding box
 * - width (number): Bounding box width
 * - height (number): Bounding box height
 * - score (number): Confidence score of manipulation in this specific localized region (0.0 to 1.0)
 */

// ============================================================================
// 5. HEATMAP DELIVERY METHOD
// ============================================================================
/**
 * Proposed Delivery Method:
 * 1. Primary: Direct URL pointing to the generated Grad-CAM or ELA heatmap image
 *    (e.g., hosted statically by FastAPI under `/static/heatmaps/<id>.png`).
 * 2. Alternative/Fallback: Base64 Data URI string ("data:image/png;base64,...")
 *    if backend opts not to write temporary files to disk.
 * 3. Null: `null` if heatmap generation is disabled, failed, or unnecessary for authentic receipts.
 */

// ============================================================================
// 6. OCR FIELD SPECIFICATION
// ============================================================================
/**
 * Key metadata extracted from the GPay screenshot:
 * - transaction_id (string | null): The UPI transaction reference or Google transaction ID
 * - amount (string | null): Parsed transaction amount formatted with currency symbol (e.g. "₹500")
 * - date (string | null): Parsed date/time string of the transaction (e.g. "2026-11-03 14:30")
 * - sender (string | null): Sender display name or UPI ID if present
 * - receiver (string | null): Receiver/Payee display name or UPI ID
 */

// ============================================================================
// 7. HTTP STATUS CODES & ERROR RESPONSES
// ============================================================================
export const HTTP_STATUS = {
  OK: 200,
  BAD_REQUEST: 400,
  PAYLOAD_TOO_LARGE: 413,
  UNSUPPORTED_MEDIA_TYPE: 415,
  UNPROCESSABLE_ENTITY: 422,
  INTERNAL_SERVER_ERROR: 500,
  SERVICE_UNAVAILABLE: 503,
};

/**
 * Standard Error Response Schema:
 * {
 *   "detail": "Descriptive human-readable error message",
 *   "error_code": "INVALID_IMAGE_FORMAT | FILE_TOO_LARGE | OCR_FAILED | MODEL_TIMEOUT",
 *   "request_id": "optional-request-trace-id"
 * }
 */

// ============================================================================
// 8. PROPOSED MOCK CONTRACT DATA (For Frontend Development & Testing)
// ============================================================================
export const MOCK_API_RESPONSES = {
  /**
   * Example: Detected Fake Receipt
   */
  FAKE_SAMPLE: {
    request_id: "demo-fake-001",
    verdict: VERDICTS.FAKE,
    confidence: 0.95,
    message: "Analysis completed. Image exhibits localized tampering around amount and font inconsistencies.",
    ocr_data: {
      transaction_id: "UPI/234809283402",
      amount: "₹5,000",
      date: "2026-11-03",
      sender: "Rahul Sharma",
      receiver: "Merchant Store"
    },
    manipulation_analysis: {
      detected: true,
      suspicious_regions: [
        {
          label: "Amount field",
          x: 100,
          y: 200,
          width: 140,
          height: 48,
          score: 0.94
        },
        {
          label: "Transaction ID alignment",
          x: 80,
          y: 350,
          width: 260,
          height: 32,
          score: 0.88
        }
      ],
      heatmap_url: null
    }
  },

  /**
   * Example: Authentic Receipt
   */
  REAL_SAMPLE: {
    request_id: "demo-real-002",
    verdict: VERDICTS.REAL,
    confidence: 0.98,
    message: "Analysis completed. Receipt integrity verified; no manipulation detected.",
    ocr_data: {
      transaction_id: "CIC492049203",
      amount: "₹350",
      date: "2026-11-03",
      sender: "Priya Patel",
      receiver: "Cafe Delight"
    },
    manipulation_analysis: {
      detected: false,
      suspicious_regions: [],
      heatmap_url: null
    }
  },

  /**
   * Example: Suspicious / Inconclusive Receipt
   */
  SUSPICIOUS_SAMPLE: {
    request_id: "demo-suspicious-003",
    verdict: VERDICTS.SUSPICIOUS,
    confidence: 0.62,
    message: "Analysis completed with moderate uncertainty due to heavy image compression artifacts.",
    ocr_data: {
      transaction_id: "UNKNOWN",
      amount: "₹1,200",
      date: "2026-11-02",
      sender: null,
      receiver: "Vendor Mart"
    },
    manipulation_analysis: {
      detected: true,
      suspicious_regions: [
        {
          label: "Compressed text block",
          x: 110,
          y: 280,
          width: 180,
          height: 40,
          score: 0.65
        }
      ],
      heatmap_url: null
    }
  }
};

