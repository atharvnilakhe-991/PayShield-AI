"""
Phase 18: Inference-Realistic Candidate ROI Selector.
Selects candidate payment field regions strictly using OCR detections and semantic anomaly indicators.
Strictly adheres to Phase 18 guidelines:
- ZERO ground-truth bounding boxes used
- ZERO edit-category or clean/edited label leakage
- Missing or ambiguous fields remain missing/ambiguous
- Fully preserves provenance and OCR coordinates
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class CandidateROI:
    candidate_id: str             # e.g. "amount", "receiver", "transaction_id", "date", "status"
    field_name: str
    is_available: bool            # True if localized by OCR, False if missing
    raw_bbox_norm: Optional[List[float]]  # [xmin, ymin, xmax, ymax] in [0, 1]
    raw_bbox_pixel: Optional[List[int]]   # [x0, y0, x1, y1] in native image pixels
    ocr_confidence: Optional[float]
    raw_text: Optional[str]
    extraction_status: str        # "extracted", "ambiguous", "not_found"
    anomaly_flag: bool            # True if semantic / geometric heuristic flagged this field
    routing_reason: str           # Explanation of why this candidate was flagged / routed
    priority_rank: int            # Routing priority (1 = highest)


class InferenceCandidateSelector:
    """
    Selects and prioritizes candidate payment regions for high-resolution visual inspection
    using exclusively inference-available signals (OCR geometry and semantic anomalies).
    """

    SUPPORTED_FIELDS = ["amount", "receiver", "transaction_id", "date", "status"]

    def __init__(self, semantic_anomaly_routing: bool = True) -> None:
        self.semantic_anomaly_routing = semantic_anomaly_routing

    def evaluate_candidates(
        self,
        image_dimensions: Tuple[int, int],
        detailed_fields: Dict[str, Any],
        field_anomalies: Dict[str, bool],
        reason_codes: List[str],
        semantic_manipulation_score: float,
    ) -> List[CandidateROI]:
        """
        Parses OCR detailed fields and semantic features to construct candidate ROIs.
        Does NOT look at ground-truth annotations or edit categories.
        """
        w_img, h_img = image_dimensions
        candidates: List[CandidateROI] = []

        # 1. Evaluate Amount Field
        amt_info = detailed_fields.get("amount", {})
        amt_detected = amt_info.get("is_detected", False)
        amt_status = amt_info.get("extraction_status", "not_found")
        amt_norm = amt_info.get("norm_box")
        amt_pix = None
        if amt_norm is not None:
            amt_pix = [
                int(round(amt_norm[0] * w_img)),
                int(round(amt_norm[1] * h_img)),
                int(round(amt_norm[2] * w_img)),
                int(round(amt_norm[3] * h_img)),
            ]

        amt_anomaly = (
            field_anomalies.get("amount_anomaly", False)
            or (amt_status == "ambiguous")
            or any("AMOUNT" in rc for rc in reason_codes)
        )
        amt_reasons = []
        if field_anomalies.get("amount_anomaly", False):
            amt_reasons.append("amount_anomaly_flag")
        if amt_status == "ambiguous":
            amt_reasons.append("ocr_ambiguous_amount")
        for rc in reason_codes:
            if "AMOUNT" in rc:
                amt_reasons.append(rc)

        candidates.append(CandidateROI(
            candidate_id="amount",
            field_name="amount",
            is_available=(amt_norm is not None),
            raw_bbox_norm=amt_norm,
            raw_bbox_pixel=amt_pix,
            ocr_confidence=amt_info.get("confidence"),
            raw_text=amt_info.get("raw_text"),
            extraction_status=amt_status,
            anomaly_flag=amt_anomaly,
            routing_reason="; ".join(amt_reasons) if amt_reasons else "normal_detection",
            priority_rank=1 if amt_anomaly else 3,
        ))

        # 2. Evaluate Recipient Field
        rec_info = detailed_fields.get("receiver", {})
        rec_detected = rec_info.get("is_detected", False)
        rec_status = rec_info.get("extraction_status", "not_found")
        rec_norm = rec_info.get("norm_box")
        rec_pix = None
        if rec_norm is not None:
            rec_pix = [
                int(round(rec_norm[0] * w_img)),
                int(round(rec_norm[1] * h_img)),
                int(round(rec_norm[2] * w_img)),
                int(round(rec_norm[3] * h_img)),
            ]

        rec_conf = rec_info.get("confidence") or 1.0
        rec_anomaly = (
            field_anomalies.get("recipient_anomaly", False)
            or field_anomalies.get("upi_handle_anomaly", False)
            or any("UPI_ID" in rc for rc in reason_codes)
            or (rec_conf < 0.60 and rec_detected)
        )
        rec_reasons = []
        if field_anomalies.get("recipient_anomaly", False):
            rec_reasons.append("recipient_anomaly_flag")
        if field_anomalies.get("upi_handle_anomaly", False):
            rec_reasons.append("upi_handle_anomaly")
        for rc in reason_codes:
            if "UPI_ID" in rc:
                rec_reasons.append(rc)
        if rec_conf < 0.60 and rec_detected:
            rec_reasons.append(f"low_rec_conf_{rec_conf:.2f}")

        candidates.append(CandidateROI(
            candidate_id="receiver",
            field_name="receiver",
            is_available=(rec_norm is not None),
            raw_bbox_norm=rec_norm,
            raw_bbox_pixel=rec_pix,
            ocr_confidence=rec_info.get("confidence"),
            raw_text=rec_info.get("raw_text"),
            extraction_status=rec_status,
            anomaly_flag=rec_anomaly,
            routing_reason="; ".join(rec_reasons) if rec_reasons else "normal_detection",
            priority_rank=1 if rec_anomaly else 3,
        ))

        # 3. Evaluate Transaction ID Field
        txn_info = detailed_fields.get("transaction_id", {})
        txn_detected = txn_info.get("is_detected", False)
        txn_status = txn_info.get("extraction_status", "not_found")
        txn_norm = txn_info.get("norm_box")
        txn_pix = None
        if txn_norm is not None:
            txn_pix = [
                int(round(txn_norm[0] * w_img)),
                int(round(txn_norm[1] * h_img)),
                int(round(txn_norm[2] * w_img)),
                int(round(txn_norm[3] * h_img)),
            ]

        txn_anomaly = (
            field_anomalies.get("ref_id_anomaly", False)
            or any("REF_ID" in rc for rc in reason_codes)
        )
        txn_reasons = []
        if field_anomalies.get("ref_id_anomaly", False):
            txn_reasons.append("ref_id_anomaly_flag")
        for rc in reason_codes:
            if "REF_ID" in rc:
                txn_reasons.append(rc)

        candidates.append(CandidateROI(
            candidate_id="transaction_id",
            field_name="transaction_id",
            is_available=(txn_norm is not None),
            raw_bbox_norm=txn_norm,
            raw_bbox_pixel=txn_pix,
            ocr_confidence=txn_info.get("confidence"),
            raw_text=txn_info.get("raw_text"),
            extraction_status=txn_status,
            anomaly_flag=txn_anomaly,
            routing_reason="; ".join(txn_reasons) if txn_reasons else "normal_detection",
            priority_rank=1 if txn_anomaly else 4,
        ))

        # 4. Evaluate Date Field
        date_info = detailed_fields.get("date", {})
        date_detected = date_info.get("is_detected", False)
        date_status = date_info.get("extraction_status", "not_found")
        date_norm = date_info.get("norm_box")
        date_pix = None
        if date_norm is not None:
            date_pix = [
                int(round(date_norm[0] * w_img)),
                int(round(date_norm[1] * h_img)),
                int(round(date_norm[2] * w_img)),
                int(round(date_norm[3] * h_img)),
            ]

        date_anomaly = (
            field_anomalies.get("date_anomaly", False)
            or any("DATE" in rc or "TIME" in rc for rc in reason_codes)
        )
        date_reasons = []
        if field_anomalies.get("date_anomaly", False):
            date_reasons.append("date_anomaly_flag")
        for rc in reason_codes:
            if "DATE" in rc or "TIME" in rc:
                date_reasons.append(rc)

        candidates.append(CandidateROI(
            candidate_id="date",
            field_name="date",
            is_available=(date_norm is not None),
            raw_bbox_norm=date_norm,
            raw_bbox_pixel=date_pix,
            ocr_confidence=date_info.get("confidence"),
            raw_text=date_info.get("raw_text"),
            extraction_status=date_status,
            anomaly_flag=date_anomaly,
            routing_reason="; ".join(date_reasons) if date_reasons else "normal_detection",
            priority_rank=2 if date_anomaly else 4,
        ))

        return candidates

    def route_candidates(
        self,
        candidates: List[CandidateROI],
        semantic_manipulation_score: float,
        layout_anomaly: bool = False,
    ) -> List[CandidateROI]:
        """
        Determines which candidate regions are selected for crop neural inspection.
        - Semantic-Guided Mode: Only routes candidates that possess active anomaly flags.
          If all candidates are clean and semantic score is 0.0, zero candidates are routed
          (bypassing crop false positives on verified clean receipts).
        - Unguided Mode: Routes all available detected candidates.
        """
        if not self.semantic_anomaly_routing:
            # Unguided mode (ablation): route all available fields
            return [c for c in candidates if c.is_available]

        # Semantic-guided routing:
        flagged = [c for c in candidates if c.is_available and c.anomaly_flag]
        if flagged:
            return flagged

        # If overall layout anomaly is present, inspect amount and receiver
        if layout_anomaly and semantic_manipulation_score > 0.20:
            return [c for c in candidates if c.is_available and c.candidate_id in ["amount", "receiver"]]

        # Clean receipt: zero candidates routed to visual inspection
        return []

