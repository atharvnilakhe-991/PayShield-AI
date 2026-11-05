"""
Semantic Feature Extraction Engine for CBL-04 Google Pay Screenshot Analysis.
Consumes structured OCR output to extract deterministic, auditable semantic features,
layout geometry heuristics, and machine-readable evidence provenance.
Adheres strictly to Phase 6 scope: does NOT invent missing fields, does NOT decide final
verdict, does NOT claim banking settlement verification.
"""

from __future__ import annotations

import calendar
import datetime
import logging
import re
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from app.schemas.enums import ReasonCodeEnum
from app.schemas.ocr import ExtractedTransactionData, OCRExtractedField, OCRResult
from app.schemas.semantic import (
    FieldAnomalyIndicators,
    OCRConfidenceInfo,
    SemanticBranchOutput,
    SemanticFeatureEvidence,
    SemanticFeatureVector,
)

logger = logging.getLogger("cbl04.semantic")


class SemanticFeatureService:
    """
    Deterministic semantic feature engine evaluating OCR text tokens and bounding regions
    for Google Pay dark-theme receipts.
    Produces auditable feature vectors and rule-level evidence provenance.
    """

    # Indian currency grouping: e.g. 1,500.00, 15,000, 1,00,000.00
    INDIAN_CURRENCY_REGEX = re.compile(
        r"^(?:[₹\{\[\(\?\}\]\)\$£€§]|Ft|Rs\.?|INR)?\s*(\d{1,2}(?:,\d{2})*|\d{1,3})(?:,\d{3})*(?:\.\d{1,2})?$|^\d+(?:\.\d{1,2})?$",
        re.IGNORECASE,
    )

    # Valid UPI VPA syntax (RFC-like: alphanumeric/dots/dashes/underscores @ bankhandle)
    UPI_VPA_SYNTAX_REGEX = re.compile(
        r"^[a-zA-Z0-9][a-zA-Z0-9.\-_]{1,255}@[a-zA-Z0-9]{2,64}$"
    )

    # 12-digit UPI RRN standard or synthetic DEMO-UPI format
    STANDARD_UPI_RRN_REGEX = re.compile(r"^\d{12}$")
    SYNTHETIC_UPI_RRN_REGEX = re.compile(r"^DEMO[\s\-_]*UPI[\s\-_]*\d+$", re.IGNORECASE)

    # Standard Google transaction ID (CIC... or DEMO-GPAY-...)
    GPAY_TXN_ID_REGEX = re.compile(r"^(?:CIC[a-zA-Z0-9_\-]+|DEMO[\s\-_]*GPAY[\s\-_]*[A-Za-z0-9\-]+)$", re.IGNORECASE)

    # Authentic GPay transaction status keywords
    AUTHENTIC_STATUSES = {
        "completed",
        "paid",
        "payment completed",
        "payment to",
        "payment started",
        "processing",
        "transferred",
        "success",
        "successful",
        "failed",
        "pending",
    }

    # Month mapping for calendar date parsing
    MONTH_NAMES = {
        "jan": 1, "january": 1,
        "feb": 2, "february": 2,
        "mar": 3, "march": 3,
        "apr": 4, "april": 4,
        "may": 5,
        "jun": 6, "june": 6,
        "jul": 7, "july": 7,
        "aug": 8, "august": 8,
        "sep": 9, "september": 9,
        "oct": 10, "october": 10,
        "nov": 11, "november": 11,
        "dec": 12, "december": 12,
    }

    def __init__(self, version: str = "v1.0.0-phase6") -> None:
        self.version = version

    def extract_features(
        self,
        ocr_result: OCRResult,
        image_dimensions: Optional[Tuple[int, int]] = None,
    ) -> SemanticBranchOutput:
        """
        Extracts semantic consistency features, anomaly indicators, and rule evidence
        from actual OCR extraction output.
        """
        start_time = time.perf_counter()
        fields = ocr_result.fields
        detailed = ocr_result.detailed_fields
        raw_tokens = ocr_result.raw_detections

        evidence: Dict[str, SemanticFeatureEvidence] = {}
        reason_codes: List[ReasonCodeEnum] = []
        anomalies = FieldAnomalyIndicators()

        # -------------------------------------------------------------
        # 1. AMOUNT FEATURES (R_AMT_001 to R_AMT_006)
        # -------------------------------------------------------------
        amt_field = detailed.get("amount")
        amt_raw = amt_field.raw_text if amt_field else fields.amount
        amt_num = amt_field.normalized_value if amt_field else fields.amount_numeric
        amt_conf = amt_field.confidence if amt_field else None
        amt_box = amt_field.bounding_box if amt_field else None

        amt_present = bool(amt_raw is not None and str(amt_raw).strip())
        amt_parse_success = bool(amt_num is not None and isinstance(amt_num, (int, float)))

        # Format check
        amt_valid_format = True
        amt_status = "available"
        if not amt_present:
            amt_valid_format = True  # Missing is not automatically invalid format
            amt_status = "not_available"
        else:
            cleaned_amt = str(amt_raw).strip()
            # If multiple decimal points or invalid chars
            if cleaned_amt.count(".") > 1 or cleaned_amt.count(",") > 3:
                amt_valid_format = False
                amt_status = "invalid_format"
            elif not self.INDIAN_CURRENCY_REGEX.match(cleaned_amt.replace(" ", "")):
                amt_valid_format = False
                amt_status = "invalid_format"

        if not amt_valid_format:
            anomalies.amount_anomaly = True
            reason_codes.append(ReasonCodeEnum.AMOUNT_FORMAT_ANOMALY)

        evidence["amount_valid_format"] = SemanticFeatureEvidence(
            feature_name="amount_valid_format",
            feature_value=amt_valid_format,
            feature_status=amt_status,
            source_ocr_text=amt_raw,
            source_bounding_box=amt_box,
            rule_id="R_AMT_003",
            explanation=(
                f"Amount format verified: '{amt_raw}'"
                if amt_valid_format
                else f"Amount format irregular or non-standard currency grouping: '{amt_raw}'"
            ),
            ocr_confidence=amt_conf,
            is_missing=not amt_present,
        )

        # Currency symbol check
        amt_status_code = getattr(amt_field, "extraction_status", "not_found") if amt_field else "not_found"
        has_currency_symbol = True
        if amt_present:
            amt_str = str(amt_raw).strip()
            has_currency_symbol = any(
                sym in amt_str
                for sym in ["₹", "{", "}", "[", "]", "(", ")", "?", "$", "£", "€", "§", "INR", "inr", "Rs", "rs", "Ft", "ft"]
            )
            if not has_currency_symbol:
                if amt_status_code == "ambiguous":
                    # Extraction is ambiguous (e.g. OCR glyph limitation on Rupee symbol) - do NOT flag anomaly
                    evidence["amount_currency_symbol_valid"] = SemanticFeatureEvidence(
                        feature_name="amount_currency_symbol_valid",
                        feature_value=True,
                        feature_status="ambiguous",
                        source_ocr_text=amt_raw,
                        source_bounding_box=amt_box,
                        rule_id="R_AMT_004",
                        explanation=f"Amount candidate '{amt_raw}' localized without confirmed currency symbol prefix; extraction status is ambiguous due to OCR engine limitations. Safeguard active: anomaly not asserted.",
                        ocr_confidence=amt_conf,
                        is_missing=False,
                    )
                else:
                    evidence["amount_currency_symbol_valid"] = SemanticFeatureEvidence(
                        feature_name="amount_currency_symbol_valid",
                        feature_value=False,
                        feature_status="invalid_format",
                        source_ocr_text=amt_raw,
                        source_bounding_box=amt_box,
                        rule_id="R_AMT_004",
                        explanation=f"Missing standard currency indicator (₹) in confirmed amount token: '{amt_raw}'",
                        ocr_confidence=amt_conf,
                        is_missing=False,
                    )
                    anomalies.amount_anomaly = True
                    if ReasonCodeEnum.AMOUNT_CURRENCY_SYMBOL_MISMATCH not in reason_codes:
                        reason_codes.append(ReasonCodeEnum.AMOUNT_CURRENCY_SYMBOL_MISMATCH)
            else:
                evidence["amount_currency_symbol_valid"] = SemanticFeatureEvidence(
                    feature_name="amount_currency_symbol_valid",
                    feature_value=True,
                    feature_status="available",
                    source_ocr_text=amt_raw,
                    source_bounding_box=amt_box,
                    rule_id="R_AMT_004",
                    explanation=f"Currency symbol verified in amount token: '{amt_raw}'",
                    ocr_confidence=amt_conf,
                    is_missing=False,
                )
        else:
            evidence["amount_currency_symbol_valid"] = SemanticFeatureEvidence(
                feature_name="amount_currency_symbol_valid",
                feature_value=True,
                feature_status="not_available",
                source_ocr_text=None,
                source_bounding_box=None,
                rule_id="R_AMT_004",
                explanation="Amount not detected in OCR tokens; currency symbol check skipped",
                ocr_confidence=None,
                is_missing=True,
            )

        # Reasonableness limit check (0 < amount <= 100,000 for standard UPI P2P)
        amt_reasonable = True
        if amt_parse_success and amt_num is not None:
            if amt_num <= 0.0 or amt_num > 1000000.0:
                amt_reasonable = False
                anomalies.amount_anomaly = True
                if ReasonCodeEnum.AMOUNT_DECIMAL_ANOMALY not in reason_codes:
                    reason_codes.append(ReasonCodeEnum.AMOUNT_DECIMAL_ANOMALY)

        evidence["amount_numeric_value_reasonable"] = SemanticFeatureEvidence(
            feature_name="amount_numeric_value_reasonable",
            feature_value=amt_reasonable,
            feature_status="available" if amt_parse_success else "not_available",
            source_ocr_text=str(amt_num) if amt_parse_success else amt_raw,
            source_bounding_box=amt_box,
            rule_id="R_AMT_005",
            explanation=(
                f"Numeric amount {amt_num} within plausible range (0 < amount <= 1,000,000)"
                if amt_reasonable
                else f"Numeric amount {amt_num} outside standard retail UPI limits"
            ),
            ocr_confidence=amt_conf,
            is_missing=not amt_parse_success,
        )

        # -------------------------------------------------------------
        # 2. RECIPIENT & UPI FEATURES (R_REC_001 to R_REC_005)
        # -------------------------------------------------------------
        rx_field = detailed.get("receiver")
        rx_upi_field = detailed.get("receiver_upi")
        tx_field = detailed.get("sender")
        tx_upi_field = detailed.get("sender_upi")

        rx_name = rx_field.normalized_value if rx_field else fields.receiver
        rx_upi = rx_upi_field.normalized_value if rx_upi_field else fields.receiver_upi
        tx_name = tx_field.normalized_value if tx_field else fields.sender
        tx_upi = tx_upi_field.normalized_value if tx_upi_field else fields.sender_upi

        # UPI Handle syntax validity
        upi_valid = True
        upi_status = "available"
        if rx_upi:
            clean_upi = str(rx_upi).strip()
            if not self.UPI_VPA_SYNTAX_REGEX.match(clean_upi):
                upi_valid = False
                upi_status = "invalid_format"
                anomalies.recipient_anomaly = True
                anomalies.upi_handle_anomaly = True
                reason_codes.append(ReasonCodeEnum.UPI_ID_SYNTAX_INVALID)
        else:
            upi_valid = True  # Not available is not an anomaly
            upi_status = "not_available"

        evidence["upi_handle_valid_format"] = SemanticFeatureEvidence(
            feature_name="upi_handle_valid_format",
            feature_value=upi_valid,
            feature_status=upi_status,
            source_ocr_text=rx_upi,
            source_bounding_box=rx_upi_field.bounding_box if rx_upi_field else None,
            rule_id="R_REC_003",
            explanation=(
                f"Payee UPI handle adheres to standard syntax: '{rx_upi}'"
                if upi_valid and rx_upi
                else ("Payee UPI handle syntax invalid or contains illegal characters" if rx_upi else "No payee UPI handle detected")
            ),
            ocr_confidence=rx_upi_field.confidence if rx_upi_field else None,
            is_missing=rx_upi is None,
        )

        # Sender vs Receiver distinctness
        distinct = True
        distinct_status = "available"
        if rx_name and tx_name:
            if str(rx_name).strip().lower() == str(tx_name).strip().lower():
                distinct = False
                distinct_status = "invalid_format"
                anomalies.recipient_anomaly = True
                reason_codes.append(ReasonCodeEnum.SENDER_RECEIVER_IDENTICAL)
        elif rx_upi and tx_upi:
            if str(rx_upi).strip().lower() == str(tx_upi).strip().lower():
                distinct = False
                distinct_status = "invalid_format"
                anomalies.recipient_anomaly = True
                reason_codes.append(ReasonCodeEnum.SENDER_RECEIVER_IDENTICAL)
        else:
            distinct_status = "not_available"

        evidence["sender_receiver_distinct"] = SemanticFeatureEvidence(
            feature_name="sender_receiver_distinct",
            feature_value=distinct,
            feature_status=distinct_status,
            source_ocr_text=f"To: {rx_name or rx_upi} | From: {tx_name or tx_upi}",
            source_bounding_box=rx_field.bounding_box if rx_field else None,
            rule_id="R_REC_004",
            explanation=(
                "Sender and receiver identifiers are distinct"
                if distinct
                else "Sender and receiver appear identical (self-transfer anomaly in standard P2P receipt)"
            ),
            ocr_confidence=rx_field.confidence if rx_field else None,
            is_missing=(rx_name is None and rx_upi is None) or (tx_name is None and tx_upi is None),
        )

        # -------------------------------------------------------------
        # 3. DATE & TIME FEATURES (R_DAT_001 to R_DAT_005)
        # -------------------------------------------------------------
        date_field = detailed.get("date")
        time_field = detailed.get("time")
        date_raw = date_field.normalized_value if date_field else fields.date
        time_raw = time_field.normalized_value if time_field else fields.time

        # Validate calendar date
        date_valid, date_status, date_expl = self._validate_calendar_date(date_raw)
        if not date_valid and date_status == "invalid_format":
            anomalies.date_anomaly = True
            reason_codes.append(ReasonCodeEnum.TIMESTAMP_CHRONOLOGY_ANOMALY)

        evidence["date_chronology_valid"] = SemanticFeatureEvidence(
            feature_name="date_chronology_valid",
            feature_value=date_valid,
            feature_status=date_status,
            source_ocr_text=date_raw,
            source_bounding_box=date_field.bounding_box if date_field else None,
            rule_id="R_DAT_003",
            explanation=date_expl,
            ocr_confidence=date_field.confidence if date_field else None,
            is_missing=date_raw is None,
        )

        # Validate time format
        time_valid, time_status, time_expl = self._validate_time_format(time_raw)
        if not time_valid and time_status == "invalid_format":
            anomalies.date_anomaly = True
            reason_codes.append(ReasonCodeEnum.TIME_FORMAT_INVALID)

        evidence["time_format_valid"] = SemanticFeatureEvidence(
            feature_name="time_format_valid",
            feature_value=time_valid,
            feature_status=time_status,
            source_ocr_text=time_raw,
            source_bounding_box=time_field.bounding_box if time_field else None,
            rule_id="R_DAT_004",
            explanation=time_expl,
            ocr_confidence=time_field.confidence if time_field else None,
            is_missing=time_raw is None,
        )

        # -------------------------------------------------------------
        # 4. TRANSACTION REFERENCE FEATURES (R_REF_001 to R_REF_004)
        # -------------------------------------------------------------
        ref_field = detailed.get("transaction_id")
        ref_raw = ref_field.normalized_value if ref_field else fields.transaction_id
        ref_box = ref_field.bounding_box if ref_field else None
        ref_conf = ref_field.confidence if ref_field else None

        ref_len_valid = True
        ref_num_only = True
        ref_status = "available"

        if ref_raw:
            ref_clean = re.sub(r"[\s\-_]", "", str(ref_raw)).upper()
            if "DEMOUPI" in ref_clean:
                # Synthetic format
                ref_len_valid = True
                ref_num_only = True
                ref_status = "available"
            elif ref_clean.isdigit():
                # Real UPI RRN: must be exactly 12 digits
                ref_num_only = True
                if len(ref_clean) == 12:
                    ref_len_valid = True
                    ref_status = "available"
                else:
                    ref_len_valid = False
                    ref_status = "invalid_format"
                    anomalies.ref_id_anomaly = True
                    reason_codes.append(ReasonCodeEnum.REF_ID_FORMAT_ANOMALY)
            else:
                # Contains letters or symbols in standard numeric RRN field
                ref_num_only = False
                ref_len_valid = False
                ref_status = "invalid_format"
                anomalies.ref_id_anomaly = True
                reason_codes.append(ReasonCodeEnum.REF_ID_FORMAT_ANOMALY)
        else:
            ref_status = "not_available"

        evidence["ref_id_length_valid"] = SemanticFeatureEvidence(
            feature_name="ref_id_length_valid",
            feature_value=ref_len_valid,
            feature_status=ref_status,
            source_ocr_text=ref_raw,
            source_bounding_box=ref_box,
            rule_id="R_REF_002",
            explanation=(
                f"UPI Reference format verified: '{ref_raw}'"
                if ref_len_valid and ref_raw
                else ("UPI Reference number length or format irregular" if ref_raw else "Transaction ID not detected")
            ),
            ocr_confidence=ref_conf,
            is_missing=ref_raw is None,
        )

        evidence["ref_id_numeric_only"] = SemanticFeatureEvidence(
            feature_name="ref_id_numeric_only",
            feature_value=ref_num_only,
            feature_status=ref_status,
            source_ocr_text=ref_raw,
            source_bounding_box=ref_box,
            rule_id="R_REF_003",
            explanation=(
                "UPI Reference contains digits only"
                if ref_num_only and ref_raw
                else ("Non-numeric characters found in numeric UPI reference field" if ref_raw else "Transaction ID not detected")
            ),
            ocr_confidence=ref_conf,
            is_missing=ref_raw is None,
        )

        # -------------------------------------------------------------
        # 5. STATUS FEATURES (R_STA_001 to R_STA_003)
        # -------------------------------------------------------------
        status_field = detailed.get("status")
        status_raw = status_field.normalized_value if status_field else fields.status
        status_authentic = True
        status_state = "available"

        if status_raw:
            s_clean = str(status_raw).strip().lower()
            if any(auth_s in s_clean for auth_s in self.AUTHENTIC_STATUSES):
                status_authentic = True
                status_state = "available"
            else:
                status_authentic = False
                status_state = "invalid_format"
                anomalies.status_anomaly = True
                reason_codes.append(ReasonCodeEnum.STATUS_TEXT_ANOMALY)
        else:
            status_state = "not_available"

        evidence["status_text_authentic"] = SemanticFeatureEvidence(
            feature_name="status_text_authentic",
            feature_value=status_authentic,
            feature_status=status_state,
            source_ocr_text=status_raw,
            source_bounding_box=status_field.bounding_box if status_field else None,
            rule_id="R_STA_002",
            explanation=(
                f"Transaction status matches authentic GPay lifecycle label: '{status_raw}'"
                if status_authentic and status_raw
                else ("Non-standard or fabricated transaction status keyword" if status_raw else "Status text not localized")
            ),
            ocr_confidence=status_field.confidence if status_field else None,
            is_missing=status_raw is None,
        )

        # -------------------------------------------------------------
        # 6. LAYOUT & KEYWORDS (R_LAY_001 to R_LAY_003)
        # -------------------------------------------------------------
        all_text = " ".join([t.raw_text or "" for t in raw_tokens]).lower()
        gpay_kw_found = any(k in all_text for k in ["upi", "google", "paid", "to", "completed", "banking name", "from"])

        evidence["gpay_keywords_present"] = SemanticFeatureEvidence(
            feature_name="gpay_keywords_present",
            feature_value=gpay_kw_found,
            feature_status="available",
            source_ocr_text=None,
            source_bounding_box=None,
            rule_id="R_LAY_001",
            explanation=(
                "Standard Google Pay dark-theme receipt structural keywords localized"
                if gpay_kw_found
                else "Standard GPay keywords missing from receipt text stream"
            ),
            ocr_confidence=None,
            is_missing=False,
        )

        # Font ratio amount to payee (height ratio)
        font_ratio = 1.0
        if amt_field and rx_field and amt_field.pixel_box and rx_field.pixel_box:
            h_amt = amt_field.pixel_box.height
            h_rx = rx_field.pixel_box.height
            if h_rx > 0:
                font_ratio = round(h_amt / h_rx, 2)

        evidence["font_size_ratio_amount_to_payee"] = SemanticFeatureEvidence(
            feature_name="font_size_ratio_amount_to_payee",
            feature_value=font_ratio,
            feature_status="available" if (amt_field and rx_field) else "not_available",
            source_ocr_text=f"Amount H: {amt_field.pixel_box.height if amt_field and amt_field.pixel_box else 'N/A'}px / Payee H: {rx_field.pixel_box.height if rx_field and rx_field.pixel_box else 'N/A'}px",
            source_bounding_box=amt_box,
            rule_id="R_LAY_002",
            explanation=f"Measured vertical height ratio of amount header to payee line: {font_ratio}",
            ocr_confidence=amt_conf,
            is_missing=(amt_field is None or rx_field is None),
        )

        # Missing critical fields count
        critical_fields = ["amount", "receiver", "status", "transaction_id"]
        missing_count = sum(1 for cf in critical_fields if not detailed.get(cf) or not detailed[cf].is_detected)
        detected_count = sum(1 for cf in detailed if detailed[cf].is_detected)

        if missing_count >= 3:
            reason_codes.append(ReasonCodeEnum.MISSING_CRITICAL_PAYMENT_FIELD)

        evidence["missing_critical_fields_count"] = SemanticFeatureEvidence(
            feature_name="missing_critical_fields_count",
            feature_value=missing_count,
            feature_status="available",
            source_ocr_text=f"Missing: {missing_count} of 4 core fields",
            source_bounding_box=None,
            rule_id="R_LAY_003",
            explanation=f"Missing critical transaction fields count: {missing_count} / {len(critical_fields)}",
            ocr_confidence=None,
            is_missing=False,
        )

        # -------------------------------------------------------------
        # 7. OCR CONFIDENCE STATS
        # -------------------------------------------------------------
        confs = [t.confidence for t in raw_tokens if t.confidence is not None]
        avg_conf = round(float(sum(confs) / len(confs)), 4) if confs else 0.0
        min_conf = round(float(min(confs)), 4) if confs else 0.0

        crit_confs = [
            detailed[k].confidence
            for k in ["amount", "status", "transaction_id"]
            if k in detailed and detailed[k].confidence is not None
        ]
        crit_avg_conf = round(float(sum(crit_confs) / len(crit_confs)), 4) if crit_confs else avg_conf

        if avg_conf < 0.50 and raw_tokens:
            reason_codes.append(ReasonCodeEnum.OCR_CONFIDENCE_LOW)

        confidence_info = OCRConfidenceInfo(
            average_confidence=avg_conf,
            min_field_confidence=min_conf,
            critical_fields_confidence=crit_avg_conf,
            total_detected_tokens=len(raw_tokens),
        )

        # -------------------------------------------------------------
        # 8. CONSERVATIVE SCORE AGGREGATION (NO TAMPERING ON MISSING DATA)
        # -------------------------------------------------------------
        # Scoring applies strictly to actively detected anomalies/inconsistencies
        score = 0.0
        if anomalies.date_anomaly:
            score += 0.35
        if anomalies.upi_handle_anomaly:
            score += 0.35
        if anomalies.ref_id_anomaly:
            score += 0.30
        if anomalies.status_anomaly:
            score += 0.30
        if anomalies.amount_anomaly:
            score += 0.20

        # Cap between 0.0 and 1.0
        manipulation_score = min(1.0, max(0.0, round(score, 4)))

        # Clean verification code if authentic
        if not reason_codes and gpay_kw_found and detected_count >= 2:
            reason_codes.append(ReasonCodeEnum.INTEGRITY_VERIFIED_CLEAN)

        feature_vector = SemanticFeatureVector(
            amount_valid_format=amt_valid_format,
            amount_currency_symbol_valid=has_currency_symbol,
            amount_numeric_value_reasonable=amt_reasonable,
            ref_id_length_valid=ref_len_valid,
            ref_id_numeric_only=ref_num_only,
            status_text_authentic=status_authentic,
            upi_handle_valid_format=upi_valid,
            date_chronology_valid=date_valid,
            time_format_valid=time_valid,
            sender_receiver_distinct=distinct,
            gpay_keywords_present=gpay_kw_found,
            font_size_ratio_amount_to_payee=font_ratio,
            text_vertical_alignment_variance=0.0,
            missing_critical_fields_count=missing_count,
            total_fields_detected=detected_count,
            custom_numeric_features={
                "ocr_overall_confidence": avg_conf,
                "amount_extracted_flag": 1.0 if amt_present else 0.0,
                "upi_extracted_flag": 1.0 if rx_upi else 0.0,
                "ref_id_extracted_flag": 1.0 if ref_raw else 0.0,
            },
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return SemanticBranchOutput(
            semantic_manipulation_score=manipulation_score,
            feature_vector=feature_vector,
            reason_codes=reason_codes,
            field_anomalies=anomalies,
            ocr_confidence_info=confidence_info,
            processing_time_ms=round(elapsed_ms, 2),
            feature_evidence=evidence,
            semantic_status="live",
        )

    def _validate_calendar_date(self, date_str: Optional[str]) -> Tuple[bool, str, str]:
        """Validates calendar date legality (days per month, leap year)."""
        if not date_str or not str(date_str).strip():
            return True, "not_available", "Date text not localized in receipt"

        clean_d = str(date_str).strip()

        # Try standard pattern: DD Mon YYYY (e.g. 14 Oct 2024 or 31 Feb 2024)
        m = re.search(r"(\d{1,2})\s+([A-Za-z]{3,9})\s+(\d{4})", clean_d)
        if m:
            day = int(m.group(1))
            mon_str = m.group(2).lower()[:3]
            year = int(m.group(3))

            if mon_str not in self.MONTH_NAMES:
                return False, "invalid_format", f"Unrecognized month name in date: '{m.group(2)}'"

            month = self.MONTH_NAMES[mon_str]
            max_days = calendar.monthrange(year, month)[1]

            if day < 1 or day > max_days:
                return False, "invalid_format", f"Impossible calendar date: day {day} exceeds max {max_days} for {mon_str.title()} {year}"

            return True, "available", f"Calendar date verified: {day} {mon_str.title()} {year}"

        # Try ISO pattern: YYYY-MM-DD
        m_iso = re.search(r"(\d{4})-(\d{2})-(\d{2})", clean_d)
        if m_iso:
            year = int(m_iso.group(1))
            month = int(m_iso.group(2))
            day = int(m_iso.group(3))

            if month < 1 or month > 12:
                return False, "invalid_format", f"Invalid month in ISO date: {month}"

            max_days = calendar.monthrange(year, month)[1]
            if day < 1 or day > max_days:
                return False, "invalid_format", f"Impossible day {day} in ISO date {clean_d}"

            return True, "available", f"ISO calendar date verified: {clean_d}"

        return True, "ambiguous", f"Date localized with non-standard format: '{clean_d}'"

    def _validate_time_format(self, time_str: Optional[str]) -> Tuple[bool, str, str]:
        """Validates timestamp hour/minute legality."""
        if not time_str or not str(time_str).strip():
            return True, "not_available", "Time text not localized in receipt"

        clean_t = str(time_str).strip()

        # Match 12-hour format: HH:MM AM/PM
        m_12 = re.search(r"(\d{1,2})[:.](\d{2})\s*(AM|PM|am|pm)", clean_t)
        if m_12:
            hour = int(m_12.group(1))
            minute = int(m_12.group(2))
            if hour < 1 or hour > 12:
                return False, "invalid_format", f"Invalid hour {hour} in 12-hour time format"
            if minute < 0 or minute > 59:
                return False, "invalid_format", f"Invalid minute {minute} in timestamp"
            return True, "available", f"12-hour timestamp verified: {clean_t}"

        # Match 24-hour format: HH:MM
        m_24 = re.search(r"(\d{1,2})[:.](\d{2})", clean_t)
        if m_24:
            hour = int(m_24.group(1))
            minute = int(m_24.group(2))
            if hour < 0 or hour > 23:
                return False, "invalid_format", f"Invalid hour {hour} in 24-hour timestamp"
            if minute < 0 or minute > 59:
                return False, "invalid_format", f"Invalid minute {minute} in timestamp"
            return True, "available", f"24-hour timestamp verified: {clean_t}"

        return True, "ambiguous", f"Time localized with non-standard format: '{clean_t}'"


# Thread-safe Singleton Management
_SEMANTIC_SERVICE_INSTANCE: Optional[SemanticFeatureService] = None
_SEMANTIC_SERVICE_LOCK = threading.Lock()


def get_semantic_service() -> SemanticFeatureService:
    """Returns thread-safe singleton instance of SemanticFeatureService."""
    global _SEMANTIC_SERVICE_INSTANCE
    if _SEMANTIC_SERVICE_INSTANCE is None:
        with _SEMANTIC_SERVICE_LOCK:
            if _SEMANTIC_SERVICE_INSTANCE is None:
                _SEMANTIC_SERVICE_INSTANCE = SemanticFeatureService()
    return _SEMANTIC_SERVICE_INSTANCE


def reset_semantic_service() -> None:
    """Resets singleton instance for testing."""
    global _SEMANTIC_SERVICE_INSTANCE
    with _SEMANTIC_SERVICE_LOCK:
        _SEMANTIC_SERVICE_INSTANCE = None

