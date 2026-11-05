"""
OCR Extraction Service for CBL-04 Google Pay Screenshot Analysis.
Integrates EasyOCR to extract visible text tokens, spatial bounding boxes,
and token-level confidence scores from Google Pay dark-theme transaction receipts.
Implements conservative structured field extraction adhering to Phase 1/3 contracts.
"""

from __future__ import annotations

import io
import logging
import re
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image, UnidentifiedImageError

from app.core.config import get_settings
from app.schemas.common import NormalizedBoundingBox, PixelBoundingBox
from app.schemas.ocr import ExtractedTransactionData, OCRExtractedField, OCRResult
from app.services.image_validation_service import ValidatedImage

logger = logging.getLogger("cbl04.ocr")

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]


class OCRError(Exception):
    """Base exception for OCR service errors."""
    pass


class OCRInitializationError(OCRError):
    """Raised when the OCR engine or pre-trained weights fail to initialize."""
    pass


class OCRInferenceError(OCRError):
    """Raised when OCR text detection/recognition inference crashes."""
    pass


class OCRImageError(OCRError):
    """Raised when input image cannot be decoded, is corrupt, or has invalid dimensions."""
    pass


class OCRServiceInterface(ABC):
    """Abstract interface defining the OCR extraction contract for CBL-04."""

    @abstractmethod
    def extract(self, image: Union[bytes, Image.Image, ValidatedImage]) -> OCRResult:
        """
        Extracts visible text, bounding boxes, confidences, and structured metadata.

        Args:
            image: Raw bytes, PIL Image, or ValidatedImage object.

        Returns:
            OCRResult containing fields, detailed_fields, raw_detections, and confidence.
        """
        pass


class TransactionFieldExtractor:
    """
    Conservative rule-based parser for Google Pay dark-theme receipt OCR tokens.
    Extracts visible transaction fields while preserving provenance, original text,
    and confidence scores. Does NOT invent missing fields or decide tampering.
    """

    # Currency patterns and common OCR misrecognitions of the Rupee symbol (₹)
    AMOUNT_PREFIX_REGEX = re.compile(r"^(?:[₹\{\[\(\?\}\]\)\$£€§=]|Ft|Rs\.?|INR)\s*", re.IGNORECASE)
    AMOUNT_VALUE_REGEX = re.compile(
        r"^(?:[₹\{\[\(\?\}\]\)\$£€§=]|Ft|Rs\.?|INR)?\s*([0-9]{1,3}(?:,[0-9]{2,3})*(?:\.[0-9]{1,2})?|[0-9]+(?:\.[0-9]{1,2})?)$",
        re.IGNORECASE,
    )
    AMOUNT_STANDALONE_REGEX = re.compile(
        r"(?:[₹\{\[\(\?\}\]\)\$£€§=]|Ft|Rs\.?|INR)?\s*([0-9]{1,3}(?:,[0-9]{2,3})*(?:\.[0-9]{1,2})?|[0-9]+(?:\.[0-9]{1,2})?)",
        re.IGNORECASE,
    )

    UPI_VPA_REGEX = re.compile(r"([a-zA-Z0-9.\-_]+@[a-zA-Z0-9]+)", re.IGNORECASE)

    DATE_PATTERNS = [
        re.compile(r"\b(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4})\b", re.IGNORECASE),
        re.compile(r"\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4})\b", re.IGNORECASE),
        re.compile(r"\b(\d{4}-\d{2}-\d{2})\b"),
    ]

    TIME_PATTERNS = [
        re.compile(r"\b(\d{1,2}[:.]\d{2}(?:[:.]\d{2})?\s*(?:am|pm|AM|PM)?)\b"),
    ]

    STATUS_KEYWORDS = [
        "completed",
        "payment completed",
        "paid",
        "payment to",
        "payment started",
        "processing",
        "transferred",
        "success",
        "successful",
        "failed",
        "pending",
    ]

    @classmethod
    def extract_fields(
        cls,
        tokens: List[OCRExtractedField],
        image_width: int,
        image_height: int,
    ) -> Tuple[ExtractedTransactionData, Dict[str, OCRExtractedField]]:
        """
        Parses token stream into structured ExtractedTransactionData and detailed_fields.
        """
        detailed: Dict[str, OCRExtractedField] = {}
        fields_data: Dict[str, Any] = {
            "transaction_id": None,
            "amount": None,
            "amount_numeric": None,
            "currency": "INR",
            "date": None,
            "time": None,
            "sender": None,
            "sender_upi": None,
            "receiver": None,
            "receiver_upi": None,
            "status": None,
            "google_txn_id": None,
        }

        if not tokens:
            for k in fields_data:
                if k != "currency":
                    detailed[k] = cls._empty_field(k)
            return ExtractedTransactionData(**fields_data), detailed

        # 1. Status Extraction
        status_field = cls._extract_status(tokens)
        if status_field.is_detected:
            fields_data["status"] = status_field.normalized_value
            detailed["status"] = status_field
        else:
            detailed["status"] = cls._empty_field("status")

        # 2. Amount Extraction
        amount_field, num_val = cls._extract_amount(tokens, image_height)
        if amount_field.is_detected:
            fields_data["amount"] = f"₹{num_val:g}" if num_val is not None else amount_field.raw_text
            fields_data["amount_numeric"] = num_val
            detailed["amount"] = amount_field
        else:
            detailed["amount"] = cls._empty_field("amount")

        # 3. Recipient and Receiver UPI
        receiver_field, rx_upi_field = cls._extract_receiver(tokens)
        if receiver_field.is_detected:
            fields_data["receiver"] = receiver_field.normalized_value
            detailed["receiver"] = receiver_field
        else:
            detailed["receiver"] = cls._empty_field("receiver")

        if rx_upi_field.is_detected:
            fields_data["receiver_upi"] = rx_upi_field.normalized_value
            detailed["receiver_upi"] = rx_upi_field
        else:
            detailed["receiver_upi"] = cls._empty_field("receiver_upi")

        # 4. Sender and Sender UPI
        sender_field, tx_upi_field = cls._extract_sender(tokens)
        if sender_field.is_detected:
            fields_data["sender"] = sender_field.normalized_value
            detailed["sender"] = sender_field
        else:
            detailed["sender"] = cls._empty_field("sender")

        if tx_upi_field.is_detected:
            fields_data["sender_upi"] = tx_upi_field.normalized_value
            detailed["sender_upi"] = tx_upi_field
        else:
            detailed["sender_upi"] = cls._empty_field("sender_upi")

        # 5. Date and Time Extraction
        date_field, time_field = cls._extract_date_time(tokens)
        if date_field.is_detected:
            fields_data["date"] = date_field.normalized_value
            detailed["date"] = date_field
        else:
            detailed["date"] = cls._empty_field("date")

        if time_field.is_detected:
            fields_data["time"] = time_field.normalized_value
            detailed["time"] = time_field
        else:
            detailed["time"] = cls._empty_field("time")

        # 6. UPI Transaction ID / Reference ID
        upi_id_field = cls._extract_upi_transaction_id(tokens)
        if upi_id_field.is_detected:
            fields_data["transaction_id"] = upi_id_field.normalized_value
            detailed["transaction_id"] = upi_id_field
        else:
            detailed["transaction_id"] = cls._empty_field("transaction_id")

        # 7. Google Transaction ID
        gpay_id_field = cls._extract_google_transaction_id(tokens)
        if gpay_id_field.is_detected:
            fields_data["google_txn_id"] = gpay_id_field.normalized_value
            detailed["google_txn_id"] = gpay_id_field
        else:
            detailed["google_txn_id"] = cls._empty_field("google_txn_id")

        return ExtractedTransactionData(**fields_data), detailed

    @classmethod
    def _empty_field(cls, field_name: str) -> OCRExtractedField:
        return OCRExtractedField(
            field_name=field_name,
            raw_text=None,
            confidence=None,
            bounding_box=None,
            pixel_box=None,
            normalized_value=None,
            is_detected=False,
            extraction_status="not_found",
        )

    @classmethod
    def _extract_status(cls, tokens: List[OCRExtractedField]) -> OCRExtractedField:
        for t in tokens:
            txt_lower = (t.raw_text or "").strip().lower()
            for kw in cls.STATUS_KEYWORDS:
                if kw in txt_lower:
                    normalized = kw.title()
                    if kw in ["completed", "payment completed"]:
                        normalized = "Completed"
                    elif kw in ["paid", "payment to"]:
                        normalized = "Paid"
                    return OCRExtractedField(
                        field_name="status",
                        raw_text=t.raw_text,
                        confidence=t.confidence,
                        bounding_box=t.bounding_box,
                        pixel_box=t.pixel_box,
                        normalized_value=normalized,
                        is_detected=True,
                        extraction_status="extracted",
                    )
        return cls._empty_field("status")

    @classmethod
    def _extract_amount(
        cls,
        tokens: List[OCRExtractedField],
        image_height: int,
    ) -> Tuple[OCRExtractedField, Optional[float]]:
        # Dedicated currency symbols that may appear as separate tokens
        CURRENCY_SYMBOLS = {
            "₹", "{", "}", "[", "]", "(", ")", "?", "ft", "rs", "rs.", "inr", "inr.",
            "§", "$", "£", "€", "re", "re.", "="
        }
        currency_tokens: List[OCRExtractedField] = []
        for t in tokens:
            raw_t = (t.raw_text or "").strip().lower()
            if raw_t in CURRENCY_SYMBOLS:
                currency_tokens.append(t)

        candidates: List[Dict[str, Any]] = []
        for t in tokens:
            txt = (t.raw_text or "").strip()
            if not txt:
                continue

            txt_lower = txt.lower()

            # 1. Date/time exclusion
            if any(m in txt_lower for m in ["am", "pm", "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]):
                continue

            # 2. UPI/Payment/Bank keyword exclusion
            if txt.startswith("+91") or "+91" in txt or "@" in txt:
                continue
            if any(kw in txt_lower for kw in ["gpay", "google pay", "upi", "bank", "account", "a/c", "acct", "card", "ending", "ref:", "id:", "rrn", "order", "txn"]):
                continue

            # 3. Spatial & quote filters (phone number fragment exclusion)
            ymin = t.bounding_box.ymin if t.bounding_box else 0.5
            box_height = t.pixel_box.height if t.pixel_box else 10.0

            if any(q in txt for q in ["'", '"', "xxx", "***", "..."]):
                continue
            if ymin < 0.08 and any(c.isdigit() for c in txt):
                continue

            # 4. Long numeric strings (transaction IDs / phone numbers) without explicit currency symbol
            digits_only = re.sub(r"\D", "", txt)
            if len(digits_only) >= 10:
                continue

            # 5. Small secondary text outside hero region
            if ymin > 0.40 and box_height < 35.0:
                continue

            # 6. Check for explicit currency symbol prefix
            has_currency_symbol = bool(cls.AMOUNT_PREFIX_REGEX.match(txt))
            adjacent_symbol = False

            # Check if an explicit currency symbol token is positioned immediately to the left
            if not has_currency_symbol and t.bounding_box:
                t_ymin = t.bounding_box.ymin
                t_ymax = t.bounding_box.ymax
                t_xmin = t.bounding_box.xmin
                for ct in currency_tokens:
                    if ct.bounding_box:
                        c_ymin = ct.bounding_box.ymin
                        c_ymax = ct.bounding_box.ymax
                        c_xmax = ct.bounding_box.xmax
                        # Same vertical band and immediately adjacent to the left
                        vert_overlap = min(t_ymax, c_ymax) - max(t_ymin, c_ymin)
                        horiz_gap = t_xmin - c_xmax
                        if vert_overlap > 0 and 0.0 <= horiz_gap < 0.12:
                            has_currency_symbol = True
                            adjacent_symbol = True
                            break

            # 7. Extract numeric portion
            m = cls.AMOUNT_VALUE_REGEX.match(txt)
            if not m:
                m = cls.AMOUNT_STANDALONE_REGEX.search(txt)

            if m:
                num_str = m.group(1).replace(",", "")
                try:
                    num_val = float(num_str)
                    if num_val <= 0.0 or num_val > 1000000.0:
                        continue

                    # Positional weighting: GPay primary transaction amount is located in the upper-middle hero area
                    if 0.17 <= ymin <= 0.32:
                        pos_score = 3.0
                    elif 0.12 <= ymin <= 0.42:
                        pos_score = 1.5
                    else:
                        pos_score = 0.3

                    # Visual prominence bonus: Primary amount text is rendered in significantly larger font
                    height_score = (box_height / 25.0) ** 1.8
                    symbol_score = 2.0 if has_currency_symbol else 1.0
                    conf = t.confidence or 0.5

                    # Slight bonus for decimal formatting (.00 or .50)
                    decimal_score = 1.1 if ("." in num_str and len(num_str.split(".")[1]) == 2) else 1.0

                    score = symbol_score * pos_score * height_score * conf * decimal_score

                    cand_info = {
                        "score": score,
                        "token": t,
                        "raw_text": txt,
                        "num_val": num_val,
                        "digits_only": digits_only,
                        "has_currency_symbol": has_currency_symbol,
                        "adjacent_symbol": adjacent_symbol,
                        "confidence": conf,
                        "bounding_box": t.bounding_box,
                        "pixel_box": t.pixel_box,
                    }
                    candidates.append(cand_info)
                except ValueError:
                    continue

        if not candidates:
            # Secondary fallback: search for confirmation line e.g. 'Payment of =90 completed'
            for t in tokens:
                txt = (t.raw_text or "").strip()
                m_conf = re.search(r"Payment of\s*(?:[₹\{\[\(\?\}\]\)\$£€§=]|Ft|Rs\.?|INR)?\s*([0-9]{1,3}(?:,[0-9]{2,3})*(?:\.[0-9]{1,2})?|[0-9]+(?:\.[0-9]{1,2})?)", txt, re.IGNORECASE)
                if m_conf:
                    try:
                        n_str = m_conf.group(1).replace(",", "")
                        n_val = float(n_str)
                        if 0.0 < n_val <= 1000000.0:
                            candidates.append({
                                "score": 2.0,
                                "token": t,
                                "raw_text": txt,
                                "num_val": n_val,
                                "digits_only": re.sub(r"\D", "", n_str),
                                "has_currency_symbol": True,
                                "adjacent_symbol": False,
                                "confidence": t.confidence or 0.7,
                                "bounding_box": t.bounding_box,
                                "pixel_box": t.pixel_box,
                            })
                            break
                    except ValueError:
                        continue

        if not candidates:
            return cls._empty_field("amount"), None

        # Sort candidates by composite prominence score
        candidates.sort(key=lambda c: c["score"], reverse=True)
        best = candidates[0]
        t = best["token"]
        num_val = best["num_val"]
        has_sym = best["has_currency_symbol"]

        # Cross-validate with confirmation line if available (e.g. "Payment of ₹127.94 completed")
        conf_amount = None
        for t_item in tokens:
            t_txt = (t_item.raw_text or "").strip()
            m_conf = re.search(
                r"Payment of\s*(?:[₹\{\[\(\?\}\]\)\$£€§=]|Ft|Rs\.?|INR|\?)*\s*([0-9]{1,3}(?:,[0-9]{2,3})*(?:\.[0-9]{1,2})?|[0-9]+(?:\.[0-9]{1,2})?)",
                t_txt,
                re.IGNORECASE,
            )
            if m_conf:
                try:
                    conf_amount = float(m_conf.group(1).replace(",", ""))
                    break
                except ValueError:
                    pass

        # If confirmation amount is detected, check if hero candidate has a leading misrecognized Rupee glyph (2, 3, 7, 8, ?, {, =)
        if conf_amount is not None and conf_amount > 0:
            str_best = f"{num_val:.2f}" if "." in str(num_val) else f"{num_val:g}"
            str_conf = f"{conf_amount:.2f}" if "." in str(conf_amount) else f"{conf_amount:g}"
            if str_best != str_conf and (str_best.endswith(str_conf) or str_best[1:] == str_conf):
                num_val = conf_amount
                has_sym = True
            elif not has_sym:
                num_val = conf_amount
                has_sym = True
        elif not has_sym and len(best["digits_only"]) >= 4 and best["digits_only"][0] in ["2", "3", "7", "8"] and "." in best["raw_text"]:
            # If hero token has no currency symbol, no comma, and starts with 2/3/7/8 followed by realistic decimal amount
            try:
                sub_val = float(best["digits_only"][1:])
                if 0.0 < sub_val <= 100000.0:
                    num_val = sub_val
                    has_sym = True
            except ValueError:
                pass

        # Ambiguity determination:
        # 1. Absence of confirmed currency symbol prefix (on token or adjacent)
        # 2. Multiple competing candidates with close composite scores (< 25% margin)
        # 3. Very low OCR token confidence (< 0.40)
        close_competitor = (
            len(candidates) > 1 and abs(candidates[0]["score"] - candidates[1]["score"]) < 0.25 * candidates[0]["score"]
        )
        low_token_conf = (best["confidence"] < 0.40)

        is_ambiguous = (not has_sym) or close_competitor or low_token_conf

        # Candidate records for telemetry & auditing
        candidate_records: List[Dict[str, Any]] = []
        for c in candidates[:5]:
            c_rec: Dict[str, Any] = {
                "raw_text": c["raw_text"],
                "numeric_value": c["num_val"],
                "has_currency_symbol": c["has_currency_symbol"],
                "score": round(c["score"], 4),
                "confidence": round(c["confidence"], 4),
                "bounding_box": c["bounding_box"].model_dump() if c["bounding_box"] else None,
                "is_primary": (c == best),
            }
            # Record hypothesized rupee misrecognition if pure digits start with 2, 3, 7, 8
            if not c["has_currency_symbol"] and len(c["digits_only"]) >= 3 and c["digits_only"][0] in ["2", "3", "7", "8"]:
                try:
                    c_rec["hypothesized_amount_without_prefix_digit"] = float(c["digits_only"][1:])
                    c_rec["hypothesized_prefix_glyph"] = c["digits_only"][0]
                except ValueError:
                    pass
            candidate_records.append(c_rec)

        field_conf = round(best["confidence"] * 0.70, 4) if is_ambiguous else round(best["confidence"], 4)
        display_raw = t.raw_text
        if num_val is not None and (not display_raw or not str(display_raw).startswith("₹")):
            display_raw = f"₹{num_val:g}"

        return (
            OCRExtractedField(
                field_name="amount",
                raw_text=display_raw,
                confidence=field_conf,
                bounding_box=t.bounding_box,
                pixel_box=t.pixel_box,
                normalized_value=num_val,
                is_detected=True,
                extraction_status="ambiguous" if is_ambiguous else "extracted",
                candidates=candidate_records,
            ),
            num_val,
        )

    @classmethod
    def _extract_receiver(
        cls, tokens: List[OCRExtractedField]
    ) -> Tuple[OCRExtractedField, OCRExtractedField]:
        receiver_field = cls._empty_field("receiver")
        rx_upi_field = cls._empty_field("receiver_upi")
        rx_idx = None

        # Find receiver name first
        for i, t in enumerate(tokens):
            txt = (t.raw_text or "").strip()
            if re.match(r"^(?:To:|To\s+|Paid to\s+|Payment to\s+)", txt, re.IGNORECASE):
                cleaned_name = re.sub(r"^(?:To:|To\s+|Paid to\s+|Payment to\s+)", "", txt, flags=re.IGNORECASE).strip()
                if cleaned_name and not cls.UPI_VPA_REGEX.match(cleaned_name):
                    receiver_field = OCRExtractedField(
                        field_name="receiver",
                        raw_text=t.raw_text,
                        confidence=t.confidence,
                        bounding_box=t.bounding_box,
                        pixel_box=t.pixel_box,
                        normalized_value=cleaned_name,
                        is_detected=True,
                        extraction_status="extracted",
                    )
                    rx_idx = i
                    break
            elif txt.lower() in ["to:", "to", "paid to"] and i + 1 < len(tokens):
                next_t = tokens[i + 1]
                next_txt = (next_t.raw_text or "").strip()
                if next_txt and not cls.UPI_VPA_REGEX.match(next_txt):
                    receiver_field = OCRExtractedField(
                        field_name="receiver",
                        raw_text=next_t.raw_text,
                        confidence=next_t.confidence,
                        bounding_box=next_t.bounding_box,
                        pixel_box=next_t.pixel_box,
                        normalized_value=next_txt,
                        is_detected=True,
                        extraction_status="extracted",
                    )
                    rx_idx = i + 1
                    break

        # Identify boundary of the sender ("From:") section to avoid claiming sender VPA
        from_idx = len(tokens)
        for i, t in enumerate(tokens):
            txt = (t.raw_text or "").strip().lower()
            if txt.startswith("from:") or txt.startswith("from ") or "banking name:" in txt:
                from_idx = i
                break

        # Search for receiver UPI strictly before from_idx
        for i in range(from_idx):
            t = tokens[i]
            txt = (t.raw_text or "").strip()
            m_vpa = cls.UPI_VPA_REGEX.search(txt)
            if m_vpa:
                rx_upi_field = OCRExtractedField(
                    field_name="receiver_upi",
                    raw_text=t.raw_text,
                    confidence=t.confidence,
                    bounding_box=t.bounding_box,
                    pixel_box=t.pixel_box,
                    normalized_value=m_vpa.group(1),
                    is_detected=True,
                    extraction_status="extracted",
                )
                break

        return receiver_field, rx_upi_field

    @classmethod
    def _extract_sender(
        cls, tokens: List[OCRExtractedField]
    ) -> Tuple[OCRExtractedField, OCRExtractedField]:
        sender_field = cls._empty_field("sender")
        tx_upi_field = cls._empty_field("sender_upi")
        from_idx = None

        for i, t in enumerate(tokens):
            txt = (t.raw_text or "").strip()
            if re.match(r"^(?:From:|From\s+|Banking Name:\s*)", txt, re.IGNORECASE):
                cleaned_name = re.sub(r"^(?:From:|From\s+|Banking Name:\s*)", "", txt, flags=re.IGNORECASE).strip()
                if cleaned_name and not cls.UPI_VPA_REGEX.match(cleaned_name):
                    sender_field = OCRExtractedField(
                        field_name="sender",
                        raw_text=t.raw_text,
                        confidence=t.confidence,
                        bounding_box=t.bounding_box,
                        pixel_box=t.pixel_box,
                        normalized_value=cleaned_name,
                        is_detected=True,
                        extraction_status="extracted",
                    )
                    from_idx = i
                    break
            elif txt.lower() in ["from:", "from"] and i + 1 < len(tokens):
                next_t = tokens[i + 1]
                next_txt = (next_t.raw_text or "").strip()
                if next_txt and not cls.UPI_VPA_REGEX.match(next_txt):
                    sender_field = OCRExtractedField(
                        field_name="sender",
                        raw_text=next_t.raw_text,
                        confidence=next_t.confidence,
                        bounding_box=next_t.bounding_box,
                        pixel_box=next_t.pixel_box,
                        normalized_value=next_txt,
                        is_detected=True,
                        extraction_status="extracted",
                    )
                    from_idx = i + 1
                    break

        # Search for sender UPI at or after from_idx
        start = from_idx if from_idx is not None else 0
        for i in range(start, len(tokens)):
            t = tokens[i]
            txt = (t.raw_text or "").strip()
            m_vpa = cls.UPI_VPA_REGEX.search(txt)
            if m_vpa:
                tx_upi_field = OCRExtractedField(
                    field_name="sender_upi",
                    raw_text=t.raw_text,
                    confidence=t.confidence,
                    bounding_box=t.bounding_box,
                    pixel_box=t.pixel_box,
                    normalized_value=m_vpa.group(1),
                    is_detected=True,
                    extraction_status="extracted",
                )
                break

        return sender_field, tx_upi_field

    @classmethod
    def _extract_date_time(
        cls, tokens: List[OCRExtractedField]
    ) -> Tuple[OCRExtractedField, OCRExtractedField]:
        date_field = cls._empty_field("date")
        time_field = cls._empty_field("time")

        for t in tokens:
            txt = (t.raw_text or "").strip()
            # Try date regex
            if not date_field.is_detected:
                for d_pat in cls.DATE_PATTERNS:
                    m_date = d_pat.search(txt)
                    if m_date:
                        date_field = OCRExtractedField(
                            field_name="date",
                            raw_text=t.raw_text,
                            confidence=t.confidence,
                            bounding_box=t.bounding_box,
                            pixel_box=t.pixel_box,
                            normalized_value=m_date.group(1).strip(),
                            is_detected=True,
                            extraction_status="extracted",
                        )
                        break

            # Try time regex
            if not time_field.is_detected:
                for t_pat in cls.TIME_PATTERNS:
                    m_time = t_pat.search(txt)
                    if m_time:
                        time_str = m_time.group(1).strip()
                        # Avoid matching lone numbers
                        if any(c in time_str for c in [":", "."]) or any(c in time_str.lower() for c in ["am", "pm"]):
                            time_field = OCRExtractedField(
                                field_name="time",
                                raw_text=t.raw_text,
                                confidence=t.confidence,
                                bounding_box=t.bounding_box,
                                pixel_box=t.pixel_box,
                                normalized_value=time_str,
                                is_detected=True,
                                extraction_status="extracted",
                            )
                            break

        return date_field, time_field

    UPI_TXN_REGEX = re.compile(r"(?:DEMO[\s\-_]*UPI[\s\-_]*[0-9]+|UPI/[0-9]+|\b[0-9]{12}\b)", re.IGNORECASE)
    GPAY_TXN_REGEX = re.compile(r"(?:DEMO[\s\-_]*GPAY[\s\-_]*[A-Za-z0-9\-]+|CIC[0-9]+)", re.IGNORECASE)

    @classmethod
    def _extract_upi_transaction_id(cls, tokens: List[OCRExtractedField]) -> OCRExtractedField:
        upi_label_idx = None
        for i, t in enumerate(tokens):
            txt = (t.raw_text or "").strip().lower()
            if "upi transaction id" in txt or "upi ref" in txt or "utr" in txt:
                upi_label_idx = i
                break

        if upi_label_idx is not None:
            # Check current token first
            curr_txt = tokens[upi_label_idx].raw_text or ""
            m = cls.UPI_TXN_REGEX.search(curr_txt)
            if m:
                t = tokens[upi_label_idx]
                return OCRExtractedField(
                    field_name="transaction_id",
                    raw_text=t.raw_text,
                    confidence=t.confidence,
                    bounding_box=t.bounding_box,
                    pixel_box=t.pixel_box,
                    normalized_value=m.group(0),
                    is_detected=True,
                    extraction_status="extracted",
                )
            # Check next 2 tokens
            for offset in [1, 2]:
                if upi_label_idx + offset < len(tokens):
                    next_t = tokens[upi_label_idx + offset]
                    next_txt = next_t.raw_text or ""
                    m2 = cls.UPI_TXN_REGEX.search(next_txt)
                    if m2:
                        return OCRExtractedField(
                            field_name="transaction_id",
                            raw_text=next_t.raw_text,
                            confidence=next_t.confidence,
                            bounding_box=next_t.bounding_box,
                            pixel_box=next_t.pixel_box,
                            normalized_value=m2.group(0),
                            is_detected=True,
                            extraction_status="extracted",
                        )

        # Fallback: scan for DEMO-UPI or 12-digit RRN pattern across all tokens
        for t in tokens:
            txt = t.raw_text or ""
            m = cls.UPI_TXN_REGEX.search(txt)
            if m:
                return OCRExtractedField(
                    field_name="transaction_id",
                    raw_text=t.raw_text,
                    confidence=t.confidence,
                    bounding_box=t.bounding_box,
                    pixel_box=t.pixel_box,
                    normalized_value=m.group(0),
                    is_detected=True,
                    extraction_status="extracted",
                )

        return cls._empty_field("transaction_id")

    @classmethod
    def _extract_google_transaction_id(cls, tokens: List[OCRExtractedField]) -> OCRExtractedField:
        gpay_label_idx = None
        for i, t in enumerate(tokens):
            txt = (t.raw_text or "").strip().lower()
            if "google transaction id" in txt:
                gpay_label_idx = i
                break

        if gpay_label_idx is not None:
            # Check current or next tokens
            for offset in [0, 1, 2]:
                if gpay_label_idx + offset < len(tokens):
                    t = tokens[gpay_label_idx + offset]
                    txt = t.raw_text or ""
                    m = cls.GPAY_TXN_REGEX.search(txt)
                    if m:
                        return OCRExtractedField(
                            field_name="google_txn_id",
                            raw_text=t.raw_text,
                            confidence=t.confidence,
                            bounding_box=t.bounding_box,
                            pixel_box=t.pixel_box,
                            normalized_value=m.group(0),
                            is_detected=True,
                            extraction_status="extracted",
                        )

        for t in tokens:
            txt = t.raw_text or ""
            m = cls.GPAY_TXN_REGEX.search(txt)
            if m:
                return OCRExtractedField(
                    field_name="google_txn_id",
                    raw_text=t.raw_text,
                    confidence=t.confidence,
                    bounding_box=t.bounding_box,
                    pixel_box=t.pixel_box,
                    normalized_value=m.group(0),
                    is_detected=True,
                    extraction_status="extracted",
                )

        return cls._empty_field("google_txn_id")


class EasyOCRService(OCRServiceInterface):
    """
    Production EasyOCR inference service wrapping PyTorch CRAFT detection and CRNN recognition.
    Enforces thread-safe singleton lifecycle and offline local model weight loading.
    """

    def __init__(
        self,
        model_storage_dir: Optional[Union[str, Path]] = None,
        gpu: Optional[bool] = None,
        auto_load: bool = True,
    ) -> None:
        settings = get_settings()
        raw_dir = model_storage_dir or settings.OCR_MODEL_DIR
        self.model_storage_dir = self._resolve_path(raw_dir)
        self.gpu = gpu if gpu is not None else settings.OCR_GPU
        self.languages = settings.OCR_LANGUAGES

        self.reader: Optional[Any] = None
        self.is_loaded: bool = False
        self.ocr_engine_name: str = "EasyOCR"
        self.ocr_version_str: str = "1.7.2"

        if auto_load:
            self.load()

    def _resolve_path(self, path: Union[str, Path]) -> Path:
        p = Path(path)
        if not p.is_absolute():
            p = (WORKSPACE_ROOT / p).resolve()
        return p

    def load(self) -> None:
        """Initializes EasyOCR reader with local model weights."""
        if self.is_loaded and self.reader is not None:
            return

        detection_model = self.model_storage_dir / "craft_mlt_25k.pth"
        recognition_model = self.model_storage_dir / "english_g2.pth"

        if not detection_model.exists():
            raise OCRInitializationError(
                f"EasyOCR detection model weights not found at: {detection_model}. "
                "Models must be pre-provisioned in models/easyocr/."
            )
        if not recognition_model.exists():
            raise OCRInitializationError(
                f"EasyOCR recognition model weights not found at: {recognition_model}. "
                "Models must be pre-provisioned in models/easyocr/."
            )

        try:
            import easyocr
            self.ocr_version_str = getattr(easyocr, "__version__", "1.7.2")
            logger.info(
                "Initializing EasyOCR reader (languages=%s, gpu=%s, model_dir=%s)",
                self.languages,
                self.gpu,
                self.model_storage_dir,
            )
            self.reader = easyocr.Reader(
                self.languages,
                gpu=self.gpu,
                model_storage_directory=str(self.model_storage_dir),
                user_network_directory=str(self.model_storage_dir),
                download_enabled=False,
            )
            self.is_loaded = True
            logger.info("EasyOCR reader initialized successfully.")
        except Exception as exc:
            self.is_loaded = False
            self.reader = None
            raise OCRInitializationError(f"Failed to initialize EasyOCR reader: {exc}") from exc

    def extract(self, image: Union[bytes, Image.Image, ValidatedImage]) -> OCRResult:
        """
        Extracts visible text, bounding boxes, confidences, and structured metadata.
        """
        if not self.is_loaded or self.reader is None:
            raise OCRInitializationError("OCR service is not initialized or model weights are missing.")

        start_time = time.perf_counter()

        # 1. Decode & validate image
        pil_image = self._coerce_pil_image(image)
        width, height = pil_image.size
        if width <= 0 or height <= 0:
            raise OCRImageError(f"Invalid image dimensions: ({width}, {height}).")

        # Ensure RGB conversion
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")

        img_np = np.array(pil_image)

        # 2. Run EasyOCR inference
        try:
            detections = self.reader.readtext(img_np)
        except Exception as exc:
            raise OCRInferenceError(f"EasyOCR inference failure: {exc}") from exc

        # 3. Build token detections
        raw_detections: List[OCRExtractedField] = []
        token_confidences: List[float] = []
        raw_text_tokens: List[str] = []

        for i, (poly, text, conf) in enumerate(detections):
            raw_str = str(text).strip()
            if not raw_str:
                continue

            raw_text_tokens.append(raw_str)
            token_conf = float(conf) if conf is not None else 0.0
            token_conf = max(0.0, min(1.0, token_conf))
            token_confidences.append(token_conf)

            # Spatial coordinates
            xs = [float(p[0]) for p in poly]
            ys = [float(p[1]) for p in poly]
            xmin = max(0.0, min(xs))
            xmax = min(float(width), max(xs))
            ymin = max(0.0, min(ys))
            ymax = min(float(height), max(ys))

            # Ensure non-negative bounds
            box_w = max(0.0, xmax - xmin)
            box_h = max(0.0, ymax - ymin)

            pixel_box = PixelBoundingBox(
                x=round(xmin, 2),
                y=round(ymin, 2),
                width=round(box_w, 2),
                height=round(box_h, 2),
            )

            # Guard against zero-division (handled by width/height check above)
            norm_box = NormalizedBoundingBox(
                xmin=round(max(0.0, min(1.0, xmin / width)), 4),
                ymin=round(max(0.0, min(1.0, ymin / height)), 4),
                xmax=round(max(0.0, min(1.0, xmax / width)), 4),
                ymax=round(max(0.0, min(1.0, ymax / height)), 4),
            )

            raw_detections.append(
                OCRExtractedField(
                    field_name=f"token_{i}",
                    raw_text=raw_str,
                    confidence=round(token_conf, 4),
                    bounding_box=norm_box,
                    pixel_box=pixel_box,
                    normalized_value=raw_str,
                    is_detected=True,
                    extraction_status="extracted",
                )
            )

        # 4. Structured Field Extraction
        fields, detailed_fields = TransactionFieldExtractor.extract_fields(
            raw_detections, image_width=width, image_height=height
        )

        # 5. Aggregate metrics
        overall_conf = (
            round(sum(token_confidences) / len(token_confidences), 4)
            if token_confidences
            else 0.0
        )
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        logger.info(
            "OCR extraction completed in %.2f ms (tokens=%d, overall_conf=%.4f)",
            elapsed_ms,
            len(raw_detections),
            overall_conf,
        )

        return OCRResult(
            fields=fields,
            detailed_fields=detailed_fields,
            raw_detections=raw_detections,
            overall_confidence=overall_conf,
            ocr_engine=self.ocr_engine_name,
            ocr_version=self.ocr_version_str,
            raw_text=" ".join(raw_text_tokens),
            processing_time_ms=round(elapsed_ms, 2),
        )

    def _coerce_pil_image(self, image: Union[bytes, bytearray, Image.Image, ValidatedImage]) -> Image.Image:
        """Converts input to PIL Image safely."""
        if isinstance(image, Image.Image):
            return image
        elif isinstance(image, (bytes, bytearray)):
            try:
                buf = io.BytesIO(image)
                img = Image.open(buf)
                img.load()  # Force load image bytes
                return img
            except UnidentifiedImageError as err:
                raise OCRImageError(f"Unsupported or unrecognizable image format: {err}") from err
            except Exception as err:
                raise OCRImageError(f"Corrupt or invalid image bytes: {err}") from err
        elif isinstance(image, ValidatedImage):
            raise OCRImageError(
                "ValidatedImage metadata does not contain raw pixel bytes; "
                "pass image bytes or PIL.Image directly to extract()."
            )
        else:
            raise OCRImageError(f"Unsupported image type: {type(image)}")


# Thread-safe Singleton Management
_OCR_SERVICE_INSTANCE: Optional[EasyOCRService] = None
_OCR_SERVICE_LOCK = threading.Lock()


def get_ocr_service() -> EasyOCRService:
    """Returns thread-safe singleton instance of EasyOCRService."""
    global _OCR_SERVICE_INSTANCE
    if _OCR_SERVICE_INSTANCE is None:
        with _OCR_SERVICE_LOCK:
            if _OCR_SERVICE_INSTANCE is None:
                _OCR_SERVICE_INSTANCE = EasyOCRService()
    return _OCR_SERVICE_INSTANCE


def reset_ocr_service() -> None:
    """Resets singleton instance for testing and error injection."""
    global _OCR_SERVICE_INSTANCE
    with _OCR_SERVICE_LOCK:
        _OCR_SERVICE_INSTANCE = None
