"""
Phase 19: Character-Level Geometry Feature Extraction Module.
Extracts continuous character and layout geometric consistency features strictly from
inference-available OCR detections and raw pixel bounding boxes:
  1. Character / Word Box Height Standard Deviation (char_height_std)
  2. Character / Word Aspect Ratio Variance (char_aspect_variance)
  3. Text Baseline Deviation Residual Variance (baseline_deviation_variance)
  4. Inter-Word Horizontal Gap Variance (inter_word_gap_variance)
  5. Amount to Payee Font Height Ratio (font_ratio_amount_payee)
  6. Amount to Transaction ID Font Height Ratio (font_ratio_amount_txnid)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class GeometryFeatureVector:
    sample_id: str
    char_height_mean: float
    char_height_std: float
    char_aspect_variance: float
    baseline_deviation_variance: float
    inter_word_gap_variance: float
    font_ratio_amount_payee: float
    font_ratio_amount_txnid: float
    geometry_anomaly_score: float
    is_available: bool


class CharacterGeometryExtractor:
    """
    Extracts geometric consistency indicators from inference-available OCR word boxes.
    Does NOT use ground-truth annotations or character segmentations.
    """

    def extract_geometry_features(
        self,
        sample_id: str,
        image_dimensions: Tuple[int, int],
        raw_tokens: List[Dict[str, Any]],
        detailed_fields: Dict[str, Any],
    ) -> GeometryFeatureVector:
        w_img, h_img = image_dimensions

        if not raw_tokens or len(raw_tokens) < 3:
            return GeometryFeatureVector(
                sample_id=sample_id,
                char_height_mean=0.0,
                char_height_std=0.0,
                char_aspect_variance=0.0,
                baseline_deviation_variance=0.0,
                inter_word_gap_variance=0.0,
                font_ratio_amount_payee=1.0,
                font_ratio_amount_txnid=1.0,
                geometry_anomaly_score=0.50,
                is_available=False,
            )

        # 1. Height and Aspect Ratio of detected word tokens
        heights = []
        aspects = []
        token_boxes = []  # [x0, y0, x1, y1] in pixels

        for tok in raw_tokens:
            nb = tok.get("norm_box")
            if nb is not None:
                x0 = nb[0] * w_img
                y0 = nb[1] * h_img
                x1 = nb[2] * w_img
                y1 = nb[3] * h_img
                bw = max(1.0, x1 - x0)
                bh = max(1.0, y1 - y0)
                heights.append(bh)
                aspects.append(bw / bh)
                token_boxes.append((x0, y0, x1, y1))

        if not heights:
            return GeometryFeatureVector(
                sample_id=sample_id,
                char_height_mean=0.0,
                char_height_std=0.0,
                char_aspect_variance=0.0,
                baseline_deviation_variance=0.0,
                inter_word_gap_variance=0.0,
                font_ratio_amount_payee=1.0,
                font_ratio_amount_txnid=1.0,
                geometry_anomaly_score=0.50,
                is_available=False,
            )

        h_mean = float(np.mean(heights))
        h_std = float(np.std(heights))
        aspect_var = float(np.var(aspects))

        # 2. Text Baseline Deviation Variance
        # Group horizontally aligned words whose vertical centers are within 10 pixels
        token_boxes_sorted = sorted(token_boxes, key=lambda b: (b[1], b[0]))
        lines: List[List[Tuple[float, float, float, float]]] = []

        for b in token_boxes_sorted:
            y_mid = (b[1] + b[3]) / 2.0
            placed = False
            for line in lines:
                line_y_mid = np.mean([(lb[1] + lb[3]) / 2.0 for lb in line])
                if abs(y_mid - line_y_mid) <= 12.0:
                    line.append(b)
                    placed = True
                    break
            if not placed:
                lines.append([b])

        # For multi-token lines, fit baseline linear regression y_bottom = m * x_center + c
        baseline_residuals = []
        gap_variances = []

        for line in lines:
            if len(line) >= 3:
                line_sorted = sorted(line, key=lambda b: b[0])
                xs = np.array([(b[0] + b[2]) / 2.0 for b in line_sorted])
                ys_bottom = np.array([b[3] for b in line_sorted])
                # Linear fit
                p = np.polyfit(xs, ys_bottom, deg=1)
                fit_ys = np.polyval(p, xs)
                res = ys_bottom - fit_ys
                baseline_residuals.extend(res.tolist())

                # Horizontal gaps between adjacent words
                gaps = [line_sorted[i+1][0] - line_sorted[i][2] for i in range(len(line_sorted) - 1)]
                if len(gaps) >= 2:
                    gap_variances.append(float(np.var(gaps)))

        base_var = float(np.var(baseline_residuals)) if len(baseline_residuals) >= 3 else 0.0
        gap_var = float(np.mean(gap_variances)) if gap_variances else 0.0

        # 3. Relative Field Height Ratios
        amt_h = 40.0
        rec_h = 30.0
        txn_h = 20.0

        if "amount" in detailed_fields and detailed_fields["amount"].get("pixel_box"):
            amt_h = max(10.0, detailed_fields["amount"]["pixel_box"][3])
        if "receiver" in detailed_fields and detailed_fields["receiver"].get("pixel_box"):
            rec_h = max(10.0, detailed_fields["receiver"]["pixel_box"][3])
        if "transaction_id" in detailed_fields and detailed_fields["transaction_id"].get("pixel_box"):
            txn_h = max(10.0, detailed_fields["transaction_id"]["pixel_box"][3])

        ratio_amt_rec = amt_h / rec_h
        ratio_amt_txn = amt_h / txn_h

        # Anomaly scoring: genuine GPay dark receipts have ratio_amt_rec in [1.2, 2.5]
        # and low baseline residual variance (< 2.5)
        anomaly_cues = []
        if base_var > 3.0:
            anomaly_cues.append(min(1.0, (base_var - 3.0) / 10.0))
        if ratio_amt_rec < 1.1 or ratio_amt_rec > 3.5:
            anomaly_cues.append(0.3)
        if h_std > 25.0:
            anomaly_cues.append(0.2)

        geom_score = float(np.clip(sum(anomaly_cues), 0.0, 1.0)) if anomaly_cues else 0.05

        return GeometryFeatureVector(
            sample_id=sample_id,
            char_height_mean=round(h_mean, 2),
            char_height_std=round(h_std, 2),
            char_aspect_variance=round(aspect_var, 4),
            baseline_deviation_variance=round(base_var, 4),
            inter_word_gap_variance=round(gap_var, 4),
            font_ratio_amount_payee=round(ratio_amt_rec, 3),
            font_ratio_amount_txnid=round(ratio_amt_txn, 3),
            geometry_anomaly_score=round(geom_score, 4),
            is_available=True,
        )

