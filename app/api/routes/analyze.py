"""
Analyze endpoint routing for CBL-04 screenshot manipulation detection.
Accepts multipart/form-data image uploads, enforces binary content checks,
invokes the analysis service, and returns structured AnalyzeResponse schemas.
Phase 3 integrates live Member 2 ResNet-18 multi-task CNN inference.
"""

import time
from typing import Optional
from fastapi import APIRouter, File, Form, Header, Query, UploadFile, status

from app.core.config import get_settings
from app.schemas.cnn import CNNModelInput
from app.schemas.enums import ErrorCodeEnum, VerdictEnum
from app.schemas.response import (
    AnalyzeResponse,
    BranchScores,
    ManipulationAnalysisEvidence,
    ModelVersionMetadata,
)
from app.services import (
    CNNModelError,
    CNNModelUnavailableError,
    get_cnn_service,
    get_ocr_service,
    get_semantic_service,
    get_stacker_service,
    get_final_verdict_service,
)
from app.services.image_validation_service import (
    ImageValidationError,
    validate_and_inspect_image,
)
from app.services.mock_analysis_service import MockAnalysisService

router = APIRouter(tags=["Analysis"])


@router.post(
    "/api/v1/analyze",
    response_model=AnalyzeResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyze Google Pay Receipt Screenshot",
    description=(
        "Upload a Google Pay receipt screenshot (.jpg, .jpeg, .png, .webp, max 10MB) "
        "to verify authenticity and detect digital tampering. Phase 3 performs live "
        "CNN visual feature analysis with simulated OCR and semantic branches."
    ),
)
async def analyze_screenshot(
    image: UploadFile = File(
        ...,
        description="The Google Pay receipt screenshot image file (binary stream)"
    ),
    request_id: Optional[str] = Form(
        None,
        description="Optional client correlation ID for request tracing"
    ),
    return_heatmap: bool = Form(
        True,
        description="Whether to generate and return Grad-CAM spatial localization evidence"
    ),
    verdict_style: Optional[str] = Query(
        None,
        description="Override verdict style: 'canonical' (GENUINE/EDITED) or 'frontend' (REAL/FAKE)"
    ),
    x_verdict_style: Optional[str] = Header(
        None,
        alias="X-Verdict-Style",
        description="Header override for verdict style preference"
    ),
) -> AnalyzeResponse:
    """
    Main image inspection endpoint.
    Performs streaming size-capped read, binary validation, and analysis synthesis.
    Integrates live CNN visual manipulation inference when enabled.
    """
    start_time = time.perf_counter()
    settings = get_settings()
    style = verdict_style or x_verdict_style or settings.DEFAULT_VERDICT_STYLE
    effective_req_id = request_id or f"req-{int(time.time() * 1000)}"

    # Read binary bytes with streaming size ceiling to protect server memory
    max_bytes = settings.MAX_FILE_SIZE_BYTES
    chunk_size = 1024 * 1024  # 1 MB chunks
    file_bytes = bytearray()

    try:
        while True:
            chunk = await image.read(chunk_size)
            if not chunk:
                break
            file_bytes.extend(chunk)
            if len(file_bytes) > max_bytes:
                max_mb = max_bytes / (1024 * 1024)
                raise ImageValidationError(
                    error_code=ErrorCodeEnum.FILE_TOO_LARGE,
                    status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                    message=f"Uploaded file exceeds maximum allowed size of {max_mb:.1f} MB.",
                )

        raw_bytes = bytes(file_bytes)

        # Binary content, format, and dimension validation
        validated_image = validate_and_inspect_image(
            file_bytes=raw_bytes,
            filename=image.filename,
            content_type=image.content_type,
            settings=settings,
        )

        # Pipeline branch dispatch: Live CNN vs Mock Mode
        if settings.CNN_ENABLED and not settings.MOCK_MODE:
            # 1. Live CNN Inference
            cnn_service = get_cnn_service()
            cnn_input = CNNModelInput(
                image_bytes=raw_bytes,
                image_dimensions=(validated_image.width, validated_image.height),
            )
            cnn_output = cnn_service.predict(cnn_input)

            # 2. OCR Extraction Branch (Live EasyOCR with robust unavailable state)
            ocr_error_warning = None
            if settings.OCR_ENABLED:
                try:
                    ocr_service = get_ocr_service()
                    ocr_res = ocr_service.extract(raw_bytes)
                    ocr_status = "live"
                except Exception as ocr_exc:
                    logger.error("OCR extraction failed [Trace: %s]: %s", effective_req_id, ocr_exc)
                    from app.schemas.ocr import ExtractedTransactionData, OCRResult
                    ocr_res = OCRResult(
                        fields=ExtractedTransactionData(),
                        detailed_fields={},
                        raw_detections=[],
                        overall_confidence=0.0,
                        ocr_engine=settings.OCR_ENGINE,
                        ocr_version="unavailable",
                        raw_text="",
                        processing_time_ms=0.0,
                    )
                    ocr_status = "unavailable"
                    ocr_error_warning = f"OCR engine was unable to extract text ({str(ocr_exc)}). Visual CNN analysis preserved."
            else:
                from app.schemas.ocr import ExtractedTransactionData, OCRResult
                ocr_res = OCRResult(
                    fields=ExtractedTransactionData(),
                    detailed_fields={},
                    raw_detections=[],
                    overall_confidence=0.0,
                    ocr_engine="None",
                    ocr_version="disabled",
                    raw_text="",
                    processing_time_ms=0.0,
                )
                ocr_status = "disabled"

            # 3. Semantic Consistency Branch (Live Phase 6 Engine)
            if settings.SEMANTIC_ENABLED and ocr_status == "live":
                semantic_service = get_semantic_service()
                semantic_res = semantic_service.extract_features(
                    ocr_result=ocr_res,
                    image_dimensions=(validated_image.width, validated_image.height),
                )
                semantic_status = "live"
            else:
                semantic_service = get_semantic_service()
                semantic_res = semantic_service.extract_features(
                    ocr_result=ocr_res,
                    image_dimensions=(validated_image.width, validated_image.height),
                )
                semantic_status = "unavailable" if ocr_status != "live" else "live"

            # 4. Out-Of-Fold (OOF) Stacker Meta-Learner (Live in Phase 8)
            stacker_service = get_stacker_service()
            stacker_res = stacker_service.predict_branches(
                cnn_result=cnn_output,
                ocr_result=ocr_res,
                semantic_result=semantic_res,
            )
            stacker_status = "live"

            # 5. Final Verdict & Calibration Decision Engine (Live in Phase 9)
            verdict_service = get_final_verdict_service()
            verdict_res = verdict_service.evaluate_verdict(
                stacker_output=stacker_res,
                cnn_result=cnn_output,
                ocr_result=ocr_res,
                semantic_result=semantic_res,
            )

            # Style-specific verdict mapping
            if style.lower() == "frontend":
                if verdict_res.verdict == VerdictEnum.GENUINE:
                    assigned_verdict = VerdictEnum.REAL
                elif verdict_res.verdict == VerdictEnum.EDITED:
                    assigned_verdict = VerdictEnum.FAKE
                elif verdict_res.verdict == VerdictEnum.SUSPICIOUS:
                    assigned_verdict = VerdictEnum.SUSPICIOUS
                else:
                    assigned_verdict = VerdictEnum.INCONCLUSIVE
            else:
                assigned_verdict = verdict_res.verdict

            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

            warnings = list(verdict_res.warnings)
            if ocr_error_warning:
                warnings.insert(0, ocr_error_warning)
            warnings.extend([
                "Notice: Screenshot OCR extracts visible receipt text for auditing and semantic parsing; "
                "it does not verify bank-side ledger entries or confirm banking settlement.",
                "CNN score and stacker score are uncalibrated experimental model outputs.",
            ])
            if cnn_output.heatmap_data_uri:
                warnings.append("Explainability: Live neural activation heatmap generated from ResNet-18 localization decoder.")

            # Enrich suspicious regions with absolute pixel bounds for UI rendering if normalized box is available
            enriched_regions = []
            for reg in verdict_res.suspicious_regions:
                reg_copy = reg.model_copy()
                if reg_copy.box_normalized and (reg_copy.x is None or reg_copy.y is None):
                    pb = reg_copy.box_normalized.to_pixel(validated_image.width, validated_image.height)
                    reg_copy.x = pb.x
                    reg_copy.y = pb.y
                    reg_copy.width = pb.width
                    reg_copy.height = pb.height
                enriched_regions.append(reg_copy)

            return AnalyzeResponse(
                request_id=effective_req_id,
                verdict=assigned_verdict,
                confidence=verdict_res.confidence,
                decision_score=verdict_res.decision_score,
                uncertainty_score=verdict_res.uncertainty_score,
                manipulation_probability=verdict_res.manipulation_probability,
                message=verdict_res.message,
                ocr_data=ocr_res.fields,
                manipulation_analysis=ManipulationAnalysisEvidence(
                    detected=len(enriched_regions) > 0,
                    suspicious_regions=enriched_regions,
                    heatmap_url=cnn_output.heatmap_data_uri,
                ),
                branch_scores=BranchScores(
                    cnn_score=cnn_output.manipulation_probability,
                    semantic_score=semantic_res.semantic_manipulation_score,
                    stacker_score=stacker_res.final_manipulation_probability,
                ),
                cnn_score=cnn_output.manipulation_probability,
                semantic_score=semantic_res.semantic_manipulation_score,
                stacker_score=stacker_res.final_manipulation_probability,
                semantic_details=semantic_res,
                reason_codes=verdict_res.reason_codes,
                model_versions=ModelVersionMetadata(
                    pipeline_version="v1.0.0-phase9-final-verdict",
                    cnn_model_version=cnn_output.model_info.model_version,
                    ocr_engine_version=f"{ocr_res.ocr_engine} {ocr_res.ocr_version}",
                    semantic_engine_version=f"{semantic_res.model_info.model_name} {semantic_res.model_info.model_version}",
                    stacker_model_version=stacker_res.stacker_model_version,
                ),
                processing_time_ms=round(elapsed_ms, 2),
                warnings=warnings,
                pipeline_status="phase9_complete",
                component_status={
                    "cnn": "live",
                    "ocr": ocr_status,
                    "semantic": semantic_status,
                    "stacker": stacker_status,
                    "verdict_engine": "live",
                },
            )

        # Fallback to pure mock pipeline when MOCK_MODE is enabled
        response = MockAnalysisService.run_mock_pipeline(
            image=validated_image,
            request_id=effective_req_id,
            return_heatmap=return_heatmap,
            verdict_style=style,
        )
        return response

    finally:
        # Guarantee release of stream and memory buffers
        await image.close()


@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    status_code=status.HTTP_200_OK,
    include_in_schema=False,
)
async def analyze_screenshot_alias(
    image: UploadFile = File(...),
    request_id: Optional[str] = Form(None),
    return_heatmap: bool = Form(True),
    verdict_style: Optional[str] = Query(None),
    x_verdict_style: Optional[str] = Header(None, alias="X-Verdict-Style"),
) -> AnalyzeResponse:
    """Compatibility alias for /analyze without /api/v1 prefix."""
    return await analyze_screenshot(
        image=image,
        request_id=request_id,
        return_heatmap=return_heatmap,
        verdict_style=verdict_style,
        x_verdict_style=x_verdict_style,
    )
