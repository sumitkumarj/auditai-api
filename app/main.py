import io
import json
import logging
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from mangum import Mangum
from pydantic import BaseModel, Field, HttpUrl, ValidationError, field_validator

from app.audit_modules.privacy import run_membership_inference_attack
from app.audit_modules.fairness import calculate_demographic_parity
from app.audit_modules.fidelity import calculate_total_variation_distance


# ---------------------------
# Logging Configuration
# ---------------------------
class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_record = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            log_record["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(log_record)


logger = logging.getLogger("audit_api")
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(JsonFormatter())
logger.setLevel(logging.INFO)
logger.addHandler(handler)


# ---------------------------
# Pydantic Models
# ---------------------------
class AuditRequest(BaseModel):
    synthetic_data_url: HttpUrl = Field(..., description="Pre-signed S3 URL to synthetic CSV")
    real_data_url: Optional[HttpUrl] = Field(
        None, description="Pre-signed S3 URL to real CSV for fidelity & privacy checks"
    )
    protected_attributes: Optional[List[str]] = Field(
        default=None, description="List of protected attribute column names for fairness analysis"
    )

    @field_validator("protected_attributes")
    @classmethod
    def validate_protected_attributes(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        if v is None:
            return v
        if not isinstance(v, list):
            raise ValueError("protected_attributes must be a list of strings")
        if any((not isinstance(item, str) or not item.strip()) for item in v):
            raise ValueError("protected_attributes must contain non-empty strings")
        return v


class ModuleScores(BaseModel):
    privacy: Optional[float] = None
    fairness: Optional[float] = None
    fidelity: Optional[float] = None


class AuditResponse(BaseModel):
    overall_score: float
    module_scores: ModuleScores
    detailed_findings: Dict[str, Any]


# ---------------------------
# Helpers
# ---------------------------
READ_CSV_KWARGS = dict(low_memory=True)

def _download_csv(url: str) -> pd.DataFrame:
    """
    Download a CSV from a (pre-signed) URL into a pandas DataFrame.

    Uses streaming GET to avoid loading the entire file into memory at once,
    then assembles into BytesIO for a single pass by Pandas.
    """
    try:
        logger.info(f"Downloading CSV from URL")
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            buf = io.BytesIO()
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    buf.write(chunk)
            buf.seek(0)
        # Pandas sniffing handles gzip if 'Content-Encoding' isn't reliable
        return pd.read_csv(buf, **READ_CSV_KWARGS)
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download CSV: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to download CSV: {e}")
    except Exception as e:
        logger.error(f"Failed to parse CSV: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {e}")


def _infer_binary_target_column(df: pd.DataFrame) -> Optional[str]:
    """
    Heuristic to find a binary outcome column for fairness:
    - prefer columns named ['label', 'target', 'outcome'] if binary
    - else any boolean or binary (0/1 or 'yes'/'no') column
    """
    candidate_names = ["label", "target", "outcome"]
    for name in candidate_names:
        if name in df.columns:
            col = df[name]
            unique = pd.Series(col.dropna().unique())
            if len(unique) == 2:
                return name

    # Try to find a boolean/binary column
    for colname in df.columns:
        col = df[colname]
        # Boolean dtype
        if pd.api.types.is_bool_dtype(col):
            return colname
        # Check binary numeric
        if pd.api.types.is_numeric_dtype(col):
            unique = pd.Series(col.dropna().unique())
            if unique.isin([0, 1]).all() and 1 in unique.values and 0 in unique.values:
                return colname
        # Check binary strings 'yes'/'no', 'true'/'false'
        if pd.api.types.is_object_dtype(col):
            normalized = col.dropna().astype(str).str.strip().str.lower().unique()
            if set(normalized).issubset({"yes", "no"}) and len(normalized) == 2:
                return colname
            if set(normalized).issubset({"true", "false"}) and len(normalized) == 2:
                return colname
    return None


def _normalize_score(value: float, lower: float, upper: float) -> float:
    """
    Normalize value to [0, 1] given known bounds.
    Clamps outside range.
    """
    if np.isnan(value):
        return 0.0
    if upper == lower:
        return 1.0
    norm = (value - lower) / (upper - lower)
    return float(np.clip(norm, 0.0, 1.0))


def _safe_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    """Return a dataframe with numeric columns only and without all-NA columns."""
    numeric_df = df.select_dtypes(include=[np.number]).copy()
    if numeric_df.empty:
        raise HTTPException(
            status_code=400,
            detail="No numeric columns found. Provide CSV with at least one numeric column."
        )
    numeric_df = numeric_df.dropna(axis=1, how="all")
    if numeric_df.empty:
        raise HTTPException(
            status_code=400,
            detail="All numeric columns are empty after NA removal."
        )
    return numeric_df


# ---------------------------
# FastAPI App
# ---------------------------
app = FastAPI(
    title="Synthetic Data Audit API",
    version="1.0.0",
    description="Production-grade API to assess synthetic data Privacy, Fairness, and Fidelity.",
)


@app.post("/audit", response_model=AuditResponse)
def audit(request: AuditRequest) -> JSONResponse:
    """
    Run Privacy, Fairness, and Fidelity audits on provided datasets.
    """
    logger.info("Starting audit request")

    # Load synthetic data
    synthetic_df = _download_csv(str(request.synthetic_data_url))
    if synthetic_df.empty:
        raise HTTPException(status_code=400, detail="Synthetic dataset is empty")

    logger.info(f"Synthetic columns: {list(synthetic_df.columns)}; rows={len(synthetic_df)}")

    # Attempt to load real data if provided
    real_df: Optional[pd.DataFrame] = None
    if request.real_data_url:
        real_df = _download_csv(str(request.real_data_url))
        if real_df.empty:
            raise HTTPException(status_code=400, detail="Real dataset is empty")
        logger.info(f"Real columns: {list(real_df.columns)}; rows={len(real_df)}")

    module_scores: Dict[str, Optional[float]] = {"privacy": None, "fairness": None, "fidelity": None}
    detailed_findings: Dict[str, Any] = {}

    # ---------------------------
    # Privacy (requires real data for strongest signal; otherwise limited self-similarity)
    # ---------------------------
    try:
        privacy_result = run_membership_inference_attack(synthetic_df=synthetic_df, real_df=real_df)
        module_scores["privacy"] = float(privacy_result["score"])
        detailed_findings["privacy"] = {k: v for k, v in privacy_result.items() if k != "score"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Privacy module failed: {e}", exc_info=True)
        detailed_findings["privacy_error"] = str(e)

    # ---------------------------
    # Fairness (requires protected attributes and a binary outcome column)
    # ---------------------------
    try:
        if request.protected_attributes:
            outcome_col = _infer_binary_target_column(synthetic_df)
            if outcome_col is None:
                detailed_findings["fairness_warning"] = (
                    "No binary outcome column found (checked: 'label', 'target', 'outcome', and general binary columns)."
                    " Fairness not computed."
                )
            else:
                fairness_result = calculate_demographic_parity(
                    df=synthetic_df,
                    outcome_column=outcome_col,
                    protected_attributes=request.protected_attributes,
                )
                module_scores["fairness"] = float(fairness_result["score"])
                detailed_findings["fairness"] = {k: v for k, v in fairness_result.items() if k != "score"}
        else:
            detailed_findings["fairness_info"] = "No protected_attributes provided; fairness not computed."
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Fairness module failed: {e}", exc_info=True)
        detailed_findings["fairness_error"] = str(e)

    # ---------------------------
    # Fidelity (requires real data)
    # ---------------------------
    try:
        if real_df is not None:
            # Use numeric columns common to both
            fidelity_result = calculate_total_variation_distance(
                real_df=_safe_numeric_df(real_df),
                synth_df=_safe_numeric_df(synthetic_df),
            )
            module_scores["fidelity"] = float(fidelity_result["score"])
            detailed_findings["fidelity"] = {k: v for k, v in fidelity_result.items() if k != "score"}
        else:
            detailed_findings["fidelity_info"] = "No real_data_url provided; fidelity not computed."
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Fidelity module failed: {e}", exc_info=True)
        detailed_findings["fidelity_error"] = str(e)

    # ---------------------------
    # Overall score: average available module scores
    # ---------------------------
    scores = [s for s in module_scores.values() if isinstance(s, (int, float))]
    overall = float(np.mean(scores)) if scores else 0.0

    response = AuditResponse(
        overall_score=overall,
        module_scores=ModuleScores(**module_scores),
        detailed_findings=detailed_findings,
    )
    logger.info("Audit completed successfully")
    return JSONResponse(content=json.loads(response.model_dump_json()))


# AWS Lambda handler
lambda_handler = Mangum(app)


# Local dev server entrypoint
if __name__ == "__main__":
    # For local testing only:
    import uvicorn  # type: ignore

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
