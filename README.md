# Synthetic Data Audit API (FastAPI + AWS Lambda via Mangum)

Production-grade MVP that audits synthetic datasets for **Privacy**, **Fairness**, and **Fidelity**.

- **Framework:** FastAPI on AWS Lambda (via Mangum)
- **Data:** Pandas + scikit-learn
- **Deploy:** AWS SAM (`sam build && sam deploy --guided`)

---

## 1) Features

- **POST `/audit`**
  - **Input:**
    - `synthetic_data_url` *(required)* – Pre-signed S3 URL to CSV
    - `real_data_url` *(optional)* – Pre-signed S3 URL to CSV
    - `protected_attributes` *(optional)* – Column names for fairness analysis
  - **Output:**
    - `overall_score`: float
    - `module_scores`: `{ privacy, fairness, fidelity }`
    - `detailed_findings`: rich per-module diagnostics

- **Modules**
  - **Privacy:** Nearest-neighbor linkage risk (membership inference approximation)
  - **Fairness:** Demographic Parity Difference across protected attributes
  - **Fidelity:** Total Variation Distance (TVD) across numeric marginals

---

## 2) Local Development

> Requires: Python 3.11+

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt