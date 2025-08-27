# Synthetic Data Audit API

A production-grade, serverless API for automatically auditing synthetic datasets for **privacy**, **fairness**, and **fidelity**. Deploys seamlessly to AWS Lambda to provide a scalable, cost-effective trust layer for your AI and data pipelines.

[![AWS Serverless](https://img.shields.io/badge/AWS-Serverless-FF9900?logo=amazonaws)](https://aws.amazon.com/serverless/)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/Framework-FastAPI-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🚀 Features

- **🔒 Privacy Audit**: Evaluates the risk of membership inference attacks (MIA), assessing how easily real individuals can be identified from the synthetic dataset.
-  **⚖️ Fairness Audit**: Calculates Demographic Parity Difference across user-specified protected attributes (e.g., gender, race) to detect potential bias.
-  **📊 Fidelity Audit**: Measures the statistical similarity between synthetic and real data using Total Variation Distance (TVD) on feature distributions.
-  **🤖 Serverless & Scalable**: Built on AWS Lambda for automatic scaling and minimal operational overhead.
-  **✅ Production-Ready**: Includes comprehensive error handling, logging, security best practices, and infrastructure-as-code.

## 📁 Project Structure
synthetic-data-auditor/
├── app/
│ ├── init.py
│ ├── main.py # FastAPI application and endpoint definitions
│ └── audit_modules/ # Core audit logic
│ ├── init.py
│ ├── privacy.py # Membership Inference Attack simulation
│ ├── fairness.py # Demographic Parity calculation
│ └── fidelity.py # Total Variation Distance calculation
├── tests/ # Unit and integration tests (Pytest)
├── template.yaml # AWS SAM template for deployment
├── requirements.txt # Python dependencies
└── README.md


## 🛠️ Installation & Local Development

### Prerequisites

- **Python 3.11+**
- **AWS SAM CLI** ([Installation Guide](https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/install-sam-cli.html))
- **Docker** (required for local API testing with SAM)

### 1. Clone and Set Up Environment

```bash
# Clone the repository (or create your directory)
git clone <your-repo-url>
cd synthetic-data-auditor

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt
2. Run the API Locally
You can run the API in two ways:

Option A: Directly with Uvicorn (Fastest for development)

bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
The API will be available at http://localhost:8000. Interactive API docs (Swagger UI) will be at http://localhost:8000/docs.

Option B: Using AWS SAM (Simulates Lambda environment)

bash
sam build
sam local start-api
The API will be available at http://localhost:3000.

3. Test the API Endpoint
Use curl to test the /audit endpoint with a sample payload.

bash
curl -X 'POST' \
  'http://localhost:8000/audit' \
  -H 'Content-Type: application/json' \
  -d '{
  "synthetic_data_url": "https://your-bucket.s3.amazonaws.com/synthetic_sample.csv",
  "real_data_url": "https://your-bucket.s3.amazonaws.com/real_sample.csv",
  "protected_attributes": ["age_group", "gender"]
}'
🚀 Deployment to AWS
Deploy the entire application to your AWS account with a single command. The AWS SAM CLI handles packaging and creating all necessary resources (Lambda, API Gateway, IAM roles).

Build the application:

bash
sam build
Deploy it:

bash
sam deploy --guided
This wizard will prompt you for an AWS region, stack name, and other parameters. This only needs to be run once.

Retrieve the API Endpoint:
After deployment, SAM will output the URL of your live API Gateway endpoint. Use this URL in your applications.

📚 API Usage
Request
Send a POST request to the /audit endpoint.

JSON Body:

json
{
  "synthetic_data_url": "string",         // Required: Pre-signed S3 URL to a CSV file
  "real_data_url": "string",              // Optional: Pre-signed S3 URL for fidelity checks
  "protected_attributes": ["string"]      // Optional: List of column names for bias analysis
}
Response
A typical successful response will look like this:

json
{
  "overall_score": 0.82,
  "module_scores": {
    "privacy": 0.75,
    "fairness": 0.90,
    "fidelity": 0.88
  },
  "detailed_findings": {
    "privacy": {
      "risk_level": "medium",
      "message": "Membership inference attack success rate was 18%."
    },
    "fairness": {
      "risk_level": "low",
      "message": "Demographic parity difference within acceptable threshold (< 0.05) for all specified attributes."
    },
    "fidelity": {
      "risk_level": "low",
      "message": "Total Variation Distance (TVD) of 0.12 indicates good statistical alignment with real data."
    }
  }
}
🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

Fork the Project

Create your Feature Branch (git checkout -b feature/AmazingFeature)

Commit your Changes (git commit -m 'feat: Add some AmazingFeature')

Push to the Branch (git push origin feature/AmazingFeature)

Open a Pull Request

📜 License
This project is licensed under the MIT License - see the LICENSE file for details.

🙏 Acknowledgments
Built with FastAPI.

Deployed with the AWS Serverless Application Model (SAM).

Audit metrics inspired by current research in ML fairness and privacy.
