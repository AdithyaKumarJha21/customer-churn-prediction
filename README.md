# 📊 Customer Churn Prediction Dashboard

An interactive Streamlit dashboard that predicts customer churn risk using machine learning and behavioral analytics. Built for customer success teams to identify high-risk accounts before they leave, understand churn drivers, and execute data-driven retention strategies.

## ✨ Key Features

- **Predict Individual Customer Churn** — Input customer tenure, contract type, billing, usage, and support data to get churn probability and actionable risk labels (Low/Medium/High)
- **Train Custom ML Models** — Upload your own customer dataset or use synthetic data; the app trains a Random Forest classifier with 78/22 validation split
- **Analyze Churn Patterns** — Explore interactive dashboards showing churn rates by contract type, tenure, billing method, monthly charges, and support engagement
- **Model Performance Metrics** — View ROC AUC, confusion matrix, feature importance rankings, and accuracy scores to validate model quality
- **Smart Retention Recommendations** — Get context-specific retention actions based on each customer's risk profile and behavioral patterns
- **Export High-Risk Customers** — Download validation results as CSV for immediate outreach and retention campaigns

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **Backend & ML** | Python 3.10+, Scikit-learn (Random Forest), Pandas, NumPy, Joblib |
| **Frontend** | Streamlit |
| **Visualization** | Plotly |
| **Version Control** | Git/GitHub |

## 📦 Installation & Setup

### Prerequisites
- Python 3.10 or newer
- Git

### Step-by-Step Setup

**On Windows (PowerShell):**
```powershell
git clone https://github.com/AdithyaKumarJha21/customer-churn-prediction.git
cd customer-churn-prediction

python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
```

**On macOS/Linux:**
```bash
git clone https://github.com/AdithyaKumarJha21/customer-churn-prediction.git
cd customer-churn-prediction

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

## 🚀 Quick Start

Start the Streamlit app:

```bash
streamlit run app.py
```

Open your browser to `http://localhost:8501`

## 📸 Example Output
![alt text](<Screenshot 2026-09-12 160310.png>)

### Output 2

![alt text](<Screenshot 2026-09-12 160523.png>)

### First Time Using It

1. The app loads with synthetic data (1,000-10,000 configurable records)
2. Browse the **Overview** tab to see churn distribution
3. Check **Segments** tab for churn trends by customer attributes
4. Go to **Model** tab to review validation metrics and feature importance
5. Use **Predict** tab to enter a single customer profile and click "Predict Churn Risk"
6. Export high-risk customers from the Model tab

### Upload Your Own Data

Create a CSV with these required columns:

| Column | Description |
|--------|-------------|
| `tenure_months` | Customer tenure in months |
| `contract_type` | Type of contract |
| `internet_service` | Service type |
| `payment_method` | Payment method |
| `paperless_billing` | Yes/No |
| `senior_citizen` | Yes/No |
| `monthly_charges` | Monthly bill amount |
| `total_charges` | Total lifetime charges |
| `support_tickets` | Number of support tickets |
| `avg_monthly_usage_gb` | Average monthly data usage |
| `late_payments` | Number of late payments |
| `Churn` | **Yes/No** (required label) |

Upload from the sidebar—the model will retrain on your data.

## 📁 Project Structure

```
customer-churn-prediction/
├── app.py
├── requirements.txt
├── churn_model.joblib
├── .gitignore
└── README.md
```