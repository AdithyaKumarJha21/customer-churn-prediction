# Customer Churn Prediction Dashboard

An interactive Streamlit dashboard that trains a Random Forest classifier on customer records, highlights churn risk drivers, and predicts the churn risk of an individual customer.

## Problem Statement

Acquiring new customers is significantly more expensive than retaining existing ones.

Businesses need a way to identify customers who are likely to leave before churn occurs.

This project helps organizations detect high-risk customers, understand churn drivers, and take proactive retention actions.

## Features

- Synthetic customer data generation
- CSV dataset upload support
- Missing value handling
- Feature engineering
- Random Forest churn prediction
- Threshold optimization
- Risk segmentation
- Feature importance analysis
- Interactive visualizations
- Individual customer prediction
- Retention recommendations
- High-risk customer export

## Tech Stack

### Programming Language

- Python

### Framework

- Streamlit

### Libraries

- Pandas
- NumPy
- Scikit-learn
- Plotly
- Joblib

## Current Default Model Result

On the built-in 5,200-record dataset, the Random Forest validation run reaches about 91% accuracy with ROC AUC around 0.97. The strongest churn indicators are short tenure, month-to-month contracts, contract risk score, and support ticket volume.

## Run Locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

The app opens in your browser at the address printed by Streamlit, usually `http://localhost:8501`.

## Data Input

The app generates a synthetic dataset by default. You can instead upload a CSV with a `Churn` target column containing `Yes` or `No` values.

Required input columns:

```text
tenure_months, contract_type, internet_service, payment_method,
paperless_billing, senior_citizen, monthly_charges, total_charges,
support_tickets, avg_monthly_usage_gb, late_payments, Churn
```

`customer_id` is optional. The trained `churn_model.joblib` artifact is generated locally at runtime and is intentionally excluded from version control.
