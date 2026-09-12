# Customer Churn Prediction System

An interactive Streamlit dashboard that trains a Random Forest classifier on customer records, highlights churn risk drivers, and predicts the churn risk of an individual customer.

## Features

- Generates a realistic 5,000+ row customer churn dataset when no external data is provided.
- Cleans and engineers predictive features with Pandas.
- Trains a Scikit-learn Random Forest model and reports validation metrics.
- Provides real-time single-customer churn prediction.
- Visualizes churn risk distribution, spending behavior, contract risk, tenure risk, and feature importance with Plotly.
- Tunes the decision threshold on validation data and reports accuracy plus ROC AUC.
- Recommends retention actions for high-risk customer profiles.

## Tech Stack

Python, Pandas, Scikit-learn, Random Forest, Streamlit, Plotly

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
