# Customer Churn Prediction Dashboard

An interactive machine learning dashboard that helps businesses identify customers at risk of churning, understand the factors driving that risk, and take proactive retention action.

The application combines customer data preparation, Random Forest classification, model evaluation, interactive visualizations, and real-time customer scoring in one Streamlit interface.

## Problem Statement

Customer acquisition is significantly more expensive than customer retention. However, businesses often discover that a customer is likely to leave only after churn has already occurred.

This project addresses that challenge by analyzing customer tenure, contract details, billing behavior, service usage, and support activity to estimate churn probability before the customer leaves.

The resulting dashboard enables teams to prioritize high-risk customers, explore churn patterns, and select targeted retention actions.

## Features

### Data Preparation

- Generate a realistic synthetic customer dataset with more than 5,000 records.
- Upload a customer CSV dataset with a `Churn` target column.
- Handle missing numeric and categorical values.
- Create predictive features such as contract risk, monthly spend, and new-customer status.

### Machine Learning

- Train a balanced Random Forest churn classifier.
- Preprocess numeric and categorical features with a Scikit-learn pipeline.
- Optimize the classification threshold using validation data.
- Evaluate accuracy, ROC AUC, classification metrics, and a confusion matrix.
- Display the features with the greatest influence on model predictions.

### Business Insights

- Segment customers into low-, medium-, and high-risk groups.
- Explore churn rates by contract type and customer tenure.
- Predict churn risk for an individual customer in real time.
- Receive recommended retention actions based on the customer profile.
- Export high-risk validation customers as a CSV file.

## Tech Stack

| Technology | Purpose |
| --- | --- |
| Python | Application, data processing, and machine learning development |
| Streamlit | Interactive dashboard and prediction interface |
| Pandas | Data loading, cleaning, transformation, and analysis |
| NumPy | Numerical operations and synthetic data generation |
| Scikit-learn | Preprocessing pipeline, Random Forest model, and evaluation metrics |
| Plotly | Interactive charts and model insight visualizations |
| Joblib | Serialization of the trained machine learning pipeline |

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
