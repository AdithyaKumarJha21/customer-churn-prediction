# Customer Churn Prediction Dashboard

An interactive Streamlit app that predicts customer churn risk and recommends targeted retention actions. Upload your customer data to identify high-risk accounts, explore churn patterns, and evaluate model performance.

## Features

- **Predict churn probability** for individual customers with risk labels (Low/Medium/High)
- **Explore churn trends** by contract type, tenure, billing method, and support activity
- **Model insights** including accuracy, ROC AUC, confusion matrix, and feature importance
- **Work with your data** — use synthetic data or upload a CSV file
- **Retention recommendations** based on customer behavior patterns
- **Export results** — download high-risk customers as CSV

## Tech Stack

Python · Streamlit · Scikit-learn · Pandas · NumPy · Plotly

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/AdithyaKumarJha21/customer-churn-prediction.git
cd customer-churn-prediction

python -m venv .venv
source .venv/bin/activate  # On Windows: .\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

## Quick Start

Run the dashboard:

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser. Use the default synthetic dataset or upload a CSV with these columns: