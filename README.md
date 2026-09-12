# Customer Churn Prediction Dashboard

An interactive Streamlit dashboard that estimates customer churn risk from tenure, contract, billing, usage, and support data. It helps teams identify high-risk customers, understand churn patterns, and choose targeted retention actions.

## Key Features

- Train a balanced Random Forest classifier on generated or uploaded customer data.
- Score individual customers with a predicted churn probability and risk label.
- Explore churn rates, risk segments, validation metrics, and feature importance.
- Generate recommended retention actions from a customer's profile.
- Export high-risk validation customers as a CSV file.

## Tech Stack

- Python
- Streamlit
- Pandas and NumPy
- Scikit-learn
- Plotly
- Joblib

## Installation & Setup

From the project directory, create and activate a virtual environment, then install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Quick Start

Start the dashboard with:

```powershell
streamlit run app.py
```

Open the local URL shown by Streamlit, usually `http://localhost:8501`. The app uses synthetic data by default; upload a CSV from the sidebar to score your own dataset. Uploaded files must include a `Churn` column with `Yes` or `No` values and the required customer feature columns.

## Project Structure

```text
.
├── app.py                 # Streamlit dashboard and ML workflow
├── churn_model.joblib     # Saved trained model pipeline
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

## Contributing

Create a focused branch, make the change, verify the app locally, and open a pull request with a concise description of the work.

## License

No license file is currently included in this repository.
