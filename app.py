from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42
MODEL_PATH = "churn_model.joblib"

NUMERIC_FEATURES = [
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "support_tickets",
    "avg_monthly_usage_gb",
    "late_payments",
    "contract_risk_score",
    "spend_per_month",
    "new_customer_flag",
]

CATEGORICAL_FEATURES = [
    "contract_type",
    "internet_service",
    "payment_method",
    "paperless_billing",
    "senior_citizen",
]


@dataclass(frozen=True)
class ModelBundle:
    pipeline: Pipeline
    accuracy: float
    roc_auc: float
    threshold: float
    report: dict
    confusion: np.ndarray
    validation_frame: pd.DataFrame


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-values))


@st.cache_data(show_spinner=False)
def generate_customer_data(rows: int = 5200, seed: int = RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    contract_type = rng.choice(
        ["Month-to-month", "One year", "Two year"],
        size=rows,
        p=[0.56, 0.25, 0.19],
    )
    internet_service = rng.choice(["Fiber optic", "DSL", "No"], size=rows, p=[0.46, 0.38, 0.16])
    payment_method = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
        size=rows,
        p=[0.36, 0.18, 0.23, 0.23],
    )
    paperless_billing = rng.choice(["Yes", "No"], size=rows, p=[0.62, 0.38])
    senior_citizen = rng.choice(["Yes", "No"], size=rows, p=[0.17, 0.83])

    tenure_months = np.clip(rng.gamma(shape=2.0, scale=15.5, size=rows).round(), 1, 72).astype(int)
    base_charge = rng.normal(64, 18, rows)
    service_lift = np.select(
        [internet_service == "Fiber optic", internet_service == "DSL", internet_service == "No"],
        [18, 2, -24],
    )
    monthly_charges = np.clip(base_charge + service_lift, 18, 125).round(2)
    total_charges = (monthly_charges * tenure_months * rng.normal(1.0, 0.04, rows)).round(2)
    support_tickets = rng.poisson(
        np.where(contract_type == "Month-to-month", 1.4, 0.8)
        + np.where(internet_service == "Fiber optic", 0.35, 0)
    )
    avg_monthly_usage_gb = np.clip(
        rng.normal(180, 85, rows) + np.where(internet_service == "Fiber optic", 95, 0),
        0,
        650,
    ).round(1)
    late_payments = rng.poisson(np.where(payment_method == "Electronic check", 0.85, 0.42))

    churn_score = (
        0.8
        + 4.2 * (tenure_months < 3)
        + 2.4 * (contract_type == "Month-to-month")
        - 2.2 * (contract_type == "Two year")
        + 1.15 * (payment_method == "Electronic check")
        + 0.8 * (internet_service == "Fiber optic")
        + 0.45 * (paperless_billing == "Yes")
        + 0.35 * (senior_citizen == "Yes")
        + 0.36 * support_tickets
        + 0.46 * late_payments
        + 0.035 * np.maximum(monthly_charges - 78, 0)
        - 0.08 * np.maximum(tenure_months - 18, 0)
        + rng.normal(0, 0.55, rows)
    )
    churn = (churn_score > 4.1).astype(int)

    data = pd.DataFrame(
        {
            "customer_id": [f"CUST-{100000 + index}" for index in range(rows)],
            "tenure_months": tenure_months,
            "contract_type": contract_type,
            "internet_service": internet_service,
            "payment_method": payment_method,
            "paperless_billing": paperless_billing,
            "senior_citizen": senior_citizen,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "support_tickets": support_tickets,
            "avg_monthly_usage_gb": avg_monthly_usage_gb,
            "late_payments": late_payments,
            "Churn": np.where(churn == 1, "Yes", "No"),
        }
    )
    return engineer_features(data)


def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    frame = data.copy()
    frame.columns = [column.strip() for column in frame.columns]

    for column in ["total_charges", "monthly_charges", "tenure_months"]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    numeric_columns = frame.select_dtypes(include=["number"]).columns
    categorical_columns = frame.select_dtypes(exclude=["number"]).columns.drop(
        ["customer_id", "Churn"], errors="ignore"
    )

    for column in numeric_columns:
        frame[column] = frame[column].fillna(frame[column].median())
    for column in categorical_columns:
        frame[column] = frame[column].fillna(frame[column].mode().iloc[0])

    frame["contract_risk_score"] = frame.get("contract_type", "Month-to-month").map(
        {"Month-to-month": 3, "One year": 2, "Two year": 1}
    ).fillna(2)
    frame["spend_per_month"] = frame["total_charges"] / frame["tenure_months"].clip(lower=1)
    frame["new_customer_flag"] = (frame["tenure_months"] < 3).astype(int)
    return frame


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    classifier = RandomForestClassifier(
        n_estimators=420,
        max_depth=13,
        min_samples_leaf=4,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return Pipeline([("preprocess", preprocessor), ("model", classifier)])


@st.cache_resource(show_spinner=False)
def train_model(data: pd.DataFrame) -> ModelBundle:
    model_data = data.dropna(subset=["Churn"]).copy()
    x = model_data[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = model_data["Churn"].map({"No": 0, "Yes": 1}).astype(int)

    x_train, x_valid, y_train, y_valid = train_test_split(
        x, y, test_size=0.22, stratify=y, random_state=RANDOM_STATE
    )

    pipeline = build_pipeline()
    pipeline.fit(x_train, y_train)
    probabilities = pipeline.predict_proba(x_valid)[:, 1]
    thresholds = np.arange(0.25, 0.76, 0.01)
    threshold_scores = [
        accuracy_score(y_valid, (probabilities >= threshold).astype(int))
        for threshold in thresholds
    ]
    best_threshold = float(thresholds[int(np.argmax(threshold_scores))])
    predictions = (probabilities >= best_threshold).astype(int)

    validation_frame = x_valid.copy()
    validation_frame["actual_churn"] = np.where(y_valid == 1, "Yes", "No")
    validation_frame["predicted_churn"] = np.where(predictions == 1, "Yes", "No")
    validation_frame["churn_risk"] = probabilities

    bundle = ModelBundle(
        pipeline=pipeline,
        accuracy=accuracy_score(y_valid, predictions),
        roc_auc=roc_auc_score(y_valid, probabilities),
        threshold=best_threshold,
        report=classification_report(y_valid, predictions, output_dict=True),
        confusion=confusion_matrix(y_valid, predictions),
        validation_frame=validation_frame,
    )
    joblib.dump(bundle.pipeline, MODEL_PATH)
    return bundle


def get_feature_importance(pipeline: Pipeline) -> pd.DataFrame:
    preprocessor = pipeline.named_steps["preprocess"]
    cat_names = preprocessor.named_transformers_["cat"].get_feature_names_out(CATEGORICAL_FEATURES)
    names = np.concatenate([NUMERIC_FEATURES, cat_names])
    importances = pipeline.named_steps["model"].feature_importances_
    return (
        pd.DataFrame({"feature": names, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(12)
    )


def predict_customer(pipeline: Pipeline, customer: dict, threshold: float) -> tuple[float, str, list[str]]:
    frame = engineer_features(pd.DataFrame([customer]))
    probability = pipeline.predict_proba(frame[NUMERIC_FEATURES + CATEGORICAL_FEATURES])[:, 1][0]
    risk_label = "High risk" if probability >= threshold + 0.18 else "Medium risk" if probability >= threshold else "Low risk"

    actions = []
    if customer["tenure_months"] < 3:
        actions.append("Prioritize onboarding check-in")
    if customer["contract_type"] == "Month-to-month":
        actions.append("Offer annual-plan incentive")
    if customer["support_tickets"] >= 3:
        actions.append("Route to support recovery")
    if customer["late_payments"] >= 2:
        actions.append("Review payment-friction options")
    if not actions:
        actions.append("Maintain standard engagement")
    return probability, risk_label, actions


def risk_bucket(probability: float) -> str:
    if probability >= 0.65:
        return "High"
    if probability >= 0.35:
        return "Medium"
    return "Low"


def render_prediction_tool(bundle: ModelBundle) -> None:
    st.subheader("Real-Time Prediction")

    with st.form("prediction_form"):
        left, right = st.columns(2)
        with left:
            tenure = st.number_input("Tenure months", min_value=1, max_value=72, value=2)
            contract = st.selectbox("Contract type", ["Month-to-month", "One year", "Two year"])
            internet = st.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
            payment = st.selectbox(
                "Payment method",
                ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
            )
            paperless = st.selectbox("Paperless billing", ["Yes", "No"])
        with right:
            monthly = st.number_input("Monthly charges", min_value=18.0, max_value=125.0, value=82.0)
            tickets = st.number_input("Support tickets", min_value=0, max_value=15, value=2)
            usage = st.number_input("Avg monthly usage GB", min_value=0.0, max_value=650.0, value=240.0)
            late = st.number_input("Late payments", min_value=0, max_value=12, value=1)
            senior = st.selectbox("Senior citizen", ["No", "Yes"])

        submitted = st.form_submit_button("Predict Churn Risk", use_container_width=True)

    if submitted:
        customer = {
            "customer_id": "LIVE-CUSTOMER",
            "tenure_months": tenure,
            "contract_type": contract,
            "internet_service": internet,
            "payment_method": payment,
            "paperless_billing": paperless,
            "senior_citizen": senior,
            "monthly_charges": monthly,
            "total_charges": monthly * tenure,
            "support_tickets": tickets,
            "avg_monthly_usage_gb": usage,
            "late_payments": late,
        }
        probability, label, actions = predict_customer(bundle.pipeline, customer, bundle.threshold)
        st.metric("Predicted churn probability", f"{probability:.1%}", label)
        st.progress(float(probability))
        st.write("Recommended retention actions:")
        for action in actions:
            st.write(f"- {action}")


def render_dashboard(data: pd.DataFrame, bundle: ModelBundle) -> None:
    validation = bundle.validation_frame.copy()
    validation["risk_bucket"] = validation["churn_risk"].map(risk_bucket)

    metric_cols = st.columns(4)
    churn_rate = (data["Churn"] == "Yes").mean()
    metric_cols[0].metric("Records", f"{len(data):,}")
    metric_cols[1].metric("Model accuracy", f"{bundle.accuracy:.1%}")
    metric_cols[2].metric("ROC AUC", f"{bundle.roc_auc:.3f}")
    metric_cols[3].metric("Observed churn", f"{churn_rate:.1%}")

    tab_overview, tab_segments, tab_model, tab_predict = st.tabs(
        ["Overview", "Segments", "Model", "Predict"]
    )

    with tab_overview:
        left, right = st.columns(2)
        risk_counts = validation["risk_bucket"].value_counts().reindex(["Low", "Medium", "High"]).fillna(0)
        left.plotly_chart(
            px.bar(
                risk_counts,
                x=risk_counts.index,
                y=risk_counts.values,
                labels={"x": "Risk bucket", "y": "Customers"},
                color=risk_counts.index,
                color_discrete_sequence=["#2a9d8f", "#e9c46a", "#e76f51"],
            ),
            use_container_width=True,
        )
        right.plotly_chart(
            px.histogram(
                validation,
                x="churn_risk",
                nbins=30,
                labels={"churn_risk": "Predicted churn probability"},
                color_discrete_sequence=["#457b9d"],
            ),
            use_container_width=True,
        )

        st.plotly_chart(
            px.scatter(
                data,
                x="tenure_months",
                y="monthly_charges",
                color="Churn",
                size="support_tickets",
                hover_data=["contract_type", "payment_method"],
                color_discrete_map={"Yes": "#d62828", "No": "#2a9d8f"},
                labels={"tenure_months": "Tenure months", "monthly_charges": "Monthly charges"},
            ),
            use_container_width=True,
        )

    with tab_segments:
        left, right = st.columns(2)
        contract_rate = data.groupby("contract_type", as_index=False).agg(
            churn_rate=("Churn", lambda values: (values == "Yes").mean()),
            customers=("Churn", "size"),
        )
        left.plotly_chart(
            px.bar(
                contract_rate,
                x="contract_type",
                y="churn_rate",
                text="customers",
                labels={"contract_type": "Contract", "churn_rate": "Churn rate"},
                color="contract_type",
                color_discrete_sequence=["#e76f51", "#f4a261", "#2a9d8f"],
            ),
            use_container_width=True,
        )

        tenure_bins = pd.cut(
            data["tenure_months"],
            bins=[0, 3, 12, 24, 48, 72],
            labels=["<3", "3-12", "13-24", "25-48", "49-72"],
        )
        tenure_rate = data.assign(tenure_group=tenure_bins).groupby("tenure_group", observed=True).agg(
            churn_rate=("Churn", lambda values: (values == "Yes").mean()),
            customers=("Churn", "size"),
        ).reset_index()
        right.plotly_chart(
            px.line(
                tenure_rate,
                x="tenure_group",
                y="churn_rate",
                markers=True,
                labels={"tenure_group": "Tenure months", "churn_rate": "Churn rate"},
            ),
            use_container_width=True,
        )

        spend = data.groupby(["contract_type", "Churn"], as_index=False)["monthly_charges"].mean()
        st.plotly_chart(
            px.bar(
                spend,
                x="contract_type",
                y="monthly_charges",
                color="Churn",
                barmode="group",
                labels={"monthly_charges": "Average monthly charge", "contract_type": "Contract"},
                color_discrete_map={"Yes": "#d62828", "No": "#2a9d8f"},
            ),
            use_container_width=True,
        )

    with tab_model:
        left, right = st.columns([1, 1.4])
        matrix = pd.DataFrame(bundle.confusion, index=["Actual No", "Actual Yes"], columns=["Pred No", "Pred Yes"])
        left.dataframe(matrix, use_container_width=True)
        left.caption("Validation confusion matrix")
        left.metric("Decision threshold", f"{bundle.threshold:.2f}")

        importance = get_feature_importance(bundle.pipeline)
        right.plotly_chart(
            px.bar(
                importance,
                x="importance",
                y="feature",
                orientation="h",
                labels={"importance": "Importance", "feature": "Feature"},
                color_discrete_sequence=["#264653"],
            ).update_layout(yaxis={"categoryorder": "total ascending"}),
            use_container_width=True,
        )

        st.dataframe(
            validation.sort_values("churn_risk", ascending=False).head(25),
            use_container_width=True,
            hide_index=True,
        )
        st.download_button(
            "Download high-risk validation customers",
            validation.sort_values("churn_risk", ascending=False).to_csv(index=False),
            "high_risk_customers.csv",
            "text/csv",
            use_container_width=True,
        )

    with tab_predict:
        render_prediction_tool(bundle)


def main() -> None:
    st.set_page_config(page_title="Customer Churn Prediction", page_icon="chart_with_downwards_trend", layout="wide")
    st.title("Customer Churn Prediction System")
    st.caption("Random Forest churn scoring with customer risk insights and real-time prediction.")

    with st.sidebar:
        st.header("Data")
        uploaded = st.file_uploader("Upload customer CSV", type=["csv"])
        sample_size = st.slider("Synthetic records", min_value=1000, max_value=10000, value=5200, step=500)

    if uploaded:
        raw_data = pd.read_csv(uploaded)
        data = engineer_features(raw_data)
        if "Churn" not in data.columns:
            st.error("The uploaded CSV must include a Churn column with Yes/No values.")
            st.stop()
    else:
        data = generate_customer_data(sample_size)

    missing = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES + ["Churn"]) - set(data.columns)
    if missing:
        st.error(f"Dataset is missing required columns: {', '.join(sorted(missing))}")
        st.stop()

    with st.spinner("Training Random Forest model..."):
        bundle = train_model(data)

    render_dashboard(data, bundle)


if __name__ == "__main__":
    main()
