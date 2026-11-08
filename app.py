
import streamlit as st
import pandas as pd
import joblib
import os

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 Loan Approval Prediction System")
st.write(
    "Predict loan approval status using five trained "
    "machine learning classification models."
)

st.markdown("---")

# --------------------------------------------------
# LOAD TRAINED MODELS
# --------------------------------------------------

MODEL_PATH = "models"

model_files = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "K-Nearest Neighbors": "knn.pkl",
    "Gaussian Naive Bayes": "gaussian_naive_bayes.pkl",
    "Support Vector Machine (SVM)": "svm.pkl"
}

@st.experimental_singleton
def load_models():
    loaded_models = {}

    for model_name, filename in model_files.items():
        path = os.path.join(MODEL_PATH, filename)

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Model file not found: {path}"
            )

        loaded_models[model_name] = joblib.load(path)

    return loaded_models

try:
    trained_models = load_models()
    st.success("All five trained models loaded successfully.")

except Exception as e:
    st.error(f"Unable to load models: {e}")
    st.stop()

# --------------------------------------------------
# INPUT FEATURES
# --------------------------------------------------

required_features = [
    "person_age",
    "person_gender",
    "person_education",
    "person_income",
    "person_emp_exp",
    "person_home_ownership",
    "loan_amnt",
    "loan_intent",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
    "credit_score",
    "previous_loan_defaults_on_file"
]

# --------------------------------------------------
# MODEL SELECTION
# --------------------------------------------------

st.subheader("1. Select Machine Learning Model")

selected_model_name = st.selectbox(
    "Choose a classification algorithm:",
    list(model_files.keys())
)

selected_model = trained_models[selected_model_name]

# --------------------------------------------------
# CSV FILE UPLOAD
# --------------------------------------------------

st.subheader("2. Upload Applicant Data")

uploaded_file = st.file_uploader(
    "Upload a CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    try:
        input_df = pd.read_csv(uploaded_file)

        st.write("Uploaded Dataset Preview")
        st.dataframe(input_df.head())

        # Check required columns
        missing_columns = [
            col for col in required_features
            if col not in input_df.columns
        ]

        if missing_columns:
            st.error(
                "Missing required columns: "
                + ", ".join(missing_columns)
            )
            st.stop()

        # Separate actual target if available
        has_target = "loan_status" in input_df.columns

        X_input = input_df[required_features].copy()

        if X_input.isnull().any().any():
            st.error(
                "The uploaded data contains missing values. "
                "Please clean the data before uploading."
            )
            st.stop()

        # --------------------------------------------------
        # PREDICTION
        # --------------------------------------------------

        if st.button("Predict Loan Approval"):

            predictions = selected_model.predict(X_input)

            result_df = input_df.copy()

            result_df["Predicted Loan Status"] = [
                "Approved" if value == 1 else "Rejected"
                for value in predictions
            ]

            st.subheader("3. Prediction Results")

            st.dataframe(result_df)

            approved_count = int((predictions == 1).sum())
            rejected_count = int((predictions == 0).sum())

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Total Applicants",
                len(predictions)
            )

            col2.metric(
                "Predicted Approved",
                approved_count
            )

            col3.metric(
                "Predicted Rejected",
                rejected_count
            )

            # --------------------------------------------------
            # EVALUATION METRICS
            # --------------------------------------------------

            if has_target:

                y_true = input_df["loan_status"]

                if not y_true.isin([0, 1]).all():
                    st.error(
                        "loan_status must contain only 0 and 1."
                    )
                    st.stop()

                if y_true.nunique() < 2:
                    st.warning(
                        "Both target classes are required "
                        "to calculate all evaluation metrics."
                    )
                    st.stop()

                y_score = None

                if hasattr(selected_model, "predict_proba"):
                    y_score = selected_model.predict_proba(
                        X_input
                    )[:, 1]

                elif hasattr(selected_model, "decision_function"):
                    y_score = selected_model.decision_function(
                        X_input
                    )

                st.subheader("4. Model Evaluation Metrics")

                accuracy = accuracy_score(y_true, predictions)
                precision = precision_score(
                    y_true, predictions, zero_division=0
                )
                recall = recall_score(
                    y_true, predictions, zero_division=0
                )
                f1 = f1_score(
                    y_true, predictions, zero_division=0
                )
                mcc = matthews_corrcoef(y_true, predictions)

                auc_score = (
                    roc_auc_score(y_true, y_score)
                    if y_score is not None else None
                )

                c1, c2, c3 = st.columns(3)
                c4, c5, c6 = st.columns(3)

                c1.metric("Accuracy", f"{accuracy:.4f}")
                c2.metric("Precision", f"{precision:.4f}")
                c3.metric("Recall", f"{recall:.4f}")

                c4.metric("F1 Score", f"{f1:.4f}")
                c5.metric(
                    "ROC-AUC",
                    f"{auc_score:.4f}"
                    if auc_score is not None else "N/A"
                )
                c6.metric("MCC", f"{mcc:.4f}")

                # Confusion Matrix
                st.subheader("5. Confusion Matrix")

                cm = confusion_matrix(
                    y_true,
                    predictions,
                    labels=[0, 1]
                )

                cm_df = pd.DataFrame(
                    cm,
                    index=["Actual Rejected", "Actual Approved"],
                    columns=[
                        "Predicted Rejected",
                        "Predicted Approved"
                    ]
                )

                st.dataframe(cm_df)

                # Classification Report
                st.subheader("6. Classification Report")

                report = classification_report(
                    y_true,
                    predictions,
                    target_names=["Rejected", "Approved"],
                    output_dict=True,
                    zero_division=0
                )

                st.dataframe(
                    pd.DataFrame(report).transpose()
                )

            else:
                st.info(
                    "No loan_status column was provided. "
                    "Predictions are available, but evaluation "
                    "metrics cannot be calculated without "
                    "actual loan status values."
                )

            # Download results
            csv_results = result_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="Download Prediction Results",
                data=csv_results,
                file_name="loan_prediction_results.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(f"Error processing uploaded file: {e}")

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("---")

st.caption(
    "Loan Approval Prediction | Machine Learning "
    "Classification Project"
)
