
import streamlit as st
import pandas as pd
import joblib

# --- load the trained pipeline (preprocessor + classifier) ---
@st.cache_resource
def load_model():
    return joblib.load("fraud_detection_pipeline.pkl")

model = load_model()

st.title("Fraud Detection Prediction App")
st.markdown("Please enter the transaction details and press **Predict**.")
st.divider()

# IMPORTANT: options must match what the model was trained on
transaction_type = st.selectbox(
    "Transaction Type",
    ["PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT"]  # trim to the ones you kept in training if needed
)

amount = st.number_input("Amount", min_value=0.0, value=1000.0, step=100.0)
oldbalanceOrg = st.number_input("Old Balance (Sender)", min_value=0.0, value=10000.0, step=100.0)
newbalanceOrig = st.number_input("New Balance (Sender)", min_value=0.0, value=9000.0, step=100.0)
oldbalanceDest = st.number_input("Old Balance (Receiver)", min_value=0.0, value=0.0, step=100.0)
newbalanceDest = st.number_input("New Balance (Receiver)", min_value=0.0, value=0.0, step=100.0)

if st.button("Predict"):
    # Build a single-row DataFrame with EXACT column names the model expects
    input_df = pd.DataFrame([{
        "type": transaction_type,
        "amount": amount,
        "oldbalanceOrg": oldbalanceOrg,
        "newbalanceOrig": newbalanceOrig,
        "oldbalanceDest": oldbalanceDest,
        "newbalanceDest": newbalanceDest
    }])

    pred = int(model.predict(input_df)[0])
    st.subheader(f"Prediction: {pred}")
    if pred == 1:
        st.error("This transaction may be fraud.")
    else:
        st.success("This transaction looks non-fraudulent.")

    # optional: show probability if the model supports it
    if hasattr(model, "predict_proba"):
        proba = float(model.predict_proba(input_df)[0, 1])
        st.caption(f"Estimated probability of fraud: {proba:.3f}")
