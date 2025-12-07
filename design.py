import streamlit as st
import numpy as np
import joblib
import os
from sklearn.preprocessing import MinMaxScaler

# -------------------------------------------------------------
# PAGE SETTINGS
# -------------------------------------------------------------
st.set_page_config(
    page_title="Churn Prediction",
    page_icon="📊",
    layout="wide",
)

# -------------------------------------------------------------
# CUSTOM CSS
# -------------------------------------------------------------
st.markdown("""
    <style>
    .result-card {
        padding: 12px;
        border-radius: 10px;
        text-align: center;
        font-size: 18px;
        font-weight: 600;
        max-width: 330px;
        margin-left: auto;
        margin-right: auto;
    }
    .high-risk {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
    }
    .low-risk {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
    }
    .results-title {
        text-align: center;
        font-size: 30px;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    div[data-testid="stNumberInput"] input {
        max-width: 150px !important;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# LOAD MODEL
# -------------------------------------------------------------
if os.path.exists("best_model.pkl"):
    model = joblib.load("best_model.pkl")
else:
    st.error("❌ Model file 'best_model.pkl' not found!")
    st.stop()

# -------------------------------------------------------------
# PAGE HEADER
# -------------------------------------------------------------
st.markdown("<h1 style='text-align:center;'>📊 Customer Churn Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Predict whether a customer is likely to leave</p>", unsafe_allow_html=True)
st.markdown("---")

# -------------------------------------------------------------
# TWO MAIN COLUMNS
# -------------------------------------------------------------
left_col, right_col = st.columns(2)

# -------------------------------------------------------------
# LEFT COLUMN: CUSTOMER PROFILE + SERVICE
# -------------------------------------------------------------
with left_col:

    st.subheader("👤 Customer Profile")
    c1, c2 = st.columns(2)
    with c1:
        senior = st.radio("Senior Citizen", ["No", "Yes"])
    with c2:
        partner = st.radio("Has Partner", ["No", "Yes"])

    st.subheader("📞 Service Details")
    c3, c4 = st.columns(2)
    with c3:
        internet = st.radio("Internet Service", ["DSL", "Fiber optic", "No Internet"])
    with c4:
        contract = st.radio("Contract Type", ["Month-to-Month", "One Year", "Two Years"])

# -------------------------------------------------------------
# RIGHT COLUMN: BILLING + SUPPORT
# -------------------------------------------------------------
with right_col:

    st.subheader("💰 Billing Information")
    b1, b2 = st.columns(2)
    with b1:
        tenure = st.number_input("Tenure (months)", 0, 72, 12)
    with b2:
        monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0)

    total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, 1000.0)

    st.subheader("🎯 Support Services")
    s1, s2 = st.columns(2)
    with s1:
        online_security = st.radio("Online Security", ["No", "Yes"])
    with s2:
        tech_support = st.radio("Tech Support", ["No", "Yes"])

# -------------------------------------------------------------
# PREDICT BUTTON
# -------------------------------------------------------------
st.markdown("---")
btnL, btnC, btnR = st.columns([1, 2, 1])
with btnC:
    predict_btn = st.button("🔮 Predict Churn Risk", use_container_width=True)
st.markdown("---")

# -------------------------------------------------------------
# RESULTS SECTION
# -------------------------------------------------------------
st.markdown('<h2 class="results-title">📈 Prediction Results</h2>', unsafe_allow_html=True)

if not predict_btn:
    st.info("Fill the details above and click **Predict Churn Risk**.")
else:
    # ---------- ENCODE INPUTS ----------
    senior_val = 1 if senior == "Yes" else 0

    contract_map = {
        "Month-to-Month": [1, 0],
        "One Year": [0, 1],
        "Two Years": [0, 0]
    }
    payment_map = {
        "Electronic Check": [1, 0, 0],
        "Mailed Check": [0, 1, 0],
        "Bank Transfer": [0, 0, 1],
        "Credit Card": [0, 0, 0]
    }
    internet_map = {
        "Fiber optic": [1, 0],
        "DSL": [0, 0],
        "No Internet": [0, 1]
    }

    features = [
        0,  # gender (default)
        senior_val,
        tenure,
        monthly_charges,
        total_charges,
    ]
    features.extend(contract_map[contract])
    features.extend(payment_map["Electronic Check"])   # default payment
    features.extend(internet_map[internet])
    features.extend([
        1 if partner == "Yes" else 0,
        0,  # dependents
        0,  # phone_service
        0,  # multiple_lines
        1 if online_security == "Yes" else 0,
        0,  # online_backup
        0,  # device_protection
        1 if tech_support == "Yes" else 0,
        0,  # streaming_tv
        0,  # streaming_movies
        0   # paperless_billing
    ])

    # ---------- SCALE NUMERIC FEATURES ----------
    scaler = MinMaxScaler()
    scaler.fit(np.array([[0, 0, 0], [72, 200, 10000]]))  # tenure, monthly, total
    scaled_vals = scaler.transform([[features[2], features[3], features[4]]])[0]
    for i, idx in enumerate([2, 3, 4]):
        features[idx] = scaled_vals[i]

    input_data = np.array([features], dtype=np.float32)

    # ---------- PREDICT ----------
    try:
        pred_prob = model.predict_proba(input_data)[0][1]
        prediction = 1 if pred_prob > 0.5 else 0

        churn = pred_prob * 100
        stay = (1 - pred_prob) * 100

        # ---------- CENTERED RESULT CARD ----------
        # Outer columns to center the card
        card_left, card_mid, card_right = st.columns([1, 2, 1])
        with card_mid:
            if prediction == 1:
                st.markdown(
                    f"<div class='result-card high-risk'>⚠️ HIGH RISK<br>"
                    f"<span style='font-size:20px;'>{churn:.1f}%</span></div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"<div class='result-card low-risk'>✅ LOW RISK<br>"
                    f"<span style='font-size:20px;'>{stay:.1f}%</span></div>",
                    unsafe_allow_html=True
                )

        # ---------- CENTERED METRICS ROW ----------
        row_left, row_mid, row_right = st.columns([1, 2, 1])
        with row_mid:
            col_churn, col_stay = st.columns(2)
            with col_churn:
                st.metric("Churn Probability", f"{churn:.1f}%")
            with col_stay:
                st.metric("Stay Probability", f"{stay:.1f}%")

    except Exception as e:
        st.error(f"❌ Prediction error: {str(e)}")
