
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

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
    /* Main Background adjustments if needed */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Result Card Styling */
    .result-card {
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        font-family: 'Arial', sans-serif;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-top: 20px;
        margin-bottom: 20px;
    }
    .high-risk {
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 99%, #fecfef 100%);
        color: #5c0002;
        border: 1px solid #ffc3c3;
    }
    .low-risk {
        background: linear-gradient(120deg, #d4fc79 0%, #96e6a1 100%);
        color: #005c0f;
        border: 1px solid #c3ffc9;
    }
    .card-header {
        font-size: 24px;
        font-weight: 700;
        margin-bottom: 10px;
        text-transform: uppercase;
    }
    .card-value {
        font-size: 48px;
        font-weight: 800;
    }
    .metric-container {
        display: flex;
        justify-content: center;
        gap: 20px;
        margin-top: 20px;
    }
    .metric-box {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 10px 20px;
        text-align: center;
        min-width: 120px;
        border: 1px solid #e9ecef;
    }
    .metric-label {
        font-size: 14px;
        color: #6c757d;
        margin-bottom: 5px;
    }
    .metric-number {
        font-size: 18px;
        font-weight: 600;
        color: #343a40;
    }
    
    /* Sidebar Styling */
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# LOAD ASSETS
# -------------------------------------------------------------
@st.cache_resource
def load_assets():
    model = None
    scaler = None
    cols = None
    
    try:
        if os.path.exists("best_model.pkl"):
            model = joblib.load("best_model.pkl")
        if os.path.exists("scaler.pkl"):
            scaler = joblib.load("scaler.pkl")
        if os.path.exists("model_columns.pkl"):
            cols = joblib.load("model_columns.pkl")
    except Exception as e:
        st.error(f"Error loading assets: {e}")
        
    return model, scaler, cols

model, scaler, model_columns = load_assets()

if model is None:
    st.error("❌ Model file 'best_model.pkl' not found. Please train the model first.")
    st.stop()

if scaler is None or model_columns is None:
    st.warning("⚠️ Scaler or Column metadata not found. Attempting legacy mode or manual fallback (Risky). Please run 'inspect_data.py' or 'main.py' to generate artifacts.")
    # Fallback to hardcoded list from my analysis if missing, but ideally they exist.
    # For now, we will proceed assuming they might exist, or error out if critical.

# -------------------------------------------------------------
# HEADER
# -------------------------------------------------------------
st.title("📊 Customer Churn Prediction")
st.markdown("Use the sidebar to input customer details and predict the likelihood of churn.")

# -------------------------------------------------------------
# SIDEBAR INPUTS
# -------------------------------------------------------------
with st.sidebar:
    st.header("👤 Customer Info")
    
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.radio("Senior Citizen", ["No", "Yes"], horizontal=True)
    partner = st.radio("Has Partner", ["No", "Yes"], horizontal=True)
    dependents = st.radio("Has Dependents", ["No", "Yes"], horizontal=True)
    
    st.header("📞 Services")
    phone_service = st.radio("Phone Service", ["No", "Yes"], horizontal=True)
    multiple_lines = st.radio("Multiple Lines", ["No", "Yes", "No phone service"])
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    
    if internet != "No":
        online_security = st.radio("Online Security", ["No", "Yes", "No internet service"])
        online_backup = st.radio("Online Backup", ["No", "Yes", "No internet service"])
        device_protection = st.radio("Device Protection", ["No", "Yes", "No internet service"])
        tech_support = st.radio("Tech Support", ["No", "Yes", "No internet service"])
        streaming_tv = st.radio("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.radio("Streaming Movies", ["No", "Yes", "No internet service"])
    else:
        # Default unavailable if no internet
        online_security = "No internet service"
        online_backup = "No internet service"
        device_protection = "No internet service"
        tech_support = "No internet service"
        streaming_tv = "No internet service"
        streaming_movies = "No internet service"

    st.header("💰 Account Info")
    contract = st.selectbox("Contract Type", ["Month-to-Month", "One Year", "Two Years"])
    paperless = st.radio("Paperless Billing", ["No", "Yes"], horizontal=True)
    payment = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    
    tenure = st.slider("Tenure (months)", 0, 72, 1) # Default to 1 (High Risk)
    monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0)
    total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, 70.0)
    
    predict_btn = st.button("🔮 Predict", type="primary", use_container_width=True)

# -------------------------------------------------------------
# MAIN PREDICTION LOGIC
# -------------------------------------------------------------
if predict_btn:
    # 1. Prepare Data Dictionary
    def bin_map(val):
        return 1 if val == "Yes" else 0

    input_data = {
        'gender': 1 if gender == "Female" else 0,
        'SeniorCitizen': bin_map(senior),
        'Partner': bin_map(partner),
        'Dependents': bin_map(dependents),
        'tenure': tenure,
        'PhoneService': bin_map(phone_service),
        'MultipleLines': 1 if multiple_lines == "Yes" else 0,
        'OnlineSecurity': 1 if online_security == "Yes" else 0,
        'OnlineBackup': 1 if online_backup == "Yes" else 0,
        'DeviceProtection': 1 if device_protection == "Yes" else 0,
        'TechSupport': 1 if tech_support == "Yes" else 0,
        'StreamingTV': 1 if streaming_tv == "Yes" else 0,
        'StreamingMovies': 1 if streaming_movies == "Yes" else 0,
        'PaperlessBilling': bin_map(paperless),
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
    }
    
    # Categorical dummies
    input_data['InternetService_Fiber optic'] = 1 if internet == "Fiber optic" else 0
    input_data['InternetService_No'] = 1 if internet == "No" else 0
    
    input_data['Contract_One year'] = 1 if contract == "One Year" else 0
    input_data['Contract_Two year'] = 1 if contract == "Two Years" else 0
    
    input_data['PaymentMethod_Credit card (automatic)'] = 1 if payment == "Credit card (automatic)" else 0
    input_data['PaymentMethod_Electronic check'] = 1 if payment == "Electronic check" else 0
    input_data['PaymentMethod_Mailed check'] = 1 if payment == "Mailed check" else 0
    
    # 2. Convert to DataFrame
    df_input = pd.DataFrame([input_data])
    
    # 3. Ensure Columns Match Model Expectation
    if model_columns:
        df_input = df_input.reindex(columns=model_columns, fill_value=0)
        
    # 4. Scale Data
    if scaler:
        cols_to_scale = ['tenure', 'SeniorCitizen', 'TotalCharges', 'MonthlyCharges']
        df_input[cols_to_scale] = scaler.transform(df_input[cols_to_scale])
        
    # Force float type to avoid warnings
    df_input = df_input.astype(float)

    # Debug Expander
    with st.expander("🛠️ Debug Info (Check this if result seems wrong)"):
        st.write("Model Columns:", model_columns)
        st.write("Input Data Stats:", df_input)
        st.write("Raw Input Dict:", input_data)
    
    # 5. Predict
    try:
        pred_prob = model.predict_proba(df_input)[0][1]

        prediction = 1 if pred_prob > 0.5 else 0
        
        churn_pct = pred_prob * 100
        stay_pct = (1 - pred_prob) * 100
        
        # 6. Display Results
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if prediction == 1:
                st.markdown(f"""
                    <div class="result-card high-risk">
                        <div class="card-header">⚠️ High Churn Risk</div>
                        <div class="card-value">{churn_pct:.1f}%</div>
                        <div>Probability of Churn</div>
                    </div>
                """, unsafe_allow_html=True)
                st.error("This customer is likely to churn. Consider offering incentives or support.")
            else:
                st.markdown(f"""
                    <div class="result-card low-risk">
                        <div class="card-header">✅ Low Churn Risk</div>
                        <div class="card-value">{churn_pct:.1f}%</div>
                        <div>Probability of Churn</div>
                    </div>
                """, unsafe_allow_html=True)
                st.success("This customer is likely to stay. Ensure continued satisfaction.")

        with col2:
            st.markdown("### Confidence")
            # Simple Matplotlib Pie Chart
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(2, 2))
            ax.pie([churn_pct, stay_pct], labels=['Churn', 'Stay'], colors=['#ff9999', '#66b3ff'], autopct='%1.1f%%', startangle=90)
            ax.axis('equal') 
            st.pyplot(fig, use_container_width=True, clear_figure=True)
            
    except Exception as e:
        st.error(f"Prediction Error: {e}")
        st.write("Debug - Input Data:", df_input)

else:
    # Initial State
    st.info("👈 Please configure the customer profile in the sidebar and click 'Predict'.")
    
    # Optional: Show Feature Importance or generic info
    st.markdown("### How it works")
    st.markdown("""
    This model analyzes customer demographics, services, and billing patterns to predict churn.
    
    **Key Factors often influencing churn:**
    - Example: High Monthly Charges
    - Example: Short-term Contracts
    - Example: Fiber Optic Service issues
    """)
