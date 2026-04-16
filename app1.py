import streamlit as st
import numpy as np
import pickle

# ── Load model & scaler ─────────────────────────────────────────────────
model  = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# ── Page config ─────────────────────────────────────────────────────────
st.set_page_config(page_title="Loan Eligibility Dashboard", layout="wide")

# ── Custom CSS ──────────────────────────────────────────────────────────
st.markdown("""
<style>

/* MAIN BACKGROUND */
[data-testid="stAppViewContainer"]{
    background: linear-gradient(135deg,#141E30,#243B55);
    color:white;
}

/* REMOVE WHITE BLOCKS */
[data-testid="stHeader"]{
    background: rgba(0,0,0,0);
}

/* TITLE */
.main-title{
    font-size:42px;
    font-weight:bold;
    color:white;
    text-align:center;
    margin-bottom:20px;
}

/* SUBTITLE */
.sub-text{
    text-align:center;
    color:#d1d1d1;
    margin-bottom:30px;
}

/* CARDS */
.card{
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(12px);
    padding:25px;
    border-radius:20px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    margin-bottom:20px;
}

/* BUTTON */
.stButton>button{
    width:100%;
    background: linear-gradient(90deg,#00C9FF,#92FE9D);
    color:black;
    font-weight:bold;
    border:none;
    padding:14px;
    border-radius:12px;
    font-size:18px;
}

.stButton>button:hover{
    transform:scale(1.02);
}

/* INPUT LABELS */
label{
    color:white !important;
    font-weight:bold;
}

</style>
""", unsafe_allow_html=True)

# ── Header ──────────────────────────────────────────────────────────────
st.markdown('<p class="big-font">🏦 Loan Eligibility Dashboard</p>', unsafe_allow_html=True)

# ── Metric cards ────────────────────────────────────────────────────────
col1, col2 = st.columns(2)
with col1:
    st.markdown('<div class="card"><h3>📊 Model Accuracy</h3><p>~83%</p></div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="card"><h3>🤖 Algorithm Used</h3><p>Gaussian Naive Bayes</p></div>', unsafe_allow_html=True)

# ── Input form ──────────────────────────────────────────────────────────
st.markdown("## 📝 Enter Applicant Details")

col3, col4, col5 = st.columns(3)

with col3:
    gender          = st.selectbox("Gender",           ["Male", "Female"])
    married         = st.selectbox("Married",          ["Yes", "No"])
    dependents      = st.selectbox("Dependents",       ["0", "1", "2", "3+"])
    education       = st.selectbox("Education",        ["Graduate", "Not Graduate"])

with col4:
    applicant_income    = st.number_input("Applicant Monthly Income (₹)",  min_value=0, value=5000)
    coapplicant_income  = st.number_input("Co-Applicant Monthly Income (₹)", min_value=0, value=0)
    loan_amount         = st.number_input("Loan Amount (in thousands ₹)",  min_value=1, value=100)
    loan_term           = st.number_input("Loan Amount Term (in days)",    min_value=1, value=360)

with col5:
    credit          = st.selectbox("Credit History",  [1.0, 0.0],
                                   format_func=lambda x: "Good (1.0)" if x == 1.0 else "Bad (0.0)")
    property_area   = st.selectbox("Property Area",   ["Urban", "Semiurban", "Rural"])

# ── Prediction ──────────────────────────────────────────────────────────
if st.button("🔍 Predict Loan Status"):

    # ── Step 1: Encode categorical inputs ───────────────────────────────
    gender_enc       = 1 if gender    == "Male"        else 0
    married_enc      = 1 if married   == "Yes"         else 0
    education_enc    = 1 if education == "Graduate"    else 0
    dependents_enc   = 3 if dependents == "3+"         else int(dependents)
    property_enc     = {"Urban": 2, "Semiurban": 1, "Rural": 0}[property_area]

    # ── Step 2: Compute derived features (MUST match notebook) ──────────
    #   Notebook trains on: LoanAmount_log = log(LoanAmount)
    #                       TotalIncome_log = log(ApplicantIncome + CoapplicantIncome)
    loan_amount_log  = np.log(loan_amount)   if loan_amount > 0 else 0.0
    total_income     = applicant_income + coapplicant_income
    total_income_log = np.log(total_income)  if total_income > 0 else 0.0

    # ── Step 3: Assemble feature vector in EXACT training order ─────────
    #   Order: Gender, Married, Dependents, Education,
    #          Loan_Amount_Term, Credit_History, LoanAmount_log,
    #          TotalIncome_log, Property_Area
    data = np.array([[
        gender_enc,
        married_enc,
        dependents_enc,
        education_enc,
        loan_term,           # Loan_Amount_Term
        credit,              # Credit_History
        loan_amount_log,     # log(LoanAmount)  ← was raw amount before (BUG #1)
        total_income_log,    # log(TotalIncome) ← was missing entirely (BUG #2)
        property_enc,        # Property_Area    ← was at wrong position (BUG #3)
    ]])

    # ── Step 4: Scale using the same scaler fitted during training ───────
    data_scaled = scaler.transform(data)

    # ── Step 5: Predict ─────────────────────────────────────────────────
    prediction = model.predict(data_scaled)
    probability = model.predict_proba(data_scaled)[0]

    st.markdown("---")
    if prediction[0] == 1:
        st.success("✅ Loan Approved")
        st.info(f"Approval confidence: **{probability[1]*100:.1f}%**")
    else:
        st.error("❌ Loan Rejected")
        st.info(f"Rejection confidence: **{probability[0]*100:.1f}%**")

    # ── Debug panel (expandable) ─────────────────────────────────────────
    with st.expander("🔎 View Processed Input"):
        import pandas as pd
        debug_df = pd.DataFrame({
            "Feature":      ["Gender", "Married", "Dependents", "Education",
                             "Loan_Amount_Term", "Credit_History",
                             "LoanAmount_log", "TotalIncome_log", "Property_Area"],
            "Raw Input":    [gender, married, dependents, education,
                             loan_term, credit, loan_amount, total_income, property_area],
            "Encoded/Transformed": [gender_enc, married_enc, dependents_enc, education_enc,
                                    loan_term, credit, round(loan_amount_log, 4),
                                    round(total_income_log, 4), property_enc]
        })
        st.dataframe(debug_df, use_container_width=True)
