import streamlit as st
import pandas as pd
import pickle


# Load model
with open("models/churn_model.pkl", "rb") as file:
    model = pickle.load(file)

# Load feature columns
with open("models/feature_columns.pkl", "rb") as file:
    feature_columns = pickle.load(file)

# Load cleaned dataset
df = pd.read_csv("data/processed/cleaned_ecommerce_churn.csv")


# Page settings
st.set_page_config(
    page_title="E-Commerce Churn Analytics",
    page_icon="📊",
    layout="wide"
)


# Custom CSS
st.markdown("""
<style>

html, body, [class*="css"] {
    font-family: Arial, sans-serif;
}

[data-testid="stAppViewContainer"] {
    background-color: #f5f7fb;
}

[data-testid="stHeader"] {
    background-color: #f5f7fb;
}

.block-container {
    max-width: 1250px;
    padding-top: 30px;
    padding-bottom: 50px;
}

/* Main text */
h1, h2, h3, h4, p, label {
    color: #182230 !important;
}

/* Header */
.header-box {
    background: linear-gradient(135deg, #172554, #2563eb);
    padding: 30px 35px;
    border-radius: 18px;
    margin-bottom: 25px;
    box-shadow: 0 8px 25px rgba(37, 99, 235, 0.15);
}

.header-box h1 {
    color: white !important;
    font-size: 38px;
    margin: 0;
}

.header-box p {
    color: #dbeafe !important;
    margin-top: 8px;
    font-size: 16px;
}

/* KPI cards */
.kpi {
    background: white;
    padding: 20px;
    border-radius: 14px;
    border: 1px solid #e5e7eb;
    box-shadow: 0 4px 15px rgba(0,0,0,0.05);
}

.kpi-title {
    color: #64748b !important;
    font-size: 14px;
}

.kpi-value {
    color: #111827 !important;
    font-size: 28px;
    font-weight: bold;
    margin-top: 5px;
}

/* Section */
.section {
    color: #172554 !important;
    font-size: 24px;
    font-weight: bold;
    margin-top: 30px;
    margin-bottom: 15px;
}

/* Information box */
.info-box {
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    border-left: 5px solid #2563eb;
    padding: 18px;
    border-radius: 10px;
    color: #1e3a8a !important;
    margin-bottom: 20px;
}

/* Prediction cards */
.prediction-safe {
    background: #ecfdf5;
    border: 1px solid #86efac;
    padding: 25px;
    border-radius: 15px;
    text-align: center;
}

.prediction-risk {
    background: #fef2f2;
    border: 1px solid #fca5a5;
    padding: 25px;
    border-radius: 15px;
    text-align: center;
}

.prediction-title {
    color: #64748b !important;
    font-size: 14px;
}

.prediction-value {
    font-size: 28px;
    font-weight: bold;
    margin-top: 6px;
}

.footer {
    text-align: center;
    color: #64748b !important;
    font-size: 13px;
    padding-top: 15px;
}

</style>
""", unsafe_allow_html=True)


# Sidebar
with st.sidebar:

    st.markdown("## 📊 E-Commerce Analytics")

    st.write("Customer Churn Prediction System")

    st.markdown("---")

    st.markdown("### 🤖 Model")
    st.write("Random Forest Classifier")

    st.markdown("### 🎯 Accuracy")
    st.write("90.98%")

    st.markdown("### 📁 Dataset")
    st.write("3,270 cleaned customers")

    st.markdown("---")

    st.markdown("### 🛠️ Technologies")
    st.write("Python")
    st.write("Pandas")
    st.write("Scikit-learn")
    st.write("Streamlit")

    st.markdown("---")
    st.caption("Data Analytics & AI Project")


# Header
st.markdown("""
<div class="header-box">
    <h1>📊 E-Commerce Customer Churn Analytics</h1>
    <p>
        Analyze customer behaviour and predict the likelihood of customer churn
        using Machine Learning.
    </p>
</div>
""", unsafe_allow_html=True)


# KPI calculations
total_customers = len(df)
churned_customers = int(df["Churn"].sum())
stayed_customers = total_customers - churned_customers
churn_rate = (churned_customers / total_customers) * 100
avg_cashback = df["CashbackAmount"].mean()


# KPI section
st.markdown(
    '<div class="section">📌 Business Overview</div>',
    unsafe_allow_html=True
)

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(f"""
    <div class="kpi">
        <div class="kpi-title">Total Customers</div>
        <div class="kpi-value">{total_customers:,}</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="kpi">
        <div class="kpi-title">Churned Customers</div>
        <div class="kpi-value">{churned_customers:,}</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="kpi">
        <div class="kpi-title">Churn Rate</div>
        <div class="kpi-value">{churn_rate:.2f}%</div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="kpi">
        <div class="kpi-title">Average Cashback</div>
        <div class="kpi-value">₹{avg_cashback:.2f}</div>
    </div>
    """, unsafe_allow_html=True)


# Analytics section
st.markdown(
    '<div class="section">📈 Customer Analytics</div>',
    unsafe_allow_html=True
)

chart1, chart2 = st.columns(2)


with chart1:

    st.markdown("### Customer Churn Distribution")

    churn_chart = pd.DataFrame({
        "Customer Status": ["Stayed", "Churned"],
        "Customers": [stayed_customers, churned_customers]
    })

    st.bar_chart(
        churn_chart.set_index("Customer Status")
    )


with chart2:

    st.markdown("### Churn by Order Category")

    category_churn = (
        df.groupby("PreferedOrderCat")["Churn"]
        .mean()
        .mul(100)
        .sort_values(ascending=False)
        .round(2)
    )

    st.bar_chart(category_churn)


# Feature importance
st.markdown(
    '<div class="section">🤖 Important Churn Factors</div>',
    unsafe_allow_html=True
)

importance = pd.Series(
    model.feature_importances_,
    index=feature_columns
).sort_values(ascending=False)

top_features = importance.head(8).sort_values(ascending=True)

st.bar_chart(top_features)


# Prediction
st.markdown(
    '<div class="section">🔮 Customer Churn Prediction</div>',
    unsafe_allow_html=True
)

st.markdown("""
<div class="info-box">
<b>How to use:</b><br>
Enter the customer's details below and click <b>Predict Customer Churn</b>.
The trained Random Forest model will estimate the probability of churn.
</div>
""", unsafe_allow_html=True)


# Customer details
st.markdown("### 👤 Customer Information")

col1, col2, col3 = st.columns(3)

with col1:
    tenure = st.number_input(
        "Tenure (Months)",
        min_value=0.0,
        max_value=100.0,
        value=10.0
    )

with col2:
    devices = st.number_input(
        "Registered Devices",
        min_value=1,
        max_value=10,
        value=3
    )

with col3:
    addresses = st.number_input(
        "Number of Addresses",
        min_value=1,
        max_value=20,
        value=3
    )


col1, col2 = st.columns(2)

with col1:
    marital = st.selectbox(
        "Marital Status",
        ["Divorced", "Married", "Single"]
    )

with col2:
    satisfaction = st.slider(
        "Satisfaction Score",
        1,
        5,
        3
    )


# Order details
st.markdown("### 🛒 Order Behaviour")

col1, col2, col3 = st.columns(3)

with col1:
    category = st.selectbox(
        "Preferred Order Category",
        [
            "Fashion",
            "Grocery",
            "Laptop & Accessory",
            "Mobile",
            "Mobile Phone",
            "Others"
        ]
    )

with col2:
    warehouse = st.number_input(
        "Warehouse to Home",
        min_value=0.0,
        max_value=100.0,
        value=15.0
    )

with col3:
    days_last_order = st.number_input(
        "Days Since Last Order",
        min_value=0.0,
        max_value=100.0,
        value=5.0
    )


# Behaviour details
st.markdown("### 📦 Customer Behaviour")

col1, col2 = st.columns(2)

with col1:
    complain = st.selectbox(
        "Customer Complaint",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

with col2:
    cashback = st.number_input(
        "Cashback Amount",
        min_value=0.0,
        max_value=1000.0,
        value=150.0
    )


st.markdown("")


# Prediction button
predict = st.button(
    "🔮 Predict Customer Churn",
    use_container_width=True
)


if predict:

    # Create input data
    customer = pd.DataFrame({
        "Tenure": [tenure],
        "WarehouseToHome": [warehouse],
        "NumberOfDeviceRegistered": [devices],
        "PreferedOrderCat": [category],
        "SatisfactionScore": [satisfaction],
        "MaritalStatus": [marital],
        "NumberOfAddress": [addresses],
        "Complain": [complain],
        "DaySinceLastOrder": [days_last_order],
        "CashbackAmount": [cashback]
    })


    # Convert categorical values
    customer = pd.get_dummies(
        customer,
        drop_first=True
    )


    # Match training columns
    customer = customer.reindex(
        columns=feature_columns,
        fill_value=0
    )


    # Model prediction
    prediction = model.predict(customer)[0]
    probability = model.predict_proba(customer)[0][1]

    probability_percent = probability * 100


    # Result
    st.markdown("---")

    st.markdown(
        '<div class="section">🎯 Prediction Result</div>',
        unsafe_allow_html=True
    )


    result1, result2 = st.columns(2)


    with result1:

        if prediction == 1:

            st.markdown(f"""
            <div class="prediction-risk">
                <div class="prediction-title">Customer Status</div>
                <div class="prediction-value" style="color:#b91c1c;">
                    ⚠️ Likely to Churn
                </div>
            </div>
            """, unsafe_allow_html=True)

        else:

            st.markdown(f"""
            <div class="prediction-safe">
                <div class="prediction-title">Customer Status</div>
                <div class="prediction-value" style="color:#15803d;">
                    ✅ Likely to Stay
                </div>
            </div>
            """, unsafe_allow_html=True)


    with result2:

        st.markdown(f"""
        <div class="kpi">
            <div class="kpi-title">Churn Probability</div>
            <div class="kpi-value">{probability_percent:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)


    st.markdown("### 📊 Churn Probability")

    st.progress(float(probability))


    # Risk level
    if probability >= 0.70:

        risk = "High Risk"
        message = (
            "The model predicts a high probability of churn. "
            "The business can consider targeted customer retention actions."
        )

    elif probability >= 0.40:

        risk = "Medium Risk"
        message = (
            "The customer has a moderate predicted churn probability. "
            "The customer can be monitored for further changes."
        )

    else:

        risk = "Low Risk"
        message = (
            "The customer has a lower predicted churn probability "
            "according to the trained model."
        )


    st.markdown(f"### Risk Level: **{risk}**")

    st.info(message)


# Footer
st.markdown("---")

st.markdown("""
<div class="footer">
    E-Commerce Customer Churn Analytics • Machine Learning Project
</div>
""", unsafe_allow_html=True)