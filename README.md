# 📊 E-Commerce Customer Churn Analysis & Prediction

A Machine Learning project that analyzes e-commerce customer behavior and predicts whether a customer is likely to churn.

The project includes data cleaning, exploratory data analysis, visualization, machine learning model training, evaluation, feature importance analysis, and an interactive Streamlit web application.

---

## 🚀 Project Overview

Customer churn is an important problem for e-commerce businesses because losing existing customers can affect revenue and long-term growth.

This project uses historical customer behavior data to identify patterns related to churn and builds a Machine Learning model to predict customer churn.

The final application allows users to enter customer details and get:

- Customer churn prediction
- Churn probability
- Customer risk level
- Customer information summary

---

## 🎯 Objectives

- Understand customer behavior using data analysis
- Clean and preprocess the customer dataset
- Analyze important factors related to churn
- Visualize customer and churn patterns
- Train a Machine Learning classification model
- Evaluate model performance
- Build an interactive prediction application
- Provide a simple interface for customer churn prediction

---

## 🧰 Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical operations |
| Matplotlib | Data visualization |
| Seaborn | Data visualization |
| Scikit-learn | Machine Learning |
| Streamlit | Web application |
| Jupyter Notebook | Data analysis and experimentation |
| Git & GitHub | Version control |

---

## 📂 Project Structure

```text
ecommerce-customer-churn-analysis/
│
├── data/
│   ├── data_ecommerce_customer_churn.csv
│   └── processed/
│       └── cleaned_ecommerce_churn.csv
│
├── models/
│   ├── churn_model.pkl
│   └── feature_columns.pkl
│
├── notebooks/
│   └── ecommerce_churn_analysis.ipynb
│
├── app.py
├── check_data.py
├── requirements.txt
├── README.md
├── .gitignore
└── .gitattributes