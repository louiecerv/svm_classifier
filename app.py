import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay
import data_generator

# Load dataset from CSV
business_data = pd.read_csv("business_data.csv")
X = business_data.iloc[:, :-1].values
y = business_data.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Business Problem Description
"""
This simulated dataset represents a business classification problem where a company is trying to categorize customer behaviors
into two distinct segments. The classification is based on factors such as purchase history, engagement levels, and
customer loyalty indicators. The data is structured in a way that requires a non-linear classification approach, making it
an ideal case for Support Vector Machines with polynomial or RBF kernels.
"""

# Streamlit App
st.title("SVM Business Classification App")
st.sidebar.header("Model Hyperparameters")
C = st.sidebar.slider("Regularization (C)", 0.01, 10.0, 1.0)
epsilon = st.sidebar.slider("Epsilon", 0.01, 1.0, 0.1)

# Tabs for different kernel types
tab1, tab2, tab3 = st.tabs(["Linear Kernel", "Polynomial Kernel", "RBF Kernel"])

def train_and_evaluate(kernel, degree=3, gamma='scale'):
    model = SVC(C=C, kernel=kernel, degree=degree, gamma=gamma)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    return model, accuracy, report, y_pred

# Linear Kernel
with tab1:
    st.subheader("Linear Kernel")
    model, acc, report, y_pred = train_and_evaluate("linear")
    st.write(f"**Accuracy:** {acc:.2f}")
    st.write("**Classification Report:**", pd.DataFrame(report).transpose())
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, ax=ax)
    st.pyplot(fig)

# Polynomial Kernel
with tab2:
    st.subheader("Polynomial Kernel")
    degree = st.slider("Polynomial Degree", 2, 5, 3)
    model, acc, report, y_pred = train_and_evaluate("poly", degree)
    st.write(f"**Accuracy:** {acc:.2f}")
    st.write("**Classification Report:**", pd.DataFrame(report).transpose())
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, ax=ax)
    st.pyplot(fig)

# RBF Kernel
with tab3:
    st.subheader("RBF Kernel")
    gamma = st.slider("Gamma", 0.01, 1.0, 0.1)
    model, acc, report, y_pred = train_and_evaluate("rbf", gamma=gamma)
    st.write(f"**Accuracy:** {acc:.2f}")
    st.write("**Classification Report:**", pd.DataFrame(report).transpose())
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, ax=ax)
    st.pyplot(fig)

st.write("This app demonstrates how different SVM kernels impact classification performance in a non-linear business problem.")
