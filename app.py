import streamlit as st
import joblib
import numpy as np
import pandas as pd

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

gender = joblib.load("gender.pkl")
married = joblib.load("married.pkl")
dependents = joblib.load("dependents.pkl")
education = joblib.load("education.pkl")
self_emp = joblib.load("self_employed.pkl")
property_area = joblib.load("property.pkl")
employment_type = joblib.load("employment.pkl")

st.title("Loan Approval Prediction")

Gender = st.selectbox("Gender",["Male","Female"])
Married = st.selectbox("Married",["Yes","No"])
Dependents = st.selectbox("Dependents",["0","1","2","3+"])
Education = st.selectbox("Education",["Graduate","Not Graduate"])
Self_Employed = st.selectbox("Self Employed",["Yes","No"])
Property = st.selectbox("Property Area",["Urban","Semiurban","Rural"])
Employment = st.selectbox("Employment Type",["Private","Contract","Government"])

ApplicantIncome = st.number_input("Applicant Income")
CoapplicantIncome = st.number_input("Coapplicant Income")
LoanAmount = st.number_input("Loan Amount")
Loan_Amount_Term = st.number_input("Loan Term")
Credit_Score = st.number_input("Credit Score")
Age = st.number_input("Age")

Dependents = Dependents.replace("3+","3")

Gender = gender.transform([Gender])[0]
Married = married.transform([Married])[0]
Dependents = dependents.transform([Dependents])[0]
Education = education.transform([Education])[0]
Self_Employed = self_emp.transform([Self_Employed])[0]
Property = property_area.transform([Property])[0]
Employment = employment_type.transform([Employment])[0]

if st.button("Predict"):

    data = pd.DataFrame([{
        "ApplicantIncome": ApplicantIncome,
        "Gender": Gender,
        "CoapplicantIncome": CoapplicantIncome,
        "Married": Married,
        "LoanAmount": LoanAmount,
        "Dependents": Dependents,
        "Loan_Amount_Term": Loan_Amount_Term,
        "Education": Education,
        "Credit_Score": Credit_Score,
        "Self_Employed": Self_Employed,
        "Age": Age,
        "Property_Area": Property,
        "Employment_Type": Employment
    }])

    # Feature Engineering 
    data["Loan_Income_Ratio"] = data["LoanAmount"] / (data["ApplicantIncome"] + 1)
    data["Total_Income"] = data["ApplicantIncome"] + data["CoapplicantIncome"]
    data["Income_per_Dependent"] = data["Total_Income"] / (data["Dependents"] + 1)

    # Scaling
    data = scaler.transform(data)

    pred = model.predict(data)[0]

    if pred == 1:
        st.success("Loan Approved")
    else:
        st.error("Loan Rejected")