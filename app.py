import streamlit as st
import pandas as pd
import joblib

# Load model and encoder
model = joblib.load("model.pkl")
encoder = joblib.load("encoder.pkl")  # This is a dictionary of LabelEncoders

st.title("🧠 Adult Income Prediction App")
st.write("Enter details below to predict whether income is >50K or <=50K")

# User input
age = st.number_input("Age", min_value=17, max_value=90, value=30)
workclass = st.selectbox("Workclass", encoder['workclass'].classes_)
fnlwgt = st.number_input("Final Weight (fnlwgt)", min_value=10000, value=100000)
education = st.selectbox("Education", encoder['education'].classes_)
education_num = st.number_input("Education Number", min_value=1, max_value=20, value=10)
marital_status = st.selectbox("Marital Status", encoder['marital-status'].classes_)
occupation = st.selectbox("Occupation", encoder['occupation'].classes_)
relationship = st.selectbox("Relationship", encoder['relationship'].classes_)
race = st.selectbox("Race", encoder['race'].classes_)
sex = st.selectbox("Sex", encoder['sex'].classes_)
capital_gain = st.number_input("Capital Gain", min_value=0, value=0)
capital_loss = st.number_input("Capital Loss", min_value=0, value=0)
hours_per_week = st.number_input("Hours Per Week", min_value=1, max_value=100, value=40)
native_country = st.selectbox("Native Country", encoder['native-country'].classes_)

# Make dataframe
input_df = pd.DataFrame({
    'age': [age],
    'workclass': [workclass],
    'fnlwgt': [fnlwgt],
    'education': [education],
    'education-num': [education_num],
    'marital-status': [marital_status],
    'occupation': [occupation],
    'relationship': [relationship],
    'race': [race],
    'sex': [sex],
    'capital-gain': [capital_gain],
    'capital-loss': [capital_loss],
    'hours-per-week': [hours_per_week],
    'native-country': [native_country]
})

# Encode categorical features
categorical_cols = encoder.keys()
for col in categorical_cols:
    input_df[col] = encoder[col].transform(input_df[col])

# Predict
if st.button("Predict Income"):
    prediction = model.predict(input_df)[0]
    if prediction == 1:
        st.success("Predicted Income: >50K ✅")
    else:
        st.warning("Predicted Income: <=50K ❌")
