import streamlit as st
import pandas as pd
import numpy as np
import pickle

import warnings


st.write("""
# Diabetes Prediction App

This app predicts the likelihood of a patient having **Diabetes** based on several input features before been subjected to an actual test if the likelihood is high!

""")


# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        Pregnancies = st.sidebar.slider('Pregnancies', 0,10,2)
        PlasmaGlucose = st.sidebar.slider('PlasmaGlucose', 30,300,180)
        DiastolicBloodPressure = st.sidebar.slider('DiastolicBloodPressure', 50,150,74)
        TricepsThickness = st.sidebar.slider('TricepsThickness', 20,35,24)
        SerumInsulin = st.sidebar.slider('SerumInsulin', 15,30,21)
        BMI = st.sidebar.slider('BMI', 15.0,30.0,23.9)
        DiabetesPedigree = st.sidebar.slider('DiabetesPedigree', 0.5,5.0,1.5)
        Age = st.sidebar.slider('Age', 20,75,35)
        data = {'Pregnancies': Pregnancies,
                'PlasmaGlucose': PlasmaGlucose,
                'DiastolicBloodPressure': DiastolicBloodPressure,
                'TricepsThickness': TricepsThickness,
                'SerumInsulin': SerumInsulin,
                'BMI': BMI,
                'DiabetesPedigree': DiabetesPedigree,
                'Age': Age
                }
        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()


# Combines user input features with entire diabetes dataset
diabetes_raw = pd.read_csv('diabetes.csv')
diabetes = diabetes_raw.drop(columns=['PatientID','Diabetic'])
df = pd.concat([input_df,diabetes],axis=0)
df = df[:1] # Selects only the first row (the user input data)


# Displays the user input features
st.subheader('User Input features')

if uploaded_file is not None:
    st.write(df)
else:
    st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
    st.write(df)

X_new = np.array([[2,180,74,24,21,23.9091702,1.488172308,22]])


# Reads in saved classification model
model = pickle.load(open('model2_dia_ran.pkl', 'rb'))

# Apply model to make predictions
prediction = model.predict(df)
prediction_proba = model.predict_proba(df)


st.subheader('Prediction')
diabetes_class = np.array(['Non-diabetic','Diabetic'])
st.write(diabetes_class[prediction])

st.subheader('Prediction Probability')
st.write(prediction_proba)