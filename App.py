import pickle
import os
import joblib
import streamlit as st

import numpy as np
import pandas as pd

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/data4i0/Downloads/adubsproject-fdc98a6b3ae0.json"

st.set_page_config(page_title='Diabetes Meal Planner', layout = 'wide', page_icon='🏥')
left_col, center_col, right_col = st.columns((1,2,1))

st.sidebar.success("Get Your Meal Plan")

with center_col:
    st.header('Diabetes Meal Plannner ❤️')

df_filename = "data/diabetes.csv"


model_filename = "best_model.pkl"

@st.cache_resource
def get_model(model_filename:str):
    with open(model_filename, "rb") as f:
        model = pickle.load(f)
    # model = pickle.load(open(model_filename, 'rb'))
    return model

model = get_model(model_filename)

@st.cache_data
def get_used_df(df_filename):
    return pd.read_csv(df_filename)

df = get_used_df(df_filename)

with center_col:
    pregnancies = st.slider('No Of Pregnancies', min_value=np.array(df.Pregnancies).min(), max_value=np.array(df.Pregnancies).max())
    glucose = st.number_input('Glucose Level', min_value=np.array(df.Glucose).min(), max_value=np.array(df.Glucose).max())
    age = st.number_input('Age', min_value=18, max_value=150)
    bp = st.number_input('Blood Pressure', min_value=np.array(df.BloodPressure).min(), max_value=np.array(df.BloodPressure).max())
    skinThickness = st.number_input('Skin Thickness', min_value=np.array(df.SkinThickness).min(), max_value=np.array(df.SkinThickness).max())
    insulin = st.number_input('Insulin', min_value=np.array(df.Insulin).min(), max_value=np.array(df.Insulin).max())
    bmi = st.number_input('BMI', min_value=np.array(df.BMI).min(), max_value=np.array(df.BMI).max())
    DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=np.array(df["DiabetesPedigreeFunction"]).max())
    vegetarian = st.selectbox("Vegetarian", options=['', 'Yes', 'No'])
    button = st.button('Get Weekly Meal Plan')

if button:
    model_prediction_infos = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": bp,
        "SkinThickness": skinThickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": DiabetesPedigreeFunction,
        "Age": age
    }
    
    model_prediction_info_df = pd.DataFrame(model_prediction_infos, index = [1], columns=df.columns[:-1])
    
    st.session_state['model_info'] = model_prediction_info_df
    
    pred = model.predict(model_prediction_info_df)
    
    st.session_state['prediction'] = pred
    
    st.session_state['isvegetarian'] = vegetarian
    
    # model_prediction_infos['vegetarian'] = vegetarian

    # if 'planner_infos' not in st.session_state:
    #     st.session_state['planner_infos'] = model_prediction_infos
        
    st.switch_page('pages/MealPlanner.py')    
    