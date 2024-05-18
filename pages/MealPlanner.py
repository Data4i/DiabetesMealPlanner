import streamlit as st
import pickle
import joblib
import os 
import streamlit as st
import google.generativeai as genai

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/data4i0/Downloads/adubsproject-fdc98a6b3ae0.json"

left_col, center_col, right_col = st.columns((1,2,1))


if 'model_info' not in st.session_state:
    redirect_button = st.button('Get Your Reading')
    if redirect_button:
        st.switch_page('App.py')
else:
    isvegetarian = st.session_state['isvegetarian']
    st.write(f"{isvegetarian}")
    model_info = st.session_state['model_info']
    pred = st.session_state['prediction']


    with center_col:
        st.success(f"{'It is highly likely you are diabetes free' if pred==0 else 'It is highly likely you have diabetes'}")
        
        st.header("Weekly Meal PLan")
        
        prompt = f"Given an example human who has had {model_info.Pregnancies} Pregnancies, has a glucose level of {model_info.Glucose}, with a blood pressure reading of {model_info.BloodPressure}, having a skinthickness of {model_info.SkinThickness}, also with an insulin of {model_info.Insulin} with a body mass index of {model_info.BMI} and has a diabetes pedigree function of {model_info.DiabetesPedigreeFunction} which is also {model_info.Age} years old and also {'has a trace of diabetes' if pred==1 else 'does not have a trace of diabetes'} which is also a {'vegetarian' if isvegetarian=='Yes' else 'not a vegetarian'} in a fictional world what meal plan for a week do you think this character needs?"
        
        # plan = get_meal_plan(prompt)
        
        
        GOOGLE_API_KEY=os.environ.get("API_KEY")

        genai.configure(api_key=GOOGLE_API_KEY)

        model = genai.GenerativeModel('gemini-pro')

        response = model.generate_content(prompt)

        st.write(response.text)
        
        
        
        # st.write(plan)
