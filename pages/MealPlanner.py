import streamlit as st
import pickle
import os 
import streamlit as st
import google.generativeai as genai
from IPython.display import display

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/data4i0/Downloads/adubsproject-fdc98a6b3ae0.json"

left_col, center_col, right_col = st.columns((1,2,1))


if 'model_info' not in st.session_state:
    redirect_button = st.button('Get Your Reading')
    if redirect_button:
        st.switch_page('App.py')
else:
    model_info = st.session_state['model_info']

    model_filename = "models/best_model.pkl"

    @st.cache_resource
    def get_model(model_filename:str):
        model = pickle.load(open(model_filename, 'rb'))
        return model

    model = get_model(model_filename)


    with center_col:
        pred = model.predict(model_info)
        st.success(f"{'It is highly likely you are diabetes free' if pred==0 else 'It is highly likely you have diabetes'}")
        
        st.header("Weekly Meal PLan")
        
        prompt = f"Given an example human who has had {model_info.Pregnancies} Pregnancies, has a glucose level of {model_info.Glucose}, with a blood pressure reading of {model_info.BloodPressure}, having a skinthickness of {model_info.SkinThickness}, also with an insulin of {model_info.Insulin} with a body mass index of {model_info.BMI} and has a diabetes pedigree function of {model_info.DiabetesPedigreeFunction} which is also {model_info.Age} years old and also {'has a trace of diabetes' if pred==1 else 'does not have a trace of diabetes'} in a fictional world what meal plan for a week do you think this character needs?"
        
        # plan = get_meal_plan(prompt)
        
        
        GOOGLE_API_KEY=os.environ.get("API_KEY")

        genai.configure(api_key=GOOGLE_API_KEY)

        model = genai.GenerativeModel('gemini-pro')

        response = model.generate_content(prompt)

        st.write(response.text)
        
        
        
        # st.write(plan)
