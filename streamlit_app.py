import streamlit as st

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

def index():
    st.title('Home')
    st.write('Welcome to the Home page!')

def predict_datapoint():
    st.title('Predict Math Score')
    st.write('Enter the required information:')

    gender = st.selectbox("Gender", ["male", "female"])
    race_ethnicity = st.selectbox("Race/Ethnicity", ["group A", "group B", "group C", "group D", "group E"])
    parental_level_of_education = st.selectbox("Parental Education Level", ["some high school", "high school", "some college", "associate's degree", "bachelor's degree", "master's degree"])
    lunch = st.selectbox("Lunch", ["standard", "free/reduced"])
    test_preparation_course = st.selectbox("Test Preparation Course", ["none", "completed"])
    reading_score = st.number_input("Reading Score", min_value=0, max_value=100)
    writing_score = st.number_input("Writing Score", min_value=0, max_value=100)
    
    if st.button('Predict'):
        data = CustomData(
            gender=gender,
            race_ethnicity=race_ethnicity,
            parental_level_of_education=parental_level_of_education,
            lunch=lunch,
            test_preparation_course=test_preparation_course,
            reading_score=reading_score,
            writing_score=writing_score
        )
        pred_df = data.get_data_as_data_frame()
        st.write(pred_df)
        # st.write("Before Prediction")
        
        predict_pipeline = PredictPipeline()
        # st.write("Mid Prediction")
        results = predict_pipeline.predict(pred_df)
        st.write("Predicted Math Score is : ")
        
        st.write(results[0])

def main():
    st.sidebar.title('Data Science Project - 1 ')
    st.sidebar.button('Predict Math Score')
    app_mode = 'Predict Math Score'
    predict_datapoint()

if __name__ == "__main__":
    main()
