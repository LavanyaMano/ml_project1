import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
import streamlit as st

## Route for the home page
def index():
    return st.markdown('<h1 style="text-align: center;">Home Page</h1>', unsafe_allow_html=True)

def main():
    st.title("Predict Data Point")

    gender = st.selectbox("Gender", ["male", "female"])
    ethnicity = st.selectbox("Race/Ethnicity", ["group A", "group B", "group C", "group D", "group E"])
    parental_level_of_education = st.selectbox("Parental Education Level", ["some high school", "high school", "some college", "associate's degree", "bachelor's degree", "master's degree"])
    lunch = st.selectbox("Lunch", ["standard", "free/reduced"])
    test_preparation_course = st.selectbox("Test Preparation Course", ["none", "completed"])
    reading_score = st.text_input("Reading Score")
    writing_score = st.text_input("Writing Score")

    if st.button("Predict"):
        data = CustomData(
            gender=gender,
            race_ethnicity=ethnicity,
            parental_level_of_education=parental_level_of_education,
            lunch=lunch,
            test_preparation_course=test_preparation_course,
            reading_score=float(reading_score),
            writing_score=float(writing_score)
        )

        pred_df = data.get_data_as_data_frame()
        st.write("Before Prediction")
        predict_pipeline = PredictPipeline()
        st.write("Mid Prediction")
        results = predict_pipeline.predict(pred_df)
        st.write("After Prediction")
        st.write("Results:", results[0])


if __name__ == "__main__":
    main()
