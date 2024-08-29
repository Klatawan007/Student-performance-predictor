import gradio as gr
import joblib
import numpy as np
import pandas as pd


# Load the model and unique brand values
model = joblib.load('model.joblib')
# Assuming unique_values contains the unique values for each categorical variable
unique_values = joblib.load('unique_values.joblib')

# Define the prediction function
def predict(age, gender, ethnicity, parental_education, study_time_weekly, absences, tutoring, parental_support, extracurricular, sports, music, volunteering):
    # Convert inputs to appropriate types
    age = int(age)
    gender = int(gender)
    ethnicity = int(ethnicity)
    parental_education = int(parental_education)
    study_time_weekly = float(study_time_weekly)
    absences = int(absences)
    tutoring = int(tutoring)
    parental_support = int(parental_support)
    extracurricular = int(extracurricular)
    sports = int(sports)
    music = int(music)
    volunteering = int(volunteering)

    # Prepare the input array for prediction
    input_data = pd.DataFrame({
        'Age': [age],
        'Gender': [gender],
        'Ethnicity': [ethnicity],
        'Parental Education': [parental_education],
        'Study Time Weekly': [study_time_weekly],
        'Absences': [absences],
        'Tutoring': [tutoring],
        'Parental Support': [parental_support],
        'Extracurricular': [extracurricular],
        'Sports': [sports],
        'Music': [music],
        'Volunteering': [volunteering]
    })

    # Perform the prediction
    prediction = model.predict(input_data)

    return prediction[0]

# Create the Gradio interface
interface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Textbox(label="Age"),
        gr.Textbox(label="Gender"),
        gr.Textbox(label="Ethnicity"),
        gr.Textbox(label="Parental Education"),
        gr.Textbox(label="Study Time Weekly"),
        gr.Textbox(label="Absences"),
        gr.Textbox(label="Tutoring"),
        gr.Textbox(label="Parental Support"),
        gr.Textbox(label="Extracurricular"),
        gr.Textbox(label="Sports"),
        gr.Textbox(label="Music"),
        gr.Textbox(label="Volunteering")
    ],
    outputs="text",
    title="Predictor",
    description="Enter the relevant information to predict the target value."
)

# Launch the app
interface.launch()