import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Load the trained model
with open('knn_model.joblib', 'rb') as model_file:
    model = pickle.load(model_file)

# Function for predictions
def predict(input_data):
    result = model.predict([input_data])
    return result

# UI configuration
st.set_page_config(page_title='Heart Disease Prediction', layout='wide')

# Creating Tabs
tabs = st.tabs(['Prediction', 'Analytics', 'Education', 'About'])

# Prediction Tab
with tabs[0]:
    st.header('Heart Disease Prediction')
    age = st.number_input('Age:', min_value=0, max_value=120)
    gender = st.selectbox('Gender:', ['Male', 'Female'])
    cholesterol = st.selectbox('Cholesterol Level:', ['Normal', 'Above Normal'])
    # Additional input fields here...  
    input_data = [age, gender, cholesterol]  # Add more features as needed
    if st.button('Predict'):
        prediction = predict(input_data)
        st.success(f'Prediction: {prediction[0]}')

# Analytics Tab
with tabs[1]:
    st.header('Data Analytics')
    data = pd.read_csv('heart_data.csv')  # Load your dataset
    st.write(data.describe())
    fig, ax = plt.subplots()
    sns.countplot(data=data, x='target', ax=ax)
    st.pyplot(fig)

# Education Tab
with tabs[2]:
    st.header('Education')
    st.write('Heart Disease Overview:')
    st.write('Heart disease affects millions of people...')  # More content here

# About Tab
with tabs[3]:
    st.header('About this Application')
    st.write('This application uses a trained model to predict heart disease...')  # Add details about the app
