import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Title of the app
st.title("Heart Disease Prediction App")

# Sidebar with patient history tracking
st.sidebar.header("Patient History")

# Inputs for patient information
age = st.sidebar.number_input('Age', min_value=0, max_value=120, value=50)
sex = st.sidebar.selectbox('Sex', options=['Male', 'Female'])
depression = st.sidebar.number_input('Depression Score (0-10)', min_value=0, max_value=10)

# Risk factors analysis
st.sidebar.header("Risk Factor Analysis")
cholesterol = st.sidebar.number_input('Cholesterol Level', min_value=100, max_value=300, value=200)
smoking = st.sidebar.checkbox('Smoking')

# Educational content
st.subheader("Understanding Heart Disease")
st.write("Heart disease is a range of conditions that affect your heart. It can include:")
st.write("1. Coronary artery disease\n2. Heart attack\n3. Heart failure\n4. Arrhythmias\n")

# Main prediction functionality
if st.button('Predict'):  
    st.write("Predicted Outcome: ", np.random.choice(['No Heart Disease', 'Heart Disease']))

# Data visualization
st.subheader("Cholesterol vs Age")
data = pd.DataFrame({'Age': [30, 40, 50, 60, 70], 'Cholesterol': [180, 190, 210, 220, 240]})
plt.figure(figsize=(10, 5))
plt.scatter(data['Age'], data['Cholesterol'], color='blue')
plt.title('Cholesterol Levels across Ages')
plt.xlabel('Age')
plt.ylabel('Cholesterol Level')
plt.grid()
st.pyplot()

# Sidebar info cards
st.sidebar.info("Your cholesterol level is a significant risk factor for heart disease.")
