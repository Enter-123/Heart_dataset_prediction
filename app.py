import pandas as pd
import numpy as np
import streamlit as st
import joblib 

st.title('Heart Ananlysis Dataset')

st.header('Enter necessary Data : ')

scaler = joblib.load('knn_model.joblib')
classifier = joblib.load('classifier.joblib')

def predict_target(d):
    sample_data = pd.DataFrame([d])
    scaled_data = scaler.transform(sample_data)
    pred = classifier.predict(scaled_data)[0]
    prob = np.max(classifier.predict_proba(scaled_data)[0])
    return pred,prob


age = st.slider('Enter Your Age : ',min_value=1,max_value=100,step=1,value=50)
sex = st.selectbox('Enter Your Gender : ',['male','female'])
cp = st.selectbox('Enter Your Type of Chest Pain : ',['Typical angina','Atypical','Non-anginal','Asymptomatic'])
trestbps = st.number_input('Enter Blood Pressure Sugar : ',min_value=94,max_value=220,step=1,value=150)
chol = st.slider('Enter Chlorostrol : ',min_value=120,max_value=600,step=1,value=250)
fbs = st.selectbox('Enter Fasting Blood Sugar  : ',['Yes','No'])
restecg = st.selectbox('Enter Resting electrocardiographic',[0,1,2])
thalach = st.number_input('Enter Max Heart Rate : ',min_value=70,max_value=220,step=1,value=200)
exang = st.selectbox('Do chest pain during Exercise : ',['Yes','No'])
oldpeak = st.slider('Enter ST depression : ',min_value=0.0,max_value=6.5,step=0.1,value=5.5)
slope = st.selectbox('Enter ST segment/heart rate : ',['Upsloping','Flat','Downsloping'])
ca = st.number_input("Select your CAD : ",min_value=0,max_value=4,step=1,value=3)
thal = st.selectbox('Enter type of thalassemia : ',['Normal','Fixed','Reversible','Alpha'])


sex_map = {'male':0,'female':1}
cp_map = {'Typical angina':0,'Atypical':1,'Non-anginal':2,'Asymptomatic':3,}
fbs_map = {'Yes':1,'No':0}
exang_map = {'Yes':1,'No':0}
slope_map = {'Upsloping':0,'Flat':1,'Downsloping':2}
thal_map = {'Normal':0,'Fixed':1,'Reversible':2,'Alpha':3}

input_data = {
        'age':age,
        'sex':sex_map[sex],
        'cp':cp_map[cp],
        'trestbps':trestbps,
        'chol':chol,
        'fbs':fbs_map[fbs],
        'restecg':restecg,
        'thalach':thalach,
        'exang':exang_map[exang],
        'oldpeak':oldpeak,
        'slope':slope_map[slope],
        'ca':ca,
        'thal':thal_map[thal]
    }

if st.button('Predict Target'):
    with st.spinner('Generating Result .....'):
        pred,prob = predict_target(input_data)

        if pred == 1:
            st.error(f'Patients has Heart Dieases with Probability of {prob:.2%}')
        else:
            st.success(f'Patients Do Not has Heart Deisease with the probability of {prob:.2%}')

