import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

# from predict import predict

if (__name__ == "__main__"):

    html_temp = """
    <div style="background-color:yellow;padding:10px">
    <h2 style="color:black;text-align:center;">Heart Health Detection System</h2>
    </div>
    <br>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    model_map = {"SVC":SVC(kernel='linear') , "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
                 "LogisticRegression":LogisticRegression(solver='liblinear'),
                 "LinearDiscriminantAnalysis":LinearDiscriminantAnalysis(), "GaussianNB" :GaussianNB(),
                    "DecisionTreeClassifier": DecisionTreeClassifier(random_state=42),
                     "KNeighborsClassifier": KNeighborsClassifier(n_neighbors=2)}

    model_name = st.selectbox(
        'Select the Machine Learning Model',
        ["SVC", "RandomForestClassifier","LogisticRegression", "LinearDiscriminantAnalysis",
              "GaussianNB","DecisionTreeClassifier","KNeighborsClassifier"])

    def convert(x):
        print(x)
        print("hello")
        return "1"
    
    sex_map = {"Male":1, "Female":2}
    chest_pain_map = {'ATA':0, 'NAP':1, 'ASY' : 2, "TA":3 }
    
    resting_ecg_map = {'Normal':0, 'ST':1, 'LVH' : 2 }
    
    exercise_Angina_map = {'N':0, 'Y':1}

    # age = st.number_input("Age")
    age = st.selectbox("Age", [i for i in range(100)])
    sex = sex_map[st.selectbox("Sex", ["Male","Female"])]
    chest_pain_type = chest_pain_map[st.selectbox("Select Chest Pain Type", ["ATA","NAP","ASY","TA"])]
    resting_bp = st.number_input("Resting BP")
    colestrol = st.number_input("Colestrol")
    fasting_bs = st.selectbox("Fasting BS", ["0","1"])
    resting_ecg_type = resting_ecg_map[st.selectbox("Select ECG Type", ["Normal","ST","LVH"])]
    max_hr = st.number_input("MAX HR")
    exercise_Angina_type = exercise_Angina_map[st.selectbox("Select Exercise Angina Type", ["N","Y"])]
    old_peak = st.number_input("Old Peak")
    
    
    if (st.button("Predict", type="primary")):
        pyfunc_model = mlflow.sklearn.load_model(f"models:/{model_name}/Production")


        prediction = pyfunc_model.predict([[age,sex,chest_pain_type,resting_bp,colestrol,fasting_bs,
                                resting_ecg_type,max_hr,exercise_Angina_type,
                                old_peak]])
        print(prediction)
        if(prediction[0]==1):
            st.write(f'<p style="color: Red; font-size: 150%">Heart Disease Detected!  \
                     Please Check your nearby doctor immediately</p>', unsafe_allow_html=True)
            
        else:
            st.write(f'<p style="color: Green; font-size: 100%">No Heart Disease Detected"</p>', unsafe_allow_html=True)
           
        
    


    


