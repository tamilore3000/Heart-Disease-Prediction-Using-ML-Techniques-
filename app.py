# IMPORT LIBRARIES 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Import libraries
import streamlit as st
import joblib
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# Load the trained model
model = joblib.load('ann_model.joblib')
best_estimator = model.best_estimator_

# Define the user interface
st.title('Heart Disease Prediction')
st.header('Input Features')
age = st.slider('Age', 18, 100, 25)
sex = st.selectbox('Sex', ['Male', 'Female'])
cp = st.selectbox('Chest Pain Type', ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'])
trestbps = st.slider('Resting Blood Pressure', 80, 250, 120)
chol = st.slider('Cholesterol Level', 0, 700, 200)
fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', ['True', 'False'])
restecg = st.selectbox('Resting Electrocardiographic Results', ['Normal', 'ST-T Wave Abnormality', 'Left Ventricular Hypertrophy'])
thalach = st.slider('Maximum Heart Rate Achieved', 50, 220, 150)
exang = st.selectbox('Exercise Induced Angina', ['Yes', 'No'])
oldpeak = st.slider('ST Depression Induced by Exercise Relative to Rest', 0.0, 7.0, 2.0)
slope = st.selectbox('Slope of the Peak Exercise ST Segment', ['Upsloping', 'Flat', 'Downsloping'])
ca = st.slider('Number of Major Vessels Colored by Fluoroscopy', 0, 4, 1)
thal = st.selectbox('Thalassemia', ['Normal', 'Fixed Defect', 'Reversible Defect'])

# Convert user inputs into a dataframe for the model
input_df = pd.DataFrame({'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol, 'fbs': fbs,
                         'restecg': restecg, 'thalach': thalach, 'exang': exang, 'oldpeak': oldpeak,
                         'slope': slope, 'ca': ca, 'thal': thal}, index=[0])

# Convert categorical variables to numerical
input_df['sex'] = input_df['sex'].map({'Male': 1, 'Female': 0})
input_df['cp'] = input_df['cp'].map({'Typical Angina': 0, 'Atypical Angina': 1, 'Non-anginal Pain': 2, 'Asymptomatic': 3})
input_df['fbs'] = input_df['fbs'].map({'True': 1, 'False': 0})
input_df['restecg'] = input_df['restecg'].map({'Normal': 0, 'ST-T Wave Abnormality': 1, 'Left Ventricular Hypertrophy': 2})
input_df['exang'] = input_df['exang'].map({'Yes': 1, 'No': 0})
input_df['slope'] = input_df['slope'].map({'Upsloping': 0, 'Flat': 1, 'Downsloping': 2})
input_df['thal'] = input_df['thal'].map({'Normal': 1, 'Fixed Defect': 2, 'Reversible Defect': 3})

# Make predictions using the user inputs and the trained model
prediction = best_estimator.predict(input_df)[0]

print("Input dataframe:", input_df)
print("Prediction:", prediction)


# Display the prediction
if (prediction == 1):
    st.write('You have a high risk of heart disease')
else:
    st.write('You have a low risk of heart disease')

