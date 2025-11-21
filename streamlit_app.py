import streamlit as st
import pickle
import pandas as pd

Gradient_Boosting_Regressor = pickle.load(open('models/GBR_model.pkl', 'rb'))
Light_Gradient_Boosting_Regressor = pickle.load(open('models/LGBR_model.pkl', 'rb'))
Extreme_Gradient_Boosting_Regressor = pickle.load(open('models/XGBoost_model.pkl', 'rb'))
Random_Forest_Regressor = pickle.load(open('models/RandomForest_model.pkl', 'rb'))

st.title('This is a Calories Burnt Prediction Project')

name=st.text_input('Name')
Gender=st.selectbox('Gender',['male','female'])
Age=st.number_input('Age')
Height=st.number_input('Height (cm)')
Weight=st.number_input('Weight (kg)')
Duration=st.number_input('Duration of Workout (min)')
Heart_Rate=st.number_input('Heart rate')
Body_Temp=st.number_input('Body temp in celsius')

le=pickle.load(open('models/LabelEncoder.pkl', 'rb'))

Gender=le.transform([Gender])
df=pd.DataFrame(data=[[Gender[0],Age,Height,Weight,Duration,Heart_Rate,Body_Temp]],columns=['Gender', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate','Body_Temp'])
if st.button('Calculate'):
    predict=Gradient_Boosting_Regressor.predict(df) +Extreme_Gradient_Boosting_Regressor.predict(df)+Random_Forest_Regressor.predict(df)+Light_Gradient_Boosting_Regressor.predict(df)
    st.write(f'Hello {name} calories burnt is ',predict[0]/4)


