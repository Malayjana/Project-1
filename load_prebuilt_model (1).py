# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 13:01:17 2020

"""



import pandas as pd
import streamlit as st 
from xgboost.sklearn import XGBRegressor
#from pickle import dump
from pickle import load

st.title('XGBoost Regressor')

st.sidebar.header('Input Parameters')

def user_input_features():
    temp = st.sidebar.number_input('Temperature')
    exh = st.sidebar.number_input('Exhaust Vacuum')
    amp = st.sidebar.number_input('Ambient Pressure')
    hum = st.sidebar.number_input("Relative Humidity")
    data = {'Temperature':temp,
            'Exhaust Vacuum':exh,
            'Ambient Pressure':amp,
            'Relative Humidity':hum}
    features = pd.DataFrame(data,index = [0])
    return features 
    
input_df = user_input_features()
st.subheader('User Input parameters')
st.write(input_df)


# load the model from disk
model = load(open('xgb_save.sav', 'rb'))

prediction = model.predict(input_df)

st.subheader('Predicted Result')
st.write(prediction)



