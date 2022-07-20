# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pickle
import pandas as pd
import streamlit as st 

from PIL import Image

pickle_in = open("model2.pkl","rb")
classifier=pickle.load(pickle_in)

def data(temperature, exhaust_vacuum, amb_pressure, r_humidity):
      prediction=classifier.predict([[temperature, exhaust_vacuum, amb_pressure, r_humidity]])
      print(prediction)
      return prediction

def main():
   
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Random Forest Regressor Model</h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    temperature = st.text_input("Temperature(Celsius) ")
    exhaust_vacuum = st.text_input("Exhaust Vacuum (Hg)")
    amb_pressure = st.text_input("Ambient Pressure (Millibar)")
    r_humidity = st.text_input("Relative Humidity(Percentage)")
    result=""
    if st.button("Predict"):
        result=data(temperature, exhaust_vacuum, amb_pressure, r_humidity)
    st.success('The net hourly electrical energy  output is {}'.format(result))
    

if __name__=='__main__':
    main()
    