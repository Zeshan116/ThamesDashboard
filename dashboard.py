# -*- coding: utf-8 -*-
"""
Created on Thu Sep  4 11:31:03 2025

@author: DELL
"""

import pandas as pd
import streamlit as st
import joblib
import matplotlib.pyplot as plt

# -----------------------
# 1️⃣ Load Data + Model
# -----------------------
df = pd.read_csv("cleaned_thames_valley_data_small.csv")

# Load model and encoders
model = joblib.load("crime_rf_model.pkl")
le_location = joblib.load("le_location.pkl")
le_falls_within = joblib.load("le_falls_within.pkl")

# -----------------------
# 2️⃣ Streamlit Dashboard
# -----------------------
st.title("Thames Valley Police: Predictive Analytics Dashboard")

# Sidebar inputs
month = st.sidebar.slider("Select Month (1-12)", 1, 12)
location = st.sidebar.selectbox("Select Location", le_location.classes_)
falls_within = st.sidebar.selectbox("Select 'Falls Within'", le_falls_within.classes_)
latitude = st.sidebar.number_input("Latitude", value=51.5)
longitude = st.sidebar.number_input("Longitude", value=-0.8)

# Encode input
location_enc = le_location.transform([location])[0]
falls_within_enc = le_falls_within.transform([falls_within])[0]

if st.sidebar.button("Predict Crime Type"):
    prediction = model.predict([[month, location_enc, falls_within_enc, latitude, longitude]])
    st.success(f"Predicted Crime Type: {prediction[0]}")

# -----------------------
# 3️⃣ Descriptive Charts
# -----------------------
st.subheader("Crime by Type")
crime_counts = df['Crime type'].value_counts()
st.bar_chart(crime_counts)

st.subheader("Top 10 Locations with Highest Crimes")
top_locations = df['Location'].value_counts().head(10)
st.bar_chart(top_locations)

st.subheader("Crime Outcomes")
outcome_counts = df['Last outcome category'].value_counts()
fig, ax = plt.subplots(figsize=(6,6))
ax.pie(outcome_counts, autopct='%1.1f%%', startangle=140, colors=plt.cm.tab20.colors)
centre_circle = plt.Circle((0,0),0.70,fc='white')
fig.gca().add_artist(centre_circle)
ax.set_title("Crime Outcomes")
st.pyplot(fig)

