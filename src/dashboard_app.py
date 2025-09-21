# src/dashboard_app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
import requests
import plotly.express as px
from dotenv import load_dotenv
import os


load_dotenv()  # load .env file
api_key = os.getenv("OPENWEATHER_API_KEY") or st.secrets["OPENWEATHER_API_KEY"]

# -----------------------------
# Load model and scaler
# -----------------------------
@st.cache_resource
def load_model():
    model = joblib.load("models/flood_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    return model, scaler

model, scaler = load_model()

# -----------------------------
# Load dataset for defaults
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("data/flood_data.csv")

data = load_data()

st.set_page_config(page_title="🌊 Flood Prediction", layout="wide")
st.title("🌊 Flood Prediction Dashboard")

# -----------------------------
# Mode selection
# -----------------------------
mode = st.radio("Select Mode:", ["Automatic (City)", "Manual Input"])

# Full list of features
all_features = ['Latitude','Longitude','Rainfall (mm)','Temperature (°C)','Humidity (%)',
                'River Discharge (m³/s)','Water Level (m)','Elevation (m)','Land Cover',
                'Soil Type','Population Density','Infrastructure','Historical Floods']

input_df = pd.DataFrame()

# -----------------------------
# Automatic Mode
# -----------------------------

# -----------------------------
# Automatic Mode
# -----------------------------
if mode == "Automatic (City)":
    st.subheader("Automatic Flood Prediction by City")
    city = st.text_input("Enter City Name:", "Delhi")  # User can type any city
    
    if city:
        weather_url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
        
        try:
            weather_data = requests.get(weather_url).json()
            
            if weather_data.get("cod") != 200:
                st.error(f"City not found: {city}. Please enter a valid city name.")
            else:
                temperature = weather_data["main"]["temp"]
                humidity = weather_data["main"]["humidity"]
                rainfall = weather_data.get("rain", {}).get("1h", 0)
                lat, lon = weather_data["coord"]["lat"], weather_data["coord"]["lon"]
                
                # Elevation
                elev_url = f"https://api.open-elevation.com/api/v1/lookup?locations={lat},{lon}"
                elev_data = requests.get(elev_url).json()
                elevation = elev_data["results"][0]["elevation"]
                
                st.markdown(f"**Weather Data for {city}:** Temperature: {temperature}°C, Humidity: {humidity}%, Rainfall: {rainfall}mm")
                
                # Fill input dataframe
                input_features = {
                    "Latitude": lat,
                    "Longitude": lon,
                    "Rainfall (mm)": rainfall,
                    "Temperature (°C)": temperature,
                    "Humidity (%)": humidity,
                    "River Discharge (m³/s)": data["River Discharge (m³/s)"].mean(),
                    "Water Level (m)": data["Water Level (m)"].mean(),
                    "Elevation (m)": elevation,
                    "Land Cover": data["Land Cover"].mode()[0],
                    "Soil Type": data["Soil Type"].mode()[0],
                    "Population Density": data["Population Density"].mean(),
                    "Infrastructure": data["Infrastructure"].mean(),
                    "Historical Floods": data["Historical Floods"].mean()
                }
                
                input_df = pd.DataFrame([input_features])
                
        except:
            st.warning("Failed to fetch API data. Using default values.")
            input_features = {
                "Latitude": 0,
                "Longitude": 0,
                "Rainfall (mm)": data["Rainfall (mm)"].mean(),
                "Temperature (°C)": data["Temperature (°C)"].mean(),
                "Humidity (%)": data["Humidity (%)"].mean(),
                "River Discharge (m³/s)": data["River Discharge (m³/s)"].mean(),
                "Water Level (m)": data["Water Level (m)"].mean(),
                "Elevation (m)": data["Elevation (m)"].mean(),
                "Land Cover": data["Land Cover"].mode()[0],
                "Soil Type": data["Soil Type"].mode()[0],
                "Population Density": data["Population Density"].mean(),
                "Infrastructure": data["Infrastructure"].mean(),
                "Historical Floods": data["Historical Floods"].mean()
            }
            input_df = pd.DataFrame([input_features])


# if mode == "Automatic (City)":
#     st.subheader("Automatic Flood Prediction by City")
#     city = st.text_input("Enter City Name:", "Rishikesh")
#     api_key = "ef1ac251f18706e4974d78e0f25f953c"  # Replace with your API key
#     weather_url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    
#     try:
#         weather_data = requests.get(weather_url).json()
#         temperature = weather_data["main"]["temp"]
#         humidity = weather_data["main"]["humidity"]
#         rainfall = weather_data.get("rain", {}).get("1h", 0)
#         lat, lon = weather_data["coord"]["lat"], weather_data["coord"]["lon"]
        
#         # Elevation
#         elev_url = f"https://api.open-elevation.com/api/v1/lookup?locations={lat},{lon}"
#         elev_data = requests.get(elev_url).json()
#         elevation = elev_data["results"][0]["elevation"]
        
#     except:
#         st.warning("Failed to fetch API data. Using default values.")
#         temperature = data["Temperature (°C)"].mean()
#         humidity = data["Humidity (%)"].mean()
#         rainfall = data["Rainfall (mm)"].mean()
#         lat, lon = 0, 0
#         elevation = data["Elevation (m)"].mean()
    
#     # Fill input dataframe with defaults or API values
#     input_features = {
#         "Latitude": lat,
#         "Longitude": lon,
#         "Rainfall (mm)": rainfall,
#         "Temperature (°C)": temperature,
#         "Humidity (%)": humidity,
#         "River Discharge (m³/s)": data["River Discharge (m³/s)"].mean(),
#         "Water Level (m)": data["Water Level (m)"].mean(),
#         "Elevation (m)": elevation,
#         "Land Cover": data["Land Cover"].mode()[0],
#         "Soil Type": data["Soil Type"].mode()[0],
#         "Population Density": data["Population Density"].mean(),
#         "Infrastructure": data["Infrastructure"].mean(),
#         "Historical Floods": data["Historical Floods"].mean()
#     }
#     st.subheader("Automatic feature values (can adjust manually below)")
#     input_df = pd.DataFrame([input_features])

# -----------------------------
# Manual Mode / Manual Overrides
# -----------------------------
with st.expander("Adjust / Input Features Manually"):
    user_input = {}
    for col in all_features:
        if input_df.empty:
            default_val = data[col].mean() if data[col].dtype != object else data[col].mode()[0]
        else:
            default_val = input_df[col][0]
        
        if data[col].dtype == object:
            user_input[col] = st.selectbox(col, data[col].unique(), index=list(data[col].unique()).index(default_val))
        else:
            min_val = float(data[col].min())
            max_val = float(data[col].max())
            user_input[col] = st.slider(col, min_val, max_val, float(default_val))
    
    input_df = pd.DataFrame([user_input])

# -----------------------------
# Encode categorical
# -----------------------------
for col in ["Land Cover","Soil Type"]:
    le = LabelEncoder().fit(data[col])
    input_df[col] = le.transform(input_df[col])

# -----------------------------
# Scale and predict
# -----------------------------
input_scaled = scaler.transform(input_df)
if st.button("Predict Flood"):
    pred = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]
    status = "⚠️ Flood Likely" if pred==1 else "✅ No Flood Expected"
    color = "red" if pred==1 else "green"
    
    st.markdown(f"<h2 style='color:{color}'>{status} (Probability: {prob:.2f})</h2>", unsafe_allow_html=True)
    
    # Pie chart
    fig = px.pie(names=["Flood Probability","No Flood Probability"], 
                 values=[prob, 1-prob], hole=0.5,
                 color_discrete_sequence=["red","green"])
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Trend & Dataset
# -----------------------------
st.subheader("Last 30 Records Flood Probability Trend")
last30 = data.tail(30).copy()
X_last30 = last30[all_features].copy()
for col in ["Land Cover","Soil Type"]:
    le = LabelEncoder().fit(data[col])
    X_last30[col] = le.transform(X_last30[col])
X_last30_scaled = scaler.transform(X_last30)
last30["Predicted Probability"] = model.predict_proba(X_last30_scaled)[:,1]

fig2 = px.line(last30, y="Predicted Probability", markers=True)
fig2.update_layout(yaxis_title="Predicted Flood Probability")
st.plotly_chart(fig2, use_container_width=True)

with st.expander("📂 Dataset Preview"):
    st.dataframe(data.head(20))
