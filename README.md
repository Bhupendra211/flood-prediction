# 🌊 Flood Prediction System

A **machine learning-based flood prediction dashboard** built with **Python** and **Streamlit**.  
The system predicts the likelihood of floods based on multiple environmental, weather, and geographical factors, and provides **interactive visualizations** for easy interpretation.

---

## 🔹 Features

- **Automatic City Mode**  
  - Search for any city and fetch **weather data** (Rainfall, Temperature, Humidity) automatically via OpenWeatherMap API.  
  - Fetch **latitude, longitude, and elevation** automatically.  

- **Manual Input Mode**  
  - Customize all input features manually: River Discharge, Water Level, Land Cover, Soil Type, etc.  

- **Prediction & Probability**  
  - Predicts flood occurrence and shows **probability**.  
  - Color-coded status: ✅ No Flood / ⚠️ Flood Likely.  

- **Visualizations**  
  - Pie chart for Flood/No Flood probability.  
  - Trend chart for last 30 predictions.  
  - Dataset preview for exploration.  

- **Secure API Keys**  
  - Uses `.env` file to store API keys securely.  

---

## 🔹 Features Used for Prediction

| Feature | Description |
|---------|-------------|
| Latitude | City latitude |
| Longitude | City longitude |
| Rainfall (mm) | Precipitation in mm |
| Temperature (°C) | Current temperature |
| Humidity (%) | Current humidity |
| River Discharge (m³/s) | River flow rate |
| Water Level (m) | Water level of nearby rivers |
| Elevation (m) | Altitude of the location |
| Land Cover | Type of land (Forest, Water Body, Urban) |
| Soil Type | Soil composition |
| Population Density | Number of people per km² |
| Infrastructure | Infrastructure development score |
| Historical Floods | Previous flood occurrences |

---

## 🔹 Technologies Used

- **Python 3.x**  
- **Machine Learning:** scikit-learn  
- **Web Dashboard:** Streamlit  
- **Data Visualization:** Plotly, Streamlit Charts  
- **API Integration:** OpenWeatherMap, Open-Elevation  
- **Environment Management:** virtualenv, `.env` for secrets  

---

## 🔹 Installation

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/flood-prediction.git
cd flood-prediction
