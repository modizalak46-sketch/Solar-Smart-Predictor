import streamlit as st
import pandas as pd
import pickle
import requests
from datetime import datetime

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="Solar AI Predictor", page_icon="☀️", layout="wide")

# --- 2. LOAD MODEL ---
@st.cache_resource
def load_solar_model():
    try:
        return pickle.load(open('solar_model.pkl', 'rb'))
    except FileNotFoundError:
        return None

model = load_solar_model()

# --- 3. WEATHER API FUNCTION ---
def get_weather_data(city, api_key):
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
        res = requests.get(url).json()
        if res.get("cod") == 200:
            cloud_pct = res["clouds"]["all"]
            temp = res["main"]["temp"]
            if cloud_pct < 20: return "Clear Sky", 1.0, temp, cloud_pct
            elif cloud_pct < 60: return "Partly Cloudy", 0.6, temp, cloud_pct
            else: return "Overcast/Rainy", 0.2, temp, cloud_pct
        return None, 1.0, None, None
    except:
        return None, 1.0, None, None

# --- 4. SIDEBAR (All Inputs Here) ---
st.sidebar.header("📍 Location & Config")
# Ahiya thi tame city change kari shaksho
city = st.sidebar.text_input("Enter City Name", value="Bharuch")

mode = st.sidebar.radio("Weather Mode", ["Auto-Weather (Live API)", "Manual Selection"])

api_key = ""
if mode == "Auto-Weather (Live API)":
    api_key = st.sidebar.text_input("Enter API Key", type="password")
    st.sidebar.caption("Get key from openweathermap.org")

st.sidebar.markdown("---")
st.sidebar.header("⏰ Time & Maintenance")
selected_date = st.sidebar.date_input("Select Date", datetime.now())
selected_hour = st.sidebar.slider("Select Hour (0-23)", 0, 23, 12)
days_since_cleaning = st.sidebar.slider("Days since cleaning", 0, 30, 0)

# --- 5. DYNAMIC LOGIC ---
weather_condition = "Clear Sky"
weather_multiplier = 1.0
temp_info = ""

if mode == "Auto-Weather (Live API)" and api_key:
    cond, mult, temp, clouds = get_weather_data(city, api_key)
    if cond:
        weather_condition, weather_multiplier = cond, mult
        temp_info = f"Current Temp: {temp}°C | Clouds: {clouds}%"
    else:
        st.sidebar.error("City not found or API key invalid!")
else:
    weather_condition = st.sidebar.selectbox("Sky Condition (Manual)", ["Clear Sky", "Partly Cloudy", "Overcast/Rainy"])
    weather_multiplier = {"Clear Sky": 1.0, "Partly Cloudy": 0.6, "Overcast/Rainy": 0.2}[weather_condition]

dust_loss = (days_since_cleaning * 0.005)

# --- 6. MAIN DASHBOARD ---
st.title("☀️ AI Solar Energy Forecaster Pro")
st.markdown(f"### Monitoring: {city}")
if temp_info:
    st.info(f"🌍 Live Weather Data: {temp_info}")

st.markdown("---")

if st.button("🚀 Run Prediction"):
    if model is None:
        st.error("Error: 'solar_model.pkl' not found!")
    else:
        # Model Input
        input_df = pd.DataFrame({
            'hour': [selected_hour],
            'month': [selected_date.month],
            'installedcapacity_mwp': [16445.0]
        })

        base_pred = model.predict(input_df)[0]
        
        # Night calculation
        if selected_hour < 6 or selected_hour > 19:
            final_output = 0.0
        else:
            final_output = float(base_pred * weather_multiplier * (1 - dust_loss))

        # --- Display Metrics ---
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("⚡ Output", f"{round(final_output, 2)} MW")
        col2.metric("☁️ Condition", weather_condition)
        col3.metric("💰 Savings", f"₹{round(final_output * 1000 * 7, 0)}")
        col4.metric("🛠️ Health", f"{round((1-dust_loss)*100, 1)}%")

        st.markdown("---")
        st.write("### 📊 Production Capacity")
        st.progress(min(final_output / 8000, 1.0))

        if days_since_cleaning > 10:
            st.warning(f"Maintenance Suggestion: Cleaning panels can improve output by {round(dust_loss*100, 1)}%")

st.caption("Developed by Zalak Modi | Built with Python & XGBoost")