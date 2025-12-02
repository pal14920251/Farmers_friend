# app.py
import streamlit as st
from PIL import Image
from torchvision import transforms
import torch
from disease_detection.resnet50 import load_model
from disease_detection.classes import class_names
import os
import sys
import pandas as pd
import numpy as np
from dotenv import load_dotenv
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from crop_recommendation.crop import load_model as load_crop_model
from crop_recommendation.crop import predict_crop, recommend_top3_crops
from weather_report.weather import build_url, fetch_weather, parse_weather
from irrigation.irrigation_model import IrrigationModel


# >>> ADDED FOR JSON STORAGE <<<
import json
RESULT_JSON_PATH = "UI/session_results.json"

def update_session_json(updates: dict):
    # Ensure directory exists
    os.makedirs(os.path.dirname(RESULT_JSON_PATH), exist_ok=True)

    # Try loading existing JSON safely
    data = {}

    if os.path.exists(RESULT_JSON_PATH):
        try:
            with open(RESULT_JSON_PATH, "r") as f:
                content = f.read().strip()

                # If file empty → treat as new dict
                if content:
                    data = json.loads(content)
                else:
                    data = {}

        except json.JSONDecodeError:
            # JSON corrupted → reset file
            data = {}

    # Apply updates
    data.update(updates)

    # Write back safely
    with open(RESULT_JSON_PATH, "w") as f:
        json.dump(data, f, indent=4)

# >>> END JSON STORAGE ADDED <<<


# load variables 
load_dotenv()
API_KEY = os.getenv("OPENWEATHER_API_KEY")

# Load Model

crop_model = load_crop_model("UI/model_paths/crop_recommendation_rf.pkl")
irrigation_model = IrrigationModel()


DISEASE_MODEL_PATH = "UI/model_paths/plant-disease-model-50-20.pth"

device = "cuda" if torch.cuda.is_available() else "cpu"

disease_model = load_model(
    DISEASE_MODEL_PATH,
    num_classes=len(class_names),
    device=device
)
# Prediction Transform

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

def predict_disease(img):
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = disease_model(img_tensor)
        _, pred = torch.max(outputs, 1)

    return class_names[pred.item()]

# ----------------------------------------------------
# Streamlit UI
# ----------------------------------------------------

st.title("🌾 Smart Farming Assistant")
st.write("Crop Recommendation + Plant Disease Detection")

tab1, tab2 = st.tabs(["🌱 Crop Recommendation", "🩺 Disease Detection"])


# ============================================================
# 🌱 TAB 1: CROP RECOMMENDATION
# ============================================================

with tab1:
    st.header("🌱 Crop Recommendation System")
    st.write("Select location to fetch soil nutrient values:")

    df_zone = pd.read_csv("/data1/home/anumalas/GENAI-PROJECT/UI/irrigation/dataset/merged_final_dataset_sorted[1].csv")

    states = sorted(df_zone["State"].unique())
    state = st.selectbox("Select State", states)

    districts = sorted(df_zone[df_zone["State"] == state]["District"].unique())
    district = st.selectbox("Select District", districts)

    # >>> ADDED FOR JSON STORAGE <<<
    update_session_json({
        "location_state": state,
        "location_district": district
    })
    # >>> END ADDED <<<

    if "last_state" not in st.session_state:
        st.session_state.last_state = None
        st.session_state.last_district = None

    if (st.session_state.last_state != state) or (st.session_state.last_district != district):
        st.session_state.soil_data = None
        st.session_state.last_state = state
        st.session_state.last_district = district

    if "soil_data" not in st.session_state:
        st.session_state.soil_data = None

    st.markdown("### 🧪 Soil Nutrients")
    
    if st.button("Fetch Soil Nutrients"):
        row = df_zone[(df_zone["State"] == state) & (df_zone["District"] == district)]
        if row.empty:
            st.error("No soil nutrient data found for this location.")
        else:
            st.session_state.soil_data = {
                "N": float(row["N"].iloc[0]),
                "P": float(row["P"].iloc[0]),
                "K": float(row["K"].iloc[0]),
                "pH": float(row["pH"].iloc[0]),
            }

            # >>> ADDED FOR JSON STORAGE <<<
            update_session_json({
                "soil_nitrogen": st.session_state.soil_data["N"],
                "soil_phosphorus": st.session_state.soil_data["P"],
                "soil_potassium": st.session_state.soil_data["K"],
                "soil_ph": st.session_state.soil_data["pH"]
            })
            # >>> END ADDED <<<

    if st.session_state.soil_data:
        N = st.session_state.soil_data["N"]
        P = st.session_state.soil_data["P"]
        K = st.session_state.soil_data["K"]
        ph = st.session_state.soil_data["pH"]

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Nitrogen (N)", f"{N:.2f}")
            st.metric("Phosphorus (P)", f"{P:.2f}")

        with col2:
            st.metric("Potassium (K)", f"{K:.2f}")
            st.metric("Soil pH", f"{ph:.2f}")

    else:
        st.info("Click **Fetch Soil Nutrients** to load N, P, K, pH values.")

    st.subheader("🌦 Weather Input")
    use_live_weather = st.checkbox("Use live weather")

    if "weather_data" not in st.session_state:
        st.session_state.weather_data = None

    if use_live_weather:
        st.info(f"Weather will be fetched for: **{district}, {state}**")

        if st.button("Fetch Weather"):
            try:
                url = build_url(city=district, api_key=API_KEY)
                raw = fetch_weather(url)
                st.session_state.weather_data = parse_weather(raw)
                st.success("Weather data fetched successfully!")

                # >>> ADDED FOR JSON STORAGE <<<
                w = st.session_state.weather_data
                update_session_json({
                    "weather_forecast": w,
                    "weather_summary": f"Temperature: {w['temperature_C']}°C, Humidity: {w['humidity_percent']}%, Description: {w['weather']}"
                })
                # >>> END ADDED <<<

            except Exception as e:
                st.error(f"Error fetching weather: {e}")

        if st.session_state.weather_data:
            html = "<table style='width:60%; border-collapse: collapse;'>"
            html += "<tr><th style='padding:8px; border:1px solid #ccc;'>Metric</th>"
            html += "<th style='padding:8px; border:1px solid #ccc;'>Value</th></tr>"

            for key, value in st.session_state.weather_data.items():
                html += (
                    f"<tr><td style='padding:8px; border:1px solid #ccc;'>{key}</td>"
                    f"<td style='padding:8px; border:1px solid #ccc;'>{value}</td></tr>"
                )
            html += "</table>"
            st.markdown(html, unsafe_allow_html=True)

    else:
        st.write("### 🌤 Manual Weather Input")
        temperature = st.number_input("Temperature (°C)", value=25.0)
        humidity = st.number_input("Humidity (%)", value=70.0)
        rainfall = st.number_input("Rainfall (mm)", value=100.0)

        # >>> ADDED FOR JSON STORAGE <<<
        update_session_json({
            "weather_forecast": {
                "temperature_C": temperature,
                "humidity_percent": humidity,
                "rainfall_mm": rainfall
            },
            "weather_summary": f"Temperature: {temperature}°C, Humidity: {humidity}%, Rainfall: {rainfall} mm"
        })
        # >>> END ADDED <<<

    if st.button("Recommend Crop"):

        if not st.session_state.soil_data:
            st.error("Please fetch soil nutrients first.")
            st.stop()

        N = st.session_state.soil_data["N"]
        P = st.session_state.soil_data["P"]
        K = st.session_state.soil_data["K"]
        ph = st.session_state.soil_data["pH"]

        if use_live_weather and st.session_state.weather_data:
            w = st.session_state.weather_data
            sample_input = {
                "N": N, "P": P, "K": K,
                "temperature": w["temperature_C"],
                "humidity": w["humidity_percent"],
                "ph": ph,
            }
        else:
            sample_input = {
                "N": N, "P": P, "K": K,
                "temperature": temperature,
                "humidity": humidity,
                "ph": ph,
            }

        predicted_crop = predict_crop(crop_model, sample_input)
        top3 = recommend_top3_crops(crop_model, sample_input, crop_model.classes_)

        st.success(f"🌾 Recommended Crop: **{predicted_crop}**")
        st.write("### 🌟 Top 3 Possible Crops:")
        for crop, prob in top3:
            st.write(f"- {crop}: **{prob:.3f}**")

        # >>> ADDED FOR JSON STORAGE <<<
        update_session_json({
            "recommended_crop": predicted_crop
        })
        # >>> END ADDED <<<


    st.markdown("---")
    st.subheader("💧 Irrigation Recommendation")

    if st.button("Recommend Irrigation"):
        if not district:
            st.error("Please enter a district name.")
        else:
            irrigation_result = irrigation_model.recommend(district)

            if "error" in irrigation_result:
                st.error(irrigation_result["error"])
            else:
                st.success("Irrigation Recommendation Generated")

                st.write(f"**District:** {irrigation_result['district']}")
                st.write(f"**State:** {irrigation_result['state']}")
                st.write(f"**Soil Type:** {irrigation_result['soil_type']}")
                st.write(f"**Soil Texture:** {irrigation_result['soil_texture']}")
                st.write(f"**Rainfall Today:** {irrigation_result['rainfall_mm_today']} mm")

                st.write("### 🚿 Recommended Irrigation:")
                for step in irrigation_result["recommended_irrigation"]:
                    st.write(f"- {step}")

                # >>> ADDED FOR JSON STORAGE <<<
                update_session_json({
                    "irrigation_type": irrigation_result.get("recommended_irrigation", [])
                })
                # >>> END ADDED <<<


# ============================================================
# 🩺 TAB 2: DISEASE DETECTION
# ============================================================

with tab2:
    st.header("🩺 Plant Disease Detection")

    uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded Image", use_column_width=True)

        if st.button("Predict Disease"):
            try:
                disease = predict_disease(img)
                st.success(f"**Predicted Disease:** {disease}")
            except Exception as e:
                st.error(f"Error predicting disease: {e}")
