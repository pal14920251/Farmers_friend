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
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

# IMPORTANT FIX: missing import
from language_translation.text_generation import (
    generate_comprehensive_crop_report,
    generate_report_from_template,
    translate_large_text
)

# >>> ADDED FOR JSON STORAGE <<<
import json
RESULT_JSON_PATH = "UI/session_results.json"

def update_session_json(updates: dict):
    os.makedirs(os.path.dirname(RESULT_JSON_PATH), exist_ok=True)
    data = {}

    if os.path.exists(RESULT_JSON_PATH):
        try:
            with open(RESULT_JSON_PATH, "r") as f:
                content = f.read().strip()
                data = json.loads(content) if content else {}
        except json.JSONDecodeError:
            data = {}

    data.update(updates)

    with open(RESULT_JSON_PATH, "w") as f:
        json.dump(data, f, indent=4)
# >>> END JSON STORAGE <<<

def load_session_inputs(path="UI/session_results.json"):
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)

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
# UI START
# ----------------------------------------------------

st.title("🌾 Smart Farming Assistant")
st.write("Crop Recommendation + Plant Disease Detection")

tab1, tab2 = st.tabs(["🌱 Crop Recommendation", "🩺 Disease Detection"])

# ============================================================
# 🌱 TAB 1: CROP RECOMMENDATION
# ============================================================

with tab1:

    st.markdown("---")
    st.subheader("🌐 Select Report Output Language")

    lang_script_map = {
        "eng_Latn": "English",
        "hin_Deva": "Hindi",
        "tam_Taml": "Tamil",
        "tel_Telu": "Telugu",
        "kan_Knda": "Kannada",
        "mal_Mlym": "Malayalam",
        "ben_Beng": "Bengali",
        "pan_Guru": "Punjabi",
        "mar_Deva": "Marathi",
        "guj_Gujr": "Gujarati",
        "ory_Orya": "Odia",
        "asm_Beng": "Assamese",
        "san_Deva": "Sanskrit",
        "npi_Deva": "Nepali",
        "gom_Deva": "Konkani",
        "kas_Arab": "Kashmiri (Arabic)",
        "kas_Deva": "Kashmiri (Devanagari)",
        "snd_Arab": "Sindhi (Arabic)",
        "snd_Deva": "Sindhi (Devanagari)",
        "urd_Arab": "Urdu"
    }

    readable_langs = {v: k for k, v in lang_script_map.items()}
    selected_lang = st.selectbox("Choose Language", options=list(readable_langs.keys()), key="lang_select")

    tgt_lang_code = readable_langs[selected_lang]
    update_session_json({"target_language": tgt_lang_code})
    st.info(f"Selected language: **{selected_lang}**")
    st.markdown("---")

    st.header("🌱 Crop Recommendation System")
    st.write("Select location to fetch soil nutrient values:")

    df_zone = pd.read_csv("/data1/home/anumalas/GENAI-PROJECT/UI/irrigation/dataset/merged_final_dataset_sorted[1].csv")

    state = st.selectbox("Select State", sorted(df_zone["State"].unique()), key="state_select")
    district = st.selectbox("Select District", sorted(df_zone[df_zone["State"] == state]["District"].unique()), key="district_select")

    update_session_json({"location_state": state, "location_district": district})

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

    # FIXED BUTTON DUPLICATION PROBLEM
    if st.button("Fetch Soil Nutrients", key="btn_fetch_soil"):
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

            update_session_json({
                "soil_nitrogen": st.session_state.soil_data["N"],
                "soil_phosphorus": st.session_state.soil_data["P"],
                "soil_potassium": st.session_state.soil_data["K"],
                "soil_ph": st.session_state.soil_data["pH"]
            })

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

    st.subheader("🌦 Weather Input")
    use_live_weather = st.checkbox("Use live weather", key="weather_checkbox")

    if "weather_data" not in st.session_state:
        st.session_state.weather_data = None

    if use_live_weather:
        st.info(f"Weather will be fetched for: **{district}, {state}**")

        if st.button("Fetch Weather", key="btn_fetch_weather"):
            try:
                url = build_url(city=district, api_key=API_KEY)
                raw = fetch_weather(url)
                st.session_state.weather_data = parse_weather(raw)
                st.success("Weather data fetched successfully!")

                w = st.session_state.weather_data
                update_session_json({
                    "weather_forecast": w,
                    "weather_summary": f"Temperature: {w['temperature_C']}°C, Humidity: {w['humidity_percent']}%, Description: {w['weather']}"
                })

            except Exception as e:
                st.error(f"Error fetching weather: {e}")

    else:
        temperature = st.number_input("Temperature (°C)", value=25.0)
        humidity = st.number_input("Humidity (%)", value=70.0)
        rainfall = st.number_input("Rainfall (mm)", value=100.0)

        update_session_json({
            "weather_forecast": {
                "temperature_C": temperature,
                "humidity_percent": humidity,
                "rainfall_mm": rainfall,
                "weather": "manual input"   # IMPORTANT FIX
            },
            "weather_summary": f"Temperature: {temperature}°C, Humidity: {humidity}%, Rainfall: {rainfall} mm"
        })

    # FIXED BUTTON
    if st.button("Recommend Crop", key="btn_recommend_crop"):

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

        update_session_json({"recommended_crop": predicted_crop})

    st.markdown("---")
    st.subheader("💧 Irrigation Recommendation")

    # FIXED BUTTON
    if st.button("Recommend Irrigation", key="btn_recommend_irrigation"):
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

            update_session_json({
                "irrigation_type": ", ".join(irrigation_result.get("recommended_irrigation", []))  # FIX: readable string
            })

    session_data = load_session_inputs()
    tgt_lang = session_data.get("target_language", "eng_Latn")

    # FIXED BUTTON
    if st.button("Generate Full Crop Report", key="btn_generate_crop_report"):
        report = generate_comprehensive_crop_report()

        if tgt_lang != "eng_Latn":
            try:
                st.info(f"Translating report to {selected_lang} using Bhashini...")
                translated = translate_large_text(
                    report,
                    src_lang="eng_Latn",
                    tgt_lang=tgt_lang,
                    max_chars=400
                )
                st.subheader(f"📄 Final Report ({selected_lang})")
                st.write(translated)
            except Exception as e:
                st.error(f"Translation failed: {e}")
                st.subheader("📄 English Report")
                st.write(report)
        else:
            st.subheader("📄 Final Report (English)")
            st.write(report)

# ============================================================
# 🩺 TAB 2: DISEASE DETECTION
# ============================================================

with tab2:
    st.header("🩺 Plant Disease Detection")

    uploaded_file = st.file_uploader(
        "Upload a leaf image", 
        type=["jpg", "jpeg", "png"],
        key="file_upload_disease"
    )

    # Initialize session storage for disease detection
    if "detected_disease" not in st.session_state:
        st.session_state.detected_disease = None
    if "disease_report" not in st.session_state:
        st.session_state.disease_report = None

    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded Image", use_column_width=True)

        # -------- Predict disease (STORE in session_state) --------
        if st.button("Predict Disease", key="btn_predict_disease"):
            try:
                disease = predict_disease(img)
                st.session_state.detected_disease = disease
                st.success(f"Predicted Disease: {disease}")

            except Exception as e:
                st.error(f"Error predicting disease: {e}")

        # -------- Show predicted disease if already stored --------
        if st.session_state.detected_disease:
            st.info(f"Detected Disease: {st.session_state.detected_disease}")

            # -------- Generate Disease Report button --------
            if st.button("Generate Disease Report", key="btn_generate_disease_report"):
                session_data = {
                    "identified_disease_or_pest_name": st.session_state.detected_disease
                }

                try:
                    disease_report = generate_report_from_template(
                        prompt_file="/data1/home/anumalas/GENAI-PROJECT/UI/language_translation/prompt_template_pesticide.json",
                        template_key="disease_treatment_prompt",
                        session_data=session_data
                    )
                    st.session_state.disease_report = disease_report
                except Exception as e:
                    st.error(f"Error generating report: {e}")

        # -------- Display disease report if available --------
        if st.session_state.disease_report:
            st.subheader("📘 Disease Advisory (English)")
            st.write(st.session_state.disease_report)

            lang_choice = st.selectbox(
                "Translate report to:",
                ["None", "Hindi", "Tamil", "Telugu", "Kannada", "Malayalam", "Bengali", "Punjabi", "Marathi"],
                key="disease_lang_select"
            )

            lang_map_ui = {
                "Hindi": "hin_Deva",
                "Tamil": "tam_Taml",
                "Telugu": "tel_Telu",
                "Kannada": "kan_Knda",
                "Malayalam": "mal_Mlym",
                "Bengali": "ben_Beng",
                "Punjabi": "pan_Guru",
                "Marathi": "mar_Deva"
            }

            if lang_choice != "None":
                tgt = lang_map_ui[lang_choice]
                st.info("Translating, please wait...")

                translated_output = translate_large_text(
                    full_text=st.session_state.disease_report,
                    src_lang="eng_Latn",
                    tgt_lang=tgt
                )

                st.subheader(f"🌐 Disease Advisory ({lang_choice})")
                st.write(translated_output)
