import os
from dotenv import load_dotenv
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from UI.crop_recommendation.crop import load_model, predict_crop, recommend_top3_crops
from weather_report.weather import build_url, fetch_weather, parse_weather
import pandas as pd
import numpy as np

# Load environment variables
load_dotenv()
API_KEY = os.getenv("OPENWEATHER_API_KEY")

# Load trained model
model = load_model("UI/model_paths/crop_recommendation_model.pkl")

print("🌾 Welcome to the Smart Crop Recommendation System 🌾")
print("------------------------------------------------------")

choice = input("Fetch live weather using city name? (y/n): ").strip().lower()

if choice == "y":
    city = input("Enter city name: ").strip()
    url = build_url(city=city, api_key=API_KEY)
    raw_data = fetch_weather(url)
    weather_data = parse_weather(raw_data)

    print("\n✅ Live weather fetched successfully:")
    for k, v in weather_data.items():
        print(f"{k}: {v}")

    N = float(input("\nEnter Nitrogen (N) content: "))
    P = float(input("Enter Phosphorus (P) content: "))
    K = float(input("Enter Potassium (K) content: "))
    ph = float(input("Enter soil pH: "))

    sample_input = {
        "N": N,
        "P": P,
        "K": K,
        "temperature": weather_data["temperature_C"],
        "humidity": weather_data["humidity_percent"],
        "ph": ph,
        "rainfall": weather_data["precipitation_mm"] + 5.0,
    }

else:
    print("\nEnter all parameters manually:")
    N = float(input("N: "))
    P = float(input("P: "))
    K = float(input("K: "))
    temperature = float(input("Temperature (°C): "))
    humidity = float(input("Humidity (%): "))
    ph = float(input("pH: "))
    rainfall = float(input("Rainfall (mm): "))

    sample_input = {
        "N": N,
        "P": P,
        "K": K,
        "temperature": temperature,
        "humidity": humidity,
        "ph": ph,
        "rainfall": rainfall,
    }

# -------- Prediction --------
predicted_crop = predict_crop(model, sample_input)
top3 = recommend_top3_crops(model, sample_input, model.classes_)

print("\n🌱 Crop Recommendation Results 🌱")
print(f"➡ Recommended Crop: {predicted_crop}")
print("➡ Top 3 Crop Suggestions:")
for crop, prob in top3:
    print(f"   - {crop}: {prob:.3f}")

# Optional logging
os.makedirs("logs", exist_ok=True)
log_df = pd.DataFrame([sample_input | {"predicted_crop": predicted_crop}])
log_df.to_csv("logs/recommendation_log.csv", mode="a", header=False, index=False)
print("\n Prediction logged to logs/recommendation_log.csv")
