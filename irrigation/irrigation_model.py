import json
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
from irrigation.soil_loader import SoilLoader
from irrigation.irrigation_rules import irrigation_decision
from weather_report.rainfall_api import get_daily_rainfall

load_dotenv()


class IrrigationModel:

    def __init__(self):
        self.soil_loader = SoilLoader()
        self.imd_api_key = os.getenv("IMD_API_KEY")

    def recommend(self, district: str):
        district_clean = district.strip()

        # Step 1: Soil type from JSON
        soil_type = self.soil_loader.get_soil_type(district_clean)
        if soil_type is None:
            return {"error": f"Soil type not found for '{district_clean}'"}

        # Step 2: Soil texture & behavior from texture CSV
        soil_info = self.soil_loader.get_soil_properties(soil_type)
        if soil_info is None:
            return {"error": f"No soil texture data found for '{soil_type}'"}

        # Step 3: State from CSV (needed for rainfall API)
        state = self.soil_loader.get_state_for_district(district_clean)
        if state is None:
            return {"error": f"State not found for district '{district_clean}'"}

        # Step 4: Rainfall from IMD API
        rain_data = get_daily_rainfall(
            api_key=self.imd_api_key,
            state=state,
            district=district_clean
        )

        rainfall = float(rain_data.get("rainfall_mm", 0) or 0)

        # Step 5: Irrigation decision
        irrigation_list = irrigation_decision(soil_info, rainfall)

        return {
            "district": district_clean,
            "state": state,
            "soil_type": soil_type,
            "soil_texture": soil_info["Soil_Texture"],
            "rainfall_mm_today": rainfall,
            "recommended_irrigation": irrigation_list
        }


