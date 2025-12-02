import pandas as pd
import json


class SoilLoader:

    def __init__(self,
                 soil_json_path="UI/irrigation/dataset/soil.json",
                 state_district_csv="UI/irrigation/dataset/district-zone.csv",
                 soil_texture_csv="UI/irrigation/dataset/soil-texture.csv"):
        

        # Load district → soil type mapping (JSON)

        with open(soil_json_path, "r") as f:
            self.district_soil_map = json.load(f)

        # Normalize keys
        self.district_soil_map = {
            k.lower().strip(): v for k, v in self.district_soil_map.items()
        }

        # Load State → District mapping from CSV

        df = pd.read_csv(state_district_csv)
        df["District_lower"] = df["District"].str.lower().str.strip()
        self.state_district_df = df

        # Load Soil Type → Texture/Irrigation rules

        df_tex = pd.read_csv(soil_texture_csv)
        df_tex["soil_type_lower"] = df_tex["Soil_Type"].str.lower().str.strip()
        self.soil_df = df_tex
      
    def get_soil_type(self, district: str):
        """Return soil type for a district using JSON."""
        return self.district_soil_map.get(district.lower().strip(), None)
        
    def get_state_for_district(self, district: str):
        """Return state for a given district using the CSV."""
        district_key = district.lower().strip()

        match = self.state_district_df[
            self.state_district_df["District_lower"] == district_key
        ]

        if match.empty:
            return None
        
        return match.iloc[0]["State"]
        
    def get_soil_properties(self, soil_type: str):
        """Return soil texture + irrigation rules."""
        soil_key = soil_type.lower().strip()

        row = self.soil_df[self.soil_df["soil_type_lower"] == soil_key]

        if row.empty:
            return None

        return row.iloc[0].to_dict()
