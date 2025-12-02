def irrigation_decision(soil_info: dict, rainfall_mm: float):
    """
    Combine CSV irrigation rules with dynamic rainfall-based logic.
    """
    soil_type = soil_info["Soil_Type"]
    texture = soil_info["Soil_Texture"]

    # Base irrigation defined in CSV
    irrigation_methods = soil_info["Recommended_Irrigation"].split(",")

    # 🌧 Rainfall rule
    if rainfall_mm != 0:
        if rainfall_mm > 10:
            irrigation_methods.append("Surface Irrigation (high rainfall)")
        elif rainfall_mm < 2:
            irrigation_methods.append("Drip (low rainfall)")
        else:
            irrigation_methods.append("Sprinkler (moderate rainfall)")

    # 🏜 Special case for black soil
    if "black" in soil_type.lower():
        irrigation_methods.append("Basin Irrigation (high retention soil)")

    return list(set(irrigation_methods))
