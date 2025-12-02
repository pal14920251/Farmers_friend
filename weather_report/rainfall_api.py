import requests
from datetime import datetime

IMD_RESOURCE_ID = "6c05cd1b-ed59-40c2-bc31-e314f39c6971"   # Daily District-wise Rainfall Data


def get_daily_rainfall(api_key: str, state: str, district: str, date: str = None):
    """
    Fetch rainfall in mm for a given Indian state + district for today's date
    using IMD Daily District-wise Rainfall API.
    """

    if date is None:
        # Today's date in DD-MM-YYYY
        date = datetime.now().strftime("%d-%m-%Y")

    base_url = f"https://api.data.gov.in/resource/{IMD_RESOURCE_ID}"

    params = {
        "api-key": api_key,
        "format": "json",
        "limit": 100,
        "filters[State]": state,
        "filters[District]": district,
        "filters[Date]": date
    }

    try:
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if "records" not in data or len(data["records"]) == 0:
            return {
                "error": f"No rainfall data found for {district}, {state} on {date}"
            }

        record = data["records"][0]

        # IMD rainfall value column may be labeled "Rainfall", "Rainfall(mm)" or "Daily_rainfall"
        rainfall_mm = (
            record.get("Rainfall") or 
            record.get("Rainfall_mm") or 
            record.get("Daily_rainfall") or
            record.get("rainfall") or 
            None
        )

        return {
            "state": state,
            "district": district,
            "date": date,
            "rainfall_mm": rainfall_mm
        }

    except Exception as e:
        return {"error": str(e)}


