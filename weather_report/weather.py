import requests
from urllib.parse import urlencode

BASE_URL = "https://api.openweathermap.org/data/2.5/weather"


def build_url(city=None, lat=None, lon=None, units="metric", lang="en", api_key=None):
    """Build OpenWeatherMap API request URL."""
    params = {"units": units, "lang": lang}
    if city:
        params["q"] = city
    elif lat is not None and lon is not None:
        params["lat"] = lat
        params["lon"] = lon
    else:
        raise ValueError("Either city or (lat, lon) must be provided.")
    if api_key:
        params["appid"] = api_key
    return f"{BASE_URL}?{urlencode(params)}"


def fetch_weather(url: str) -> dict:
    """Fetch weather data from API."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"Network error while contacting API: {exc}") from exc


def parse_weather(data: dict) -> dict:
    """Extract and format relevant weather details."""
    try:
        weather = data["weather"][0]["description"]
        temp = data["main"]["temp"]
        feels_like = data["main"]["feels_like"]
        humidity = data["main"]["humidity"]
        pressure = data["main"]["pressure"]
        wind_speed = data["wind"]["speed"]
        city = data.get("name", "Unknown")
        country = data.get("sys", {}).get("country", "")

        rain_1h = data.get("rain", {}).get("1h", 0.0)
        snow_1h = data.get("snow", {}).get("1h", 0.0)
        precipitation_value = rain_1h + snow_1h

        return {
            "city": city,
            "country": country,
            "weather": weather,
            "temperature_C": round(temp, 1),
            "feels_like_C": round(feels_like, 1),
            "humidity_percent": humidity,
            "pressure_hPa": pressure,
            "wind_speed_m_s": round(wind_speed, 1),
            "precipitation_mm": round(precipitation_value, 1),
        }
    except KeyError as e:
        raise KeyError(f"Missing expected field in response: {e}") from e
