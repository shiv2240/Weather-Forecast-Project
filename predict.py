import sys
import csv
import pickle
import argparse
import logging
import datetime
import urllib.request
import urllib.parse
import json
from pathlib import Path
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_resources():
    """Loads the saved model and scaler parameters from pickle files."""
    model_path = Path("model.pkl")
    scaler_path = Path("scaler.pkl")
    
    if not model_path.exists() or not scaler_path.exists():
        logger.error("Model or Scaler files not found. Please run train.py first to train the model.")
        sys.exit(1)
        
    with open(model_path, "rb") as f:
        model = pickle.load(f)
        
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
        
    return model, scaler

def geocode_city(city_name):
    """Geocodes a city name to latitude and longitude using Open-Meteo Geocoding API."""
    city_name_stripped = city_name.strip()
    if city_name_stripped.lower() == "bangalore":
        search_name = "Bengaluru"
    else:
        search_name = city_name_stripped
        
    encoded_city = urllib.parse.quote(search_name)
    url = f"https://geocoding-api.open-meteo.com/v1/search?name={encoded_city}&count=1"
    
    logger.info(f"Resolving coordinates for '{city_name_stripped}'...")
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
    except Exception as e:
        logger.error(f"Geocoding request failed: {e}")
        return None, None, None
        
    results = data.get("results")
    if not results:
        logger.error(f"No results found for city: {city_name}")
        return None, None, None
        
    loc = results[0]
    resolved_name = f"{loc.get('name')}, {loc.get('country')}"
    return loc.get("latitude"), loc.get("longitude"), resolved_name

def parse_date_string(date_str):
    """Parses various date formats to a datetime.date object. Supports 'tomorrow', 'today', and common date strings."""
    date_str = date_str.strip().lower()
    
    if date_str == "today":
        return datetime.date.today()
    elif date_str == "tomorrow":
        return datetime.date.today() + datetime.timedelta(days=1)
    elif date_str == "yesterday":
        return datetime.date.today() - datetime.timedelta(days=1)
        
    # Attempt parsing with common formats
    formats = [
        "%Y-%m-%d",      # 2026-07-15
        "%d-%m-%Y",      # 15-07-2026
        "%d/%m/%Y",      # 15/07/2026
        "%d %B %Y",      # 15 July 2026
        "%d %b %Y",      # 15 Jul 2026
        "%B %d, %Y",     # July 15, 2026
        "%b %d, %Y"      # Jul 15, 2026
    ]
    
    for fmt in formats:
        try:
            return datetime.datetime.strptime(date_str, fmt).date()
        except ValueError:
            continue
            
    # Custom parser for freeform inputs (e.g. "15 July 2026" or "July 15 2026" without commas)
    try:
        cleaned_str = date_str.replace(",", " ").replace("-", " ").replace("/", " ")
        tokens = cleaned_str.split()
        if len(tokens) == 3:
            month_map = {
                "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
                "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
                "aug": 8, "august": 8, "sep": 9, "september": 9, "oct": 10, "october": 10,
                "nov": 11, "november": 11, "dec": 12, "december": 12
            }
            t0, t1, t2 = tokens[0], tokens[1], tokens[2]
            
            # Case: DD Month YYYY
            if t1 in month_map:
                day = int(t0)
                month = month_map[t1]
                year = int(t2)
                return datetime.date(year, month, day)
            # Case: Month DD YYYY
            elif t0 in month_map:
                month = month_map[t0]
                day = int(t1)
                year = int(t2)
                return datetime.date(year, month, day)
    except Exception:
        pass
        
    return None

def get_latest_date_in_csv(filename="weather_data.csv"):
    """Reads the CSV and returns the date of the last row, or yesterday's date as fallback."""
    path = Path(filename)
    if not path.exists():
        return (datetime.date.today() - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
        
    last_date = None
    try:
        with open(path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("Date"):
                    last_date = row["Date"]
    except Exception as e:
        logger.warning(f"Could not read latest date from CSV: {e}")
        
    if not last_date:
        return (datetime.date.today() - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
    return last_date

def find_date_in_csv(target_date, filename="weather_data.csv"):
    """Searches for a date in weather_data.csv and returns the feature list if found."""
    path = Path(filename)
    if not path.exists():
        return None
        
    try:
        with open(path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("Date") == target_date:
                    features = [
                        float(row["Today_Max_Temperature"]),
                        float(row["Today_Min_Temperature"]),
                        float(row["Today_Mean_Temperature"]),
                        float(row["Today_Humidity"]),
                        float(row["Today_Pressure"]),
                        float(row["Today_Wind_Speed"]),
                        float(row["Today_Rainfall"]),
                        float(row["Today_Cloud_Cover"]),
                        float(row["Today_UV_Index"])
                    ]
                    return np.array(features).reshape(1, -1)
    except Exception as e:
        logger.warning(f"Error reading CSV: {e}")
    return None

def fetch_date_from_api(target_date, latitude, longitude):
    """Fetches weather observations for a specific date dynamically from Open-Meteo API."""
    try:
        target_dt = datetime.datetime.strptime(target_date, "%Y-%m-%d").date()
    except ValueError:
        raise ValueError(f"Invalid date format: {target_date}. Please use YYYY-MM-DD.")
        
    today = datetime.date.today()
    delta_days = (target_dt - today).days
    
    # Decide which API endpoint to use based on date range
    if -90 <= delta_days <= 16:
        url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={latitude}&longitude={longitude}"
            f"&start_date={target_date}&end_date={target_date}"
            f"&hourly=temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m,precipitation,cloud_cover,uv_index"
            f"&timezone=auto"
        )
        logger.info(f"Date {target_date} is recent/upcoming. Querying Forecast API...")
    else:
        url = (
            f"https://archive-api.open-meteo.com/v1/archive"
            f"?latitude={latitude}&longitude={longitude}"
            f"&start_date={target_date}&end_date={target_date}"
            f"&hourly=temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m,precipitation,cloud_cover,uv_index"
            f"&timezone=auto"
        )
        logger.info(f"Date {target_date} is historical. Querying Archive API...")
    
    logger.info(f"Retrieving weather data on-the-fly from Open-Meteo...")
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
    except Exception as e:
        logger.error(f"Failed to fetch data from Open-Meteo API: {e}")
        return None
        
    if "hourly" not in data:
        logger.warning(f"No hourly data returned in API response for date {target_date}.")
        return None
        
    hourly = data["hourly"]
    temps = hourly.get("temperature_2m", [])
    humidities = hourly.get("relative_humidity_2m", [])
    pressures = hourly.get("surface_pressure", [])
    winds = hourly.get("wind_speed_10m", [])
    precips = hourly.get("precipitation", [])
    clouds = hourly.get("cloud_cover", [])
    uvs = hourly.get("uv_index", [])
    
    # Sanitize and aggregate slices
    day_temp_slice = [t for t in temps if t is not None]
    day_hum_slice = [h for h in humidities if h is not None]
    day_press_slice = [p for p in pressures if p is not None]
    day_wind_slice = [w for w in winds if w is not None]
    day_rain_slice = [r for r in precips if r is not None]
    day_cloud_slice = [c for c in clouds if c is not None]
    day_uv_slice = [u for u in uvs if u is not None]
    
    day_temp_mean = sum(day_temp_slice) / len(day_temp_slice) if day_temp_slice else 23.5
    day_temp_max = max(day_temp_slice) if day_temp_slice else 28.5
    day_temp_min = min(day_temp_slice) if day_temp_slice else 18.5
    day_hum = sum(day_hum_slice) / len(day_hum_slice) if day_hum_slice else 65.0
    day_press = sum(day_press_slice) / len(day_press_slice) if day_press_slice else 912.0
    day_wind = sum(day_wind_slice) / len(day_wind_slice) if day_wind_slice else 12.0
    day_rain = sum(day_rain_slice) if day_rain_slice else 0.0
    day_cloud = sum(day_cloud_slice) / len(day_cloud_slice) if day_cloud_slice else 50.0
    day_uv = max(day_uv_slice) if day_uv_slice else 0.0
    
    features = [
        day_temp_max,
        day_temp_min,
        day_temp_mean,
        day_hum,
        day_press,
        day_wind,
        day_rain,
        day_cloud,
        day_uv
    ]
    return np.array(features).reshape(1, -1)

def get_interactive_input(scaler):
    """Prompts the user interactively for each weather feature, using mean as default."""
    mean = scaler["mean"]
    features = scaler["features"]
    
    descriptions = {
        "Today_Max_Temperature": ("Today's Maximum Temperature", "°C"),
        "Today_Min_Temperature": ("Today's Minimum Temperature", "°C"),
        "Today_Mean_Temperature": ("Today's Average Temperature", "°C"),
        "Today_Humidity": ("Today's Humidity", "%"),
        "Today_Pressure": ("Today's Atmospheric Pressure", "hPa"),
        "Today_Wind_Speed": ("Today's Wind Speed", "km/h"),
        "Today_Rainfall": ("Today's Rainfall", "mm"),
        "Today_Cloud_Cover": ("Today's Cloud Cover", "%"),
        "Today_UV_Index": ("Today's UV Index", "")
    }
    
    inputs = []
    print("\n--- Enter Today's Weather Observations ---")
    print("(Press ENTER to use the historical average default)")
    
    for i, feature in enumerate(features):
        desc, unit = descriptions.get(feature, (feature, ""))
        default_val = mean[i]
        unit_str = f" in {unit}" if unit else ""
        
        while True:
            user_input = input(f"{desc}{unit_str} [Default: {default_val:.2f}]: ").strip()
            if not user_input:
                inputs.append(default_val)
                break
            try:
                val = float(user_input)
                inputs.append(val)
                break
            except ValueError:
                print("Invalid input. Please enter a valid number.")
                
    return np.array(inputs).reshape(1, -1)

def main():
    parser = argparse.ArgumentParser(description="Predict weather using trained linear regression.")
    parser.add_argument("--city", type=str, help="Target city name (e.g. Bangalore, Delhi)")
    parser.add_argument("--date", type=str, help="Forecast target date (e.g. 'Tomorrow', '15 July 2026')")
    parser.add_argument("--temp-max", type=float, help="Today's maximum temperature in °C")
    parser.add_argument("--temp-min", type=float, help="Today's minimum temperature in °C")
    parser.add_argument("--temp-mean", type=float, help="Today's average temperature in °C")
    parser.add_argument("--humidity", type=float, help="Today's humidity in %")
    parser.add_argument("--pressure", type=float, help="Today's atmospheric pressure in hPa")
    parser.add_argument("--wind", type=float, help="Today's wind speed in km/h")
    parser.add_argument("--rainfall", type=float, help="Today's rainfall in mm")
    parser.add_argument("--cloud", type=float, help="Today's cloud cover in %")
    parser.add_argument("--uv", type=float, help="Today's UV index")
    
    args = parser.parse_args()
    
    # Load models and scaler
    models_dict, scaler = load_resources()
    model_max = models_dict["max"]
    model_min = models_dict["min"]
    model_mean = models_dict["mean"]
    metrics_dict = models_dict.get("metrics", {})
    
    # Check if CLI parameters for manual values were provided
    cli_manual_inputs = [
        args.temp_max, args.temp_min, args.temp_mean,
        args.humidity, args.pressure, args.wind, args.rainfall, args.cloud, args.uv
    ]
    
    features_input = None
    resolved_city = None
    target_date_str = None
    source = "manual entry"
    
    if all(v is not None for v in cli_manual_inputs):
        features_input = np.array(cli_manual_inputs).reshape(1, -1)
    else:
        # Resolve target City
        entered_city = args.city.strip() if args.city else None
        if not entered_city:
            user_city = input("Enter city [Default: Bangalore]: ").strip()
            entered_city = user_city if user_city else "Bangalore"
            
        # Geocode City
        latitude, longitude, resolved_city = geocode_city(entered_city)
        if not resolved_city:
            logger.error("Could not resolve city coordinates. Falling back to Bangalore.")
            latitude, longitude, resolved_city = 12.9716, 77.5946, "Bangalore, India"
            
        # Resolve Target Date
        entered_date = args.date.strip() if args.date else None
        if not entered_date:
            user_date = input("Forecast date (e.g. 'Tomorrow', '15 July 2026')\n[Default: Tomorrow]: ").strip()
            entered_date = user_date if user_date else "Tomorrow"
            
        forecast_date_obj = parse_date_string(entered_date)
        if not forecast_date_obj:
            logger.error(f"Could not parse forecast date: '{entered_date}'. Falling back to Tomorrow.")
            forecast_date_obj = datetime.date.today() + datetime.timedelta(days=1)
            
        target_date_str = forecast_date_obj.strftime("%Y-%m-%d")
        
        # Calculate Observation Date (X features date is Forecast Date - 1 day)
        observation_date_obj = forecast_date_obj - datetime.timedelta(days=1)
        observation_date_str = observation_date_obj.strftime("%Y-%m-%d")
        
        logger.info(f"Forecast Target Date: {target_date_str} | Fetching observations for: {observation_date_str}")
        
        # 1. Search locally in CSV if city is Bangalore
        is_bangalore = "bangalore" in resolved_city.lower()
        if is_bangalore:
            features_input = find_date_in_csv(observation_date_str)
            if features_input is not None:
                source = f"local database (Date: {observation_date_str})"
                
        # 2. Fetch dynamically from API if not found locally
        if features_input is None:
            logger.info("Retrieving observations from Open-Meteo API...")
            try:
                features_input = fetch_date_from_api(observation_date_str, latitude, longitude)
            except Exception as e:
                logger.error(f"API Fetch Error: {e}")
                
            if features_input is not None:
                source = f"Open-Meteo API (Date: {observation_date_str})"
            else:
                logger.warning("Could not automatically retrieve weather observations.")
                features_input = get_interactive_input(scaler)
                source = "manual entry fallback"
                
    # Scale features
    mean = scaler["mean"]
    std = scaler["std"]
    features_scaled = (features_input - mean) / std
    
    # Run predictions
    pred_max = model_max.predict(features_scaled)[0]
    pred_min = model_min.predict(features_scaled)[0]
    pred_mean = model_mean.predict(features_scaled)[0]
    
    # Display the result beautifully
    display_date = target_date_str if target_date_str else "Tomorrow"
    print("\n" + "="*90)
    print(f"                       WEATHER FORECAST FOR {resolved_city.upper()}                        ")
    print(f"                               Target Date: {display_date}                               ")
    print("="*90)
    print(f"Observations Source: {source}")
    print("Today's Observation Values (Features):")
    for i, feature in enumerate(scaler["features"]):
        print(f"  - {feature.replace('Today_', ''):22s}: {features_input[0][i]:.2f}")
    print("-"*90)
    
    def format_acc(metrics):
        if not metrics:
            return "N/A"
        mae, rmse, r2, acc_1c, acc_1_5c = metrics
        return f"MAE: {mae:.2f}°C | Acc(±1.0°C): {acc_1c:.1f}% | Acc(±1.5°C): {acc_1_5c:.1f}%"
        
    max_acc = format_acc(metrics_dict.get("max"))
    min_acc = format_acc(metrics_dict.get("min"))
    mean_acc = format_acc(metrics_dict.get("mean"))
    
    print(f"Predictions for {display_date}:")
    print(f"  >>> Max Temperature     : {pred_max:5.1f}°C   [{max_acc}]")
    print(f"  >>> Min Temperature     : {pred_min:5.1f}°C   [{min_acc}]")
    print(f"  >>> Average Temperature : {pred_mean:5.1f}°C   [{mean_acc}]")
    print("="*90)
    
    # Notice for non-Bangalore runs
    if not is_bangalore:
        print("\n[NOTE] This model was trained on Bangalore weather data. Predictions for other cities")
        print("       are computed using z-score normalization but may be less accurate due to climate variation.")
        print("="*90)
    print("\n")

if __name__ == "__main__":
    main()
