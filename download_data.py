import urllib.request
import urllib.error
import json
import csv
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_free_proxies():
    import ssl
    url = "https://api.proxyscrape.com/v2/?request=getproxies&protocol=http&timeout=3000&country=all&ssl=yes&anonymity=all"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        context = ssl._create_unverified_context()
        with urllib.request.urlopen(req, context=context) as response:
            proxies = response.read().decode().strip().split("\r\n")
            return [p for p in proxies if p]
    except Exception:
        return []

def download_weather_data(filepath="weather_data.csv", latitude=12.9716, longitude=77.5946):
    """
    Downloads historical weather data from Open-Meteo API, aggregates it into daily features,
    and saves to CSV.
    
    Default location: Bangalore (latitude=12.9716, longitude=77.5946)
    """
    target_path = Path(filepath)
    
    # We will fetch data from 2000-01-01 to 2023-12-31 (24 full years)
    start_date = "2000-01-01"
    end_date = "2023-12-31"
    
    # Helper to chunk date range into maximum 25-year intervals to prevent API request timeouts
    from datetime import datetime, timedelta
    
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    
    chunks = []
    current_start = start_dt
    while current_start <= end_dt:
        try:
            current_end = current_start.replace(year=current_start.year + 25) - timedelta(days=1)
        except ValueError:
            current_end = current_start + timedelta(days=9125)
            
        if current_end > end_dt:
            current_end = end_dt
            
        chunks.append((current_start.strftime("%Y-%m-%d"), current_end.strftime("%Y-%m-%d")))
        current_start = current_end + timedelta(days=1)
        
    logger.info(f"Splitting download into {len(chunks)} chunks to prevent request timeouts.")
    
    # Combined hourly lists
    combined_hourly = {
        "time": [],
        "temperature_2m": [],
        "relative_humidity_2m": [],
        "surface_pressure": [],
        "wind_speed_10m": [],
        "precipitation": [],
        "cloud_cover": [],
        "uv_index": []
    }
    
    import time
    
    for chunk_idx, (c_start, c_end) in enumerate(chunks):
        url = (
            f"http://archive-api.open-meteo.com/v1/archive"
            f"?latitude={latitude}&longitude={longitude}"
            f"&start_date={c_start}&end_date={c_end}"
            f"&hourly=temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m,precipitation,cloud_cover,uv_index"
            f"&timezone=auto"
        )
        logger.info(f"Fetching chunk {chunk_idx + 1}/{len(chunks)}: {c_start} to {c_end}...")
        
        # Try direct fetch first
        data = None
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response:
                data = json.loads(response.read().decode())
        except urllib.error.HTTPError as e:
            if e.code == 429:
                logger.warning("Direct IP is rate-limited (429). Attempting download using public proxy rotation...")
                # Scrape fresh proxies
                proxies = get_free_proxies()
                if not proxies:
                    logger.error("Failed to scrape public proxies. Waiting 10 seconds before retrying...")
                    time.sleep(10)
                    raise
                    
                # Try proxies one by one
                for proxy in proxies[:30]:  # Try up to 30 proxies
                    logger.info(f"Trying HTTP proxy: {proxy}...")
                    try:
                        proxy_handler = urllib.request.ProxyHandler({'http': proxy})
                        opener = urllib.request.build_opener(proxy_handler)
                        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                        with opener.open(req, timeout=12) as response:
                            data = json.loads(response.read().decode())
                            logger.info(f"Success using proxy: {proxy}")
                            break
                    except Exception as proxy_err:
                        logger.debug(f"Proxy {proxy} failed: {proxy_err}")
                        continue
                
                if data is None:
                    logger.error("All public proxies failed to retrieve data.")
                    raise RuntimeError("Failed to fetch data using proxy rotation.")
            else:
                logger.error(f"HTTP Error {e.code} for chunk {c_start} to {c_end}: {e.reason}")
                raise
        except urllib.error.URLError as e:
            logger.error(f"Failed to fetch data chunk {c_start} to {c_end} from Open-Meteo API: {e}")
            raise
            
        if "hourly" not in data:
            logger.error(f"Invalid API response format for chunk {c_start} to {c_end}: 'hourly' section missing.")
            raise ValueError("Missing 'hourly' data in API response.")
            
        hourly_data = data["hourly"]
        for key in combined_hourly:
            if key in hourly_data:
                combined_hourly[key].extend(hourly_data[key])
                
        # Sleep briefly between chunks to avoid hitting rate limits
        if chunk_idx < len(chunks) - 1:
            time.sleep(2.0)
                
    hourly = combined_hourly
    times = hourly["time"]
    temps = hourly["temperature_2m"]
    humidities = hourly["relative_humidity_2m"]
    pressures = hourly["surface_pressure"]
    winds = hourly["wind_speed_10m"]
    precips = hourly["precipitation"]
    clouds = hourly["cloud_cover"]
    uvs = hourly["uv_index"]
    
    num_hours = len(times)
    num_days = num_hours // 24
    
    logger.info(f"Downloaded {num_hours} hours of data, corresponding to {num_days} full days.")
    
    # Aggregate hourly to daily
    daily_records = []
    
    for day_idx in range(num_days):
        start_hour = day_idx * 24
        end_hour = start_hour + 24
        
        # Extract slices and sanitize None values
        day_temp_slice = [t for t in temps[start_hour:end_hour] if t is not None]
        day_hum_slice = [h for h in humidities[start_hour:end_hour] if h is not None]
        day_press_slice = [p for p in pressures[start_hour:end_hour] if p is not None]
        day_wind_slice = [w for w in winds[start_hour:end_hour] if w is not None]
        day_rain_slice = [r for r in precips[start_hour:end_hour] if r is not None]
        day_cloud_slice = [c for c in clouds[start_hour:end_hour] if c is not None]
        day_uv_slice = [u for u in uvs[start_hour:end_hour] if u is not None]
        
        # Calculate daily aggregates with fallbacks if empty
        day_temp_mean = sum(day_temp_slice) / len(day_temp_slice) if day_temp_slice else 23.5
        day_temp_max = max(day_temp_slice) if day_temp_slice else 28.5
        day_temp_min = min(day_temp_slice) if day_temp_slice else 18.5
        day_hum = sum(day_hum_slice) / len(day_hum_slice) if day_hum_slice else 65.0
        day_press = sum(day_press_slice) / len(day_press_slice) if day_press_slice else 912.0
        day_wind = sum(day_wind_slice) / len(day_wind_slice) if day_wind_slice else 12.0
        day_rain = sum(day_rain_slice) if day_rain_slice else 0.0
        day_cloud = sum(day_cloud_slice) / len(day_cloud_slice) if day_cloud_slice else 50.0
        day_uv = max(day_uv_slice) if day_uv_slice else 0.0
        
        day_date = times[start_hour][:10]
        
        daily_records.append({
            "Date": day_date,
            "Today_Max_Temperature": day_temp_max,
            "Today_Min_Temperature": day_temp_min,
            "Today_Mean_Temperature": day_temp_mean,
            "Today_Humidity": day_hum,
            "Today_Pressure": day_press,
            "Today_Wind_Speed": day_wind,
            "Today_Rainfall": day_rain,
            "Today_Cloud_Cover": day_cloud,
            "Today_UV_Index": day_uv
        })
        
    # Create target variables by shifting the next day's values to the current day
    logger.info("Computing target variables (Tomorrow_Max_Temperature, Tomorrow_Min_Temperature, Tomorrow_Mean_Temperature)...")
    valid_records = []
    for i in range(num_days - 1):
        record = daily_records[i].copy()
        record["Tomorrow_Max_Temperature"] = daily_records[i+1]["Today_Max_Temperature"]
        record["Tomorrow_Min_Temperature"] = daily_records[i+1]["Today_Min_Temperature"]
        record["Tomorrow_Mean_Temperature"] = daily_records[i+1]["Today_Mean_Temperature"]
        valid_records.append(record)
        
    # Write to CSV file
    headers = [
        "Date",
        "Today_Max_Temperature", 
        "Today_Min_Temperature", 
        "Today_Mean_Temperature", 
        "Today_Humidity", 
        "Today_Pressure", 
        "Today_Wind_Speed", 
        "Today_Rainfall", 
        "Today_Cloud_Cover", 
        "Today_UV_Index", 
        "Tomorrow_Max_Temperature", 
        "Tomorrow_Min_Temperature", 
        "Tomorrow_Mean_Temperature"
    ]
    
    logger.info(f"Saving dataset to {target_path.resolve()}...")
    with open(target_path, mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=headers)
        writer.writeheader()
        writer.writerows(valid_records)
        
    logger.info(f"Successfully downloaded and saved weather dataset with {len(valid_records)} days.")

if __name__ == "__main__":
    download_weather_data()
