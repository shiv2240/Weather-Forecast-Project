from numpy.polynomial import chebyshev
import csv
import numpy as np
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_weather_data(filename="weather_data.csv", num_days=1000, seed=None):
    """
    Generates synthetic Bangalore-like weather data for multiple features and saves to CSV.
    Uses pathlib and logging.
    """
    logger.info(f"Generating synthetic weather data (Bangalore-like) for {num_days} days...")
    
    if seed is not None:
        np.random.seed(seed)
        logger.info(f"Random seed set to {seed}")
    else:
        logger.info("No random seed specified (non-deterministic).")
        
    days = np.arange(num_days)
    
    # 1. Temperature: seasonal sine wave + noise (Bangalore avg temperature is around 23.5°C)
    temp_mean_today = 23.5 + 4.5 * np.sin(2.0 * np.pi * (days - 80) / 365.0) + np.random.normal(0, 1.2, num_days)
    # Bangalore diurnal temperature range is typically 8°C to 12°C
    temp_max_today = temp_mean_today + 5.0 + np.random.normal(0, 0.8, num_days)
    temp_min_today = temp_mean_today - 5.0 + np.random.normal(0, 0.8, num_days)
    
    # 2. Humidity: inverse seasonal pattern + noise (monsoon is humid, summer is drier)
    humidity_today = 65.0 - 15.0 * np.sin(2.0 * np.pi * (days - 80) / 365.0) + np.random.normal(0, 5.0, num_days)
    humidity_today = np.clip(humidity_today, 20.0, 100.0)
    
    # 3. Pressure: typical Bangalore pressure around 912 hPa (due to 920m elevation)
    pressure_today = 912.0 + np.random.normal(0, 4.0, num_days)
    
    # 4. Wind Speed: positive value, average 12 km/h
    wind_today = np.abs(12.0 + np.random.normal(0, 4.0, num_days))
    
    # 5. Rainfall: sparse, exponential distribution on rainy days
    rain_chance = 0.22
    rain_today = np.where(
        np.random.rand(num_days) < rain_chance,
        np.random.exponential(8.0, num_days),
        0.0
    )
    
    # 6. Cloud Cover: percentage [0, 100]
    cloud_today = 50.0 - 15.0 * np.sin(2.0 * np.pi * (days - 80) / 365.0) + np.random.normal(0, 18.0, num_days)
    cloud_today = np.clip(cloud_today, 0.0, 100.0)
    
    # 7. UV Index: seasonal pattern, typical range [0, 12]
    uv_today = 8.0 + 3.0 * np.sin(2.0 * np.pi * (days - 80) / 365.0) + np.random.normal(0, 1.0, num_days)
    uv_today = np.clip(uv_today, 0.0, 12.0)
    
    # Define ground truth relationship for target Tomorrow_Mean_Temperature:
    # Tomorrow's Mean Temp = 0.8 * Today's Mean Temp - 0.03 * Today's Hum - 0.05 * Wind + 0.15 * UV - 0.2 * Rain + 0.01 * (Pressure - 912.0) + 5.0 + Noise
    true_w_mean_temp = 0.8
    true_w_mean_hum = -0.03
    true_w_mean_press = 0.01
    true_w_mean_wind = -0.05
    true_w_mean_rain = -0.2
    true_w_mean_uv = 0.15
    true_mean_bias = 5.0
    noise_mean = np.random.normal(0, 0.8, num_days)
    temp_mean_tomorrow = (
        (true_w_mean_temp * temp_mean_today) +
        (true_w_mean_hum * humidity_today) +
        (true_w_mean_press * (pressure_today - 912.0)) +
        (true_w_mean_wind * wind_today) +
        (true_w_mean_rain * rain_today) +
        (true_w_mean_uv * uv_today) +
        true_mean_bias +
        noise_mean
    )
    
    # Define ground truth relationship for target Tomorrow_Max_Temperature:
    # Tomorrow's Max Temp = 0.85 * Today's Max Temp - 0.02 * Today's Hum - 0.03 * Wind + 0.20 * UV - 0.1 * Rain + 0.01 * (Pressure - 912.0) + 4.0 + Noise
    true_w_max_temp = 0.85
    true_w_max_hum = -0.02
    true_w_max_press = 0.01
    true_w_max_wind = -0.03
    true_w_max_rain = -0.1
    true_w_max_uv = 0.20
    true_max_bias = 4.0
    noise_max = np.random.normal(0, 1.0, num_days)
    temp_max_tomorrow = (
        (true_w_max_temp * temp_max_today) +
        (true_w_max_hum * humidity_today) +
        (true_w_max_press * (pressure_today - 912.0)) +
        (true_w_max_wind * wind_today) +
        (true_w_max_rain * rain_today) +
        (true_w_max_uv * uv_today) +
        true_max_bias +
        noise_max
    )
    
    # Define ground truth relationship for target Tomorrow_Min_Temperature:
    # Tomorrow's Min Temp = 0.75 * Today's Min Temp - 0.04 * Today's Hum - 0.08 * Wind + 0.05 * UV - 0.3 * Rain + 0.01 * (Pressure - 912.0) + 6.0 + Noise
    true_w_min_temp = 0.75
    true_w_min_hum = -0.04
    true_w_min_press = 0.01
    true_w_min_wind = -0.08
    true_w_min_rain = -0.3
    true_w_min_uv = 0.05
    true_min_bias = 6.0
    noise_min = np.random.normal(0, 0.6, num_days)
    temp_min_tomorrow = (
        (true_w_min_temp * temp_min_today) +
        (true_w_min_hum * humidity_today) +
        (true_w_min_press * (pressure_today - 912.0)) +
        (true_w_min_wind * wind_today) +
        (true_w_min_rain * rain_today) +
        (true_w_min_uv * uv_today) +
        true_min_bias +
        noise_min
    )
    
    import datetime
    start_date_base = datetime.date(2020, 1, 1)
    dates = [(start_date_base + datetime.timedelta(days=int(i))).strftime("%Y-%m-%d") for i in range(num_days)]
    
    # Save to CSV
    target_path = Path(filename)
    header = [
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
    
    data = np.column_stack((
        temp_max_today, 
        temp_min_today, 
        temp_mean_today, 
        humidity_today, 
        pressure_today, 
        wind_today, 
        rain_today, 
        cloud_today, 
        uv_today, 
        temp_max_tomorrow, 
        temp_min_tomorrow, 
        temp_mean_tomorrow
    ))
    
    logger.info(f"Saving synthetic dataset to {target_path.resolve()}...")
    with open(target_path, mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)
        for date, row in zip(dates, data):
            writer.writerow([date] + list(row))
        
    logger.info("Successfully generated synthetic weather dataset.")

if __name__ == "__main__":
    generate_weather_data()
