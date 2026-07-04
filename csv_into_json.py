import pandas as pd

# Read CSV
df = pd.read_csv("weather_data.csv")

# Convert to JSON
df.to_json("weather_data.json", orient="records", indent=4)