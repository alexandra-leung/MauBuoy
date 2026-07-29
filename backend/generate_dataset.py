import pandas as pd
import numpy as np

# Set seed for reproducibility
np.random.seed(42)

# Define dataset size
n = 10000  # Change this to 1000, 5000, etc., as needed

# 1. Geographic bounds for coral reefs
latitude = np.random.uniform(-30, 30, n)
longitude = np.random.uniform(-180, 180, n)

# 2. Depth (shallow reefs bleach more easily)
depth_m = np.random.uniform(1, 30, n)

# 3. Year (1990 to 2023)
year = np.random.randint(1990, 2024, n)

# 4. Sea Surface Temperature (with slight warming trend over years)
base_sst = np.random.uniform(26.0, 30.0, n)
sst_trend = (year - 1990) * 0.02  
sea_surface_temperature_c = np.clip(base_sst + sst_trend + np.random.normal(0, 0.5, n), 25.0, 33.0)

# 5. Degree Heating Weeks (spikes when SST > 29°C)
dhw = np.maximum(0, (sea_surface_temperature_c - 29.0) * 3.0 + np.random.normal(0, 1.5, n))
dhw = np.clip(dhw, 0, 15.0)

# 6. pH (slight decreasing trend due to ocean acidification)
base_ph = np.random.uniform(7.9, 8.2, n)
ph_trend = (year - 1990) * -0.0015
ph = np.clip(base_ph + ph_trend + np.random.normal(0, 0.05, n), 7.7, 8.3)

# 7. Salinity & Turbidity
salinity_psu = np.random.uniform(33.0, 37.0, n)
turbidity_ntu = np.random.uniform(0.5, 15.0, n)

# 8. Coral Cover (decreases with higher DHW and lower pH)
base_coral_cover = np.random.uniform(30, 80, n)
coral_cover_percent = np.clip(base_coral_cover - (dhw * 1.5) - ((8.1 - ph) * 15) + np.random.normal(0, 5, n), 5, 90)

# 9. Marine Heatwave (highly correlated with high DHW)
marine_heatwave = (dhw > 4) & (np.random.uniform(0, 1, n) > 0.2)
marine_heatwave = marine_heatwave | ((dhw > 2) & (np.random.uniform(0, 1, n) > 0.8))

# 10. Bleaching Risk (0-100)
risk_score = (dhw * 10) + ((sea_surface_temperature_c - 28.5) * 15) + (marine_heatwave * 25) - (depth_m * 0.8) + (turbidity_ntu * 0.5)
risk_score = np.clip(risk_score + np.random.normal(0, 15, n), 0, 100)

# 11. Bleaching Label (0/1) - Probabilistic based on risk score for realistic imbalance (~25% positive)
prob_bleach = 1 / (1 + np.exp(-(risk_score - 30) / 12))
bleaching_label = (np.random.uniform(0, 1, n) < prob_bleach).astype(int)

# Create DataFrame
df = pd.DataFrame({
    'Latitude': np.round(latitude, 4),
    'Longitude': np.round(longitude, 4),
    'Depth_m': np.round(depth_m, 2),
    'Year': year,
    'Sea_Surface_Temperature_C': np.round(sea_surface_temperature_c, 2),
    'Degree_Heating_Weeks': np.round(dhw, 2),
    'pH': np.round(ph, 3),
    'Salinity_PSU': np.round(salinity_psu, 2),
    'Turbidity_NTU': np.round(turbidity_ntu, 2),
    'Coral_Cover_Percent': np.round(coral_cover_percent, 2),
    'Marine_Heatwave': marine_heatwave,
    'Bleaching_Risk': np.round(risk_score, 2),
    'Bleaching_Label': bleaching_label
})

# Save to CSV
df.to_csv('synthetic_bleaching_dataset.csv', index=False)
print("Dataset saved to 'synthetic_bleaching_dataset.csv'")
print(f"Shape: {df.shape} | Bleaching Label Distribution:\n{df['Bleaching_Label'].value_counts(normalize=True)}")