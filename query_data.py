
'''
Original file is located at
    https://colab.research.google.com/drive/1WDzQuVwX9HYdgQChgSMuG7EM0bU_9wyC

There are several goals for this project:

*   Be able to identify and outline crops within our ROI based on the USDA/NASS/CDL dataset from Google Earth Engine
*   Calculate the NDVI, soil moisture**, and weather data for each crop taken from above
*   Predict the crop yield using the NDVI value of each plot of land


**Soil moisture data currently not included in model.
'''
import ee
from google.colab import drive
import datetime
import pandas as pd
import os

ee.Authenticate()
ee.Initialize(project='ee-ndvi-change-detection')

# Define your Area of Interest (AOI) and the year of interest
aoi = ee.Geometry.Polygon([
    [-94.70937177214154, 41.15239644721922],
    [-93.93483563932904, 41.15239644721922],
    [-93.93483563932904, 41.42069426963426],
    [-94.70937177214154, 41.42069426963426],
    [-94.70937177214154, 41.15239644721922]
])

year = 2020

# Load the USDA/NASS/CDL dataset for the specified year and filter by the AOI
cdl = ee.Image(f'USDA/NASS/CDL/{year}').select('cropland').clip(aoi)

# Filter for corn areas
corn_areas = cdl.eq(1)

# Generate random points within the AOI
random_points = ee.FeatureCollection.randomPoints(region=aoi, points=100, seed=42)

# Correctly select a few random points for analysis
num_points_to_analyze = 5
selected_points = random_points.limit(num_points_to_analyze)

# Function to extract daily NDVI, temperature, precipitation, and soil moisture
def extract_daily_values(date_str):
    date = ee.Date(date_str)
    next_day = date.advance(1, 'day')

    # NDVI calculation
    ndvi = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')\
        .filterDate(date, next_day)\
        .filterBounds(aoi)\
        .select(['B8', 'B4'])\
        .mean()\
        .normalizedDifference(['B8', 'B4']).rename('NDVI')

    # Soil moisture extraction
    soil_moisture = ee.ImageCollection('NASA_USDA/HSL/SMAP_soil_moisture')\
        .filterDate(date, next_day)\
        .filterBounds(aoi)\
        .select('ssm')\
        .mean().rename('soil moisture')

    # Weather data extraction
    weather = ee.ImageCollection('ECMWF/ERA5/DAILY')\
        .filterDate(date, next_day)\
        .filterBounds(aoi)\
        .select(['mean_2m_air_temperature', 'total_precipitation'])\
        .mean().rename(['temperature', 'precipitation'])

    # Combine all bands
    combined = ndvi.addBands([soil_moisture, weather])

    # Sample the combined data at the selected points
    sampled = combined.sampleRegions(collection=selected_points, scale=30, geometries=True)

    # Export the sampled data
    task = ee.batch.Export.table.toDrive(
        collection=sampled,
        description='DailyCornData_' + date_str,
        folder='GEE_Exports',
        fileNamePrefix='daily_' + date_str,
        fileFormat='CSV'
    )

    task.start()

# Create a list of daily dates for January 2020
start_date = datetime.date(2020, 1, 1)
end_date = datetime.date(2020, 6, 30)
date_generated = [start_date + datetime.timedelta(days=x) for x in range((end_date-start_date).days + 1)]

# Iterate over the generated daily dates
for date_obj in date_generated:
    date_str = date_obj.strftime("%Y-%m-%d")
    extract_daily_values(date_str)

# @title Tracking task status for bigger jobs
tasks = ee.batch.Task.list()
print(tasks[0].status())

# @title Import the data
directory_path = '/content/drive/My Drive/GEE_Exports'

# List all files in the specified directory
all_files = os.listdir(directory_path)

# Initialize an empty DataFrame to hold the aggregated data
aggregated_df = pd.DataFrame()

for file_name in all_files:
    if file_name.endswith('.csv'):
        try:
            # Extract the base name without extension
            base_name = file_name.split('.')[0]
            date_str = base_name.split('_')[1]
            date_str = date_str.split(' ')[0].strip()
            # Convert date_str to a proper datetime format
            temp_df = pd.read_csv(os.path.join(directory_path, file_name))
            temp_df['Date'] = pd.to_datetime(date_str, format='%Y-%m-%d')
            aggregated_df = pd.concat([aggregated_df, temp_df], ignore_index=True)
        except Exception as e:
            print(f"Error processing {file_name}: {e}")

# Ensure the 'Date' column is in datetime format (this may be redundant but is a safeguard)
aggregated_df['Date'] = pd.to_datetime(aggregated_df['Date'])

print(aggregated_df.describe())
