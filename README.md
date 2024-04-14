# Google-Earth-Engine-Crop-Yield-Prediction
Using a random forest regression model to predict annual yield for corn crops using Google Earth Engine's satellite data. 

## Table of Contents
* [Data](#data)
* [Model](#model)
* [Results](#results)

### Data
All of the data used in this project was taken from the [Google Earth Engine Data Catalog](https://developers.google.com/earth-engine/datasets). I will hyperlink each library I used in this project as I mention them.

The first step in this project was to select an area of interest. I used the [geometry.polygon](https://developers.google.com/earth-engine/apidocs/ee-geometry-polygon) method because it fits well into how I planned to extract data from the Earth Engine libraries, which I explain below. I selected a location with several corn fields, but the aoi can easily be changed to someplace else by changing the coordinates.
```python
aoi = ee.Geometry.Polygon([
    [-94.70937177214154, 41.15239644721922],
    [-93.93483563932904, 41.15239644721922],
    [-93.93483563932904, 41.42069426963426],
    [-94.70937177214154, 41.42069426963426],
    [-94.70937177214154, 41.15239644721922]
])
```

With my area of interest defined, the next step was to classify the crop fields within the aoi. Luckily there is a library that classifies all crops in the United States by assigning a class to it named [USDA/NASS/CDL](https://developers.google.com/earth-engine/datasets/catalog/USDA_NASS_CDL). In the snippet below I selected all croplands in my aoi with class = 1 (corn). This can also easily be changed to select any crop in the classlist. 
```python
cdl = ee.Image(f'USDA/NASS/CDL/{year}').select('cropland').clip(aoi)
corn_mask = cdl.eq(1)
```

To randomly select crops I first created 100 random points in the aoi. The points were assigned a new feature, 'corn' with a value of 1 if over a corn field or 0 if otherwise. I then filtered the points to only ones with corn set to 1 and limited the count to a max of 5 points. 
```python
random_points = ee.FeatureCollection.randomPoints(region=aoi, points=100, seed=42)

corn_points = random_points.map(lambda feature: feature.set({'corn': corn_mask.reduceRegion(ee.Reducer.first(), feature.geometry(), 30)}))
corn_points = corn_points.filter(ee.Filter.eq('corn', 1))

num_points_to_analyze = 5
selected_points = corn_points.limit(num_points_to_analyze)
```

Instead of showing how I extracted all of the features for these points, I'll only show one, but they were all done in almost the same way. To extract soil moisture data for the points I used the [NASA_USDA/HSL/SMAP_soil_moisture](https://explorer.earthengine.google.com/#detail/NASA_USDA%2FHSL%2FSMAP_soil_moisture) library. Notice how the [filterBounds](https://developers.google.com/earth-engine/apidocs/ee-imagecollection-filterbounds) method  utilizes the aoi geometry we drew above. 

```python
    soil_moisture = ee.ImageCollection('NASA_USDA/HSL/SMAP_soil_moisture')\
        .filterDate(date, next_day)\
        .filterBounds(aoi)\
        .select('ssm')\
        .mean().rename('soil moisture')
```

I wrapped the extraction of each feature into a function and iterated through each day of a given date range. Finally, I took all of this data and stored it into a big dataframe with daily values for each point. 
```python
def extract_daily_values(date_str):
    date = ee.Date(date_str)
    next_day = date.advance(1, 'day')
    ...

for date_obj in date_generated:
    date_str = date_obj.strftime("%Y-%m-%d")
    extract_daily_values(date_str)
```

### Model
