# Google-Earth-Engine-Crop-Yield-Prediction
Using a random forest regression model to predict annual yield for corn crops using Google Earth Engine's satellite data. 

## Table of Contents
* [Motivation](#motivation)
* [Metrics](#metrics)
* [Data](#data)
* [Model](#model)
* [Results](#results)

### Motivation
Several machine learning methods are already being utilized in crop managements such as a classifier that classifies crops as harvestable or not harvestable [1] and a model for the estimation of grassland biomass [2]. The applications seem to be endless. In this project I use the [naturalized difference vegetation index](https://en.wikipedia.org/wiki/Normalized_difference_vegetation_index) to quantify a crop's health, and several features to predict the NDVI over a given time period such as soil moisture, temperature and precipitation. I used corn in this project because of how much there is, but the crop being tested can easily be changed within the script. 

This project also gave me the opportunity to extract and clean a large amount of data instead of searching for it through Kaggle. I used the Pandas library to get my data ready for the model, and Scikit Learn's Random Forest Regression model for my predictions. 

Below is as explanation of the Google Earth Engine specific features and machine learning model I used for this project. Please enjoy! 

### Metrics
The primary metric in this project is the Normalized Difference Vegetation Index (NDVI), a widely used index for assessing vegetation health and density via sensor data. NDVI measures the differential levels of red (absorbed by healthy plants) and near-infrared (reflected by healthy plants) light. Typically, higher NDVI values indicate healthier vegetation, which correlates with enhanced photosynthesis capabilities. This index is invaluable for farmers and agronomists to forecast crop yields, aiding in effective planning and marketing of produce.

#### Key Features of the Model
* **soil moisture:** Adequate soil moisture is necessary for the absorption of nutrients and proper physiological functions of plants. Decreased soil moisture can lead to reduced plant height, leaf number, and total leaf area[3].
* **Temperature:** A critical factor in crop development and function, altering enzyme functions within a leaf and triggering changes in developmental growth stages linked to crop yield[4].
* **Precipitation:** Essential for maintaining plant health, with variations affecting stress levels and NDVI[5].

I also considered land coverage or vegetation type in the analysis, which might affect NDVI readings differently across crop types. This presents an interesting avenue for future experiments to compare results across different crops.

Other potential features not included here are sunlight exposure, soil pH, and humidity.

### Data Aqcuisition
All of the data used in this project was taken from the [Google Earth Engine Data Catalog](https://developers.google.com/earth-engine/datasets). I will hyperlink each library I used in this project as I mention them.

The initial step involved selecting an area of interest (AOI). I used the [geometry.polygon](https://developers.google.com/earth-engine/apidocs/ee-geometry-polygon) method because it fits well into how I planned to extract data from the Earth Engine libraries, which I explain below.
```python
aoi = ee.Geometry.Polygon([
    [-94.70937177214154, 41.15239644721922],
    [-93.93483563932904, 41.15239644721922],
    [-93.93483563932904, 41.42069426963426],
    [-94.70937177214154, 41.42069426963426],
    [-94.70937177214154, 41.15239644721922]
])
```

Using the USDA/NASS/CDL library, I classified crop fields within this AOI and selected corn crops specifically. To gather randomized data points across these fields, I used:
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

I wrapped the extraction of each feature into a function and iterated through each day in a date range. Finally, I took all of this data and stored it into a big dataframe with daily values for each point. 
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
Although I sought an accurate prediction of NDVI, I was also interested in understanding the model's inference - that is, learning about the features and their roles in the prediction process. To acquire this information, I needed a highly interpretable model. A more complex model might have yielded better predictions, but it would have obscured the importance of individual features. Conversely, a simpler model would have offered greater clarity about the features, but might have compromised accuracy by failing to capture the data's intricacies as effectively as a more flexible model could. 

This is where the [ensemble learning](https://en.wikipedia.org/wiki/Ensemble_learning) [random forest regression](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html) model fit perfectly. It is flexible, controls over-fitting through the use of many trees, and provides valuable information about feature importance.

I iterated through each point in my aoi and trained a model based on each location's data. 
```python
for location_id in location_ids:
```

After splitting my data into training and test data, I imputed them to remove any null values still present in my data. I waited until after I separated the data to prevent the averages from one portion of the data affecting the other. 
```python
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        imputer = KNNImputer(n_neighbors=5)
        X_train_imputed = imputer.fit_transform(X_train)
        X_test_imputed = imputer.transform(X_test)
```

### Results
Because there were several locations, I stored the metrics of each in a dictionary. The metrics I chose are [root mean squared error](https://en.wikipedia.org/wiki/Root-mean-square_deviation), [mean squared error](https://en.wikipedia.org/wiki/Mean_squared_error), and [r^2](https://en.wikipedia.org/wiki/Coefficient_of_determination). 
```python
        rmse = sqrt(mean_squared_error(y_test, y_test_pred))
        mae = mean_absolute_error(y_test, y_test_pred)
        
        r2_train = r2_score(y_train, y_train_pred)
        r2_test = r2_score(y_test, y_test_pred)

        metrics_dict[location_id] = {'RMSE': rmse, 'MAE': mae, 'R2_test': r2_test, 'R2_train': r2_train}
```
One red flag that caught my attention was that the coefficient of determination for the training model was very high, while it was somewhat lower for the test model. This discrepancy suggests that my model might be overfitting the training data. One possible solution would be to tune the hyper-parameters of the model such as tree depth, minimum samples split, or number of trees.

```python
Location_ID              RMSE      MAE       R2_test   R2_train
Lat: -94.38, Lon: 41.28  0.183029  0.119504  0.613310  0.934019
Lat: -94.27, Lon: 41.18  0.121630  0.085492  0.673736  0.942292
```

When looking at feature importance, it appears the temperature plays the biggest role in the health of a crop over all locations.
![image](https://github.com/gbarbosa99/Google-Earth-Engine-Crop-Yield-Prediction/assets/99455542/5dfbe6f8-5673-4886-b0e0-6170f94965b8)

### Citations
[1] Ramos P.J., Prieto F.A., Montoya E.C., Oliveros C.E. Automatic fruit count on coffee branches using computer vision. Comput. Electron. Agric. 2017;137:9–22. doi: 10.1016/j.compag.2017.03.010.
[2] Kung H.-Y., Kuo T.-H., Chen C.-H., Tsai P.-Y. Accuracy Analysis Mechanism for Agriculture Data Using the Ensemble Neural Network Method. Sustainability. 2016
[3] Vennam RR, Ramamoorthy P, Poudel S, Reddy KR, Henry WB, Bheemanahalli R. Developing Functional Relationships between Soil Moisture Content and Corn Early-Season Physiology, Growth, and Development. Plants (Basel). 2023 Jun 28;12(13):2471. doi: 10.3390/plants12132471. PMID: 37447032; PMCID: PMC10346487.
[4] Moore CE, Meacham-Hensold K, Lemonnier P, Slattery RA, Benjamin C, Bernacchi CJ, Lawson T, Cavanagh AP. The effect of increasing temperature on crop photosynthesis: from enzymes to ecosystems. J Exp Bot. 2021 Apr 2;72(8):2822-2844. doi: 10.1093/jxb/erab090. PMID: 33619527; PMCID: PMC8023210.
[5] https://health2016.globalchange.gov/low/ClimateHealth2016_07_Food_small.pdf
