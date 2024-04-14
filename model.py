from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from math import sqrt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from matplotlib import pyplot as plt
from sklearn.model_selection import cross_val_score
import json
from sklearn.impute import KNNImputer

def format_location_id(location_str):
    # Convert the string to a dictionary
    location_dict = json.loads(location_str.replace("'", "\""))

    # Extract latitude and longitude
    lat, lon = location_dict['coordinates']

    # Return a formatted string (or any other format you prefer)
    return f"Lat: {lat:.2f}, Lon: {lon:.2f}"

aggregated_df['readable_location_id'] = aggregated_df['.geo'].apply(format_location_id)

location_ids = aggregated_df['readable_location_id'].unique()  # Get unique readable location IDs

# To store RMSE values for locations with enough data for model evaluation
metrics_dict = {}  # Initialize an empty dictionary to store metrics
locations_with_sufficient_data =[]

# To store feature importance for locations with enough data for model evaluation
feature_importances_dict = {}

for location_id in location_ids:
    df_location = aggregated_df[aggregated_df['readable_location_id'] == location_id]

    X = df_location[['temperature', 'precipitation', 'soil moisture']]
    y = df_location['NDVI']

    if len(df_location) > 3:  # Minimal sample check
        # Proceed with training/testing
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Initialize an imputer to remove NaN values based on the K-nearest neighbors
        imputer = KNNImputer(n_neighbors=5)
        X_train_imputed = imputer.fit_transform(X_train)
        X_test_imputed = imputer.transform(X_test)

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train_imputed, y_train)

        # Generate predictions for the entire dataset to visualize
        X_imputed = imputer.fit_transform(X)
        predicted_ndvi = model.predict(X_imputed) # Only using X_imputed for modeling purposes to avoid data leakage from test set into training set

        # Calculate RMSE for evaluation
        predictions_test = model.predict(X_test_imputed)

        # Assign predictions to the DataFrame
        aggregated_df.loc[aggregated_df['readable_location_id'] == location_id, 'Predicted_NDVI'] = predicted_ndvi

        locations_with_sufficient_data.append(location_id)

        rmse = sqrt(mean_squared_error(y_test, predictions_test))
        mae = mean_absolute_error(y_test, predictions_test)
        r2 = r2_score(y_test, predictions_test)

        # Store the metrics in the dictionary
        metrics_dict[location_id] = {'RMSE': rmse, 'MAE': mae, 'R-squared': r2}

        # Extract feature importance
        feature_importance = model.feature_importances_

        # Store feature importance metrics in the dictionary
        features = X.columns
        feature_importance = model.feature_importances_
        feature_importance_dict = dict(zip(features, feature_importance))
        feature_importances_dict[location_id] = feature_importance_dict

    else:
        print(f"Location {location_id} has insufficient data for training/testing split. Skipping.")



# Step 3: Visualization
for location_id in locations_with_sufficient_data:
    df_plot = aggregated_df[aggregated_df['readable_location_id'] == location_id].sort_values('Date')

    plt.figure(figsize=(10, 6))
    plt.plot(df_plot['Date'], df_plot['NDVI'], label='Actual NDVI', color='blue', marker='o')
    plt.plot(df_plot['Date'], df_plot['Predicted_NDVI'], label='Predicted NDVI', color='red', linestyle='--', marker='x')
    plt.title(f'Actual vs Predicted NDVI for Location: {location_id}')
    plt.xlabel('Date')
    plt.ylabel('NDVI Value')
    plt.xticks(rotation=45)
    plt.legend()
    plt.show()

import numpy as np

# Convert the dictionary to a DataFrame for easier analysis
metrics_df = pd.DataFrame.from_dict(metrics_dict, orient='index')
metrics_df.reset_index(inplace=True)
metrics_df.rename(columns={'index': 'Location_ID'}, inplace=True)

print(metrics_df)
