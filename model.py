from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from math import sqrt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from matplotlib import pyplot as plt
from sklearn.model_selection import cross_val_score
import numpy as np

# Example using location_id as the identifier
location_ids = aggregated_df['.geo'].unique()  # Get unique location IDs

# To store RMSE values for locations with enough data for model evaluation
rmse_values = {}
locations_with_sufficient_data = []

for location_id in location_ids:
    df_location = aggregated_df[aggregated_df['.geo'] == location_id]

    X = df_location[['temperature', 'precipitation']]
    y = df_location['NDVI']

    if len(df_location) > 3:  # Minimal sample check
        # Proceed with training/testing
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Generate predictions for the entire dataset to visualize
        predicted_ndvi = model.predict(X)

        # Calculate RMSE for evaluation
        predictions_test = model.predict(X_test)

        # Assign predictions to the DataFrame
        aggregated_df.loc[aggregated_df['.geo'] == location_id, 'Predicted_NDVI'] = predicted_ndvi

        locations_with_sufficient_data.append(location_id)
    else:
        print(f"Location {location_id} has insufficient data for training/testing split. Skipping.")
      
# Step 3: Visualization
for location_id in locations_with_sufficient_data:
    df_plot = aggregated_df[aggregated_df['.geo'] == location_id].sort_values('Date')

    plt.figure(figsize=(10, 6))
    plt.plot(df_plot['Date'], df_plot['NDVI'], label='Actual NDVI', color='blue', marker='o')
    plt.plot(df_plot['Date'], df_plot['Predicted_NDVI'], label='Predicted NDVI', color='red', linestyle='--', marker='x')
    plt.title(f'Actual vs Predicted NDVI for Location: {location_id}')
    plt.xlabel('Date')
    plt.ylabel('NDVI Value')
    plt.xticks(rotation=45)
    plt.legend()
    plt.show()

ndvi_std = aggregated_df['NDVI'].std()

metrics_dict = {}  # Initialize an empty dictionary to store metrics

for location_id in locations_with_sufficient_data:
    # Compute metrics for each location
    rmse = sqrt(mean_squared_error(y_test, predictions_test))
    mae = mean_absolute_error(y_test, predictions_test)
    r2 = r2_score(y_test, predictions_test)

    # Store the metrics in the dictionary
    metrics_dict[location_id] = {'RMSE': rmse, 'MAE': mae, 'R-squared': r2}

# Convert the dictionary to a DataFrame for easier analysis
metrics_df = pd.DataFrame.from_dict(metrics_dict, orient='index')
metrics_df.reset_index(inplace=True)
metrics_df.rename(columns={'index': 'Location_ID'}, inplace=True)

print(metrics_df)
