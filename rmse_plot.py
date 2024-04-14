locations = list(metrics_dict.keys())
rmse_values = [metrics_dict[loc]['RMSE'] for loc in locations]

plt.figure(figsize=(10, 6))
plt.bar(locations, rmse_values, color='skyblue')
plt.xlabel('Location')
plt.ylabel('RMSE')
plt.title('RMSE for NDVI Predictions Across Locations')
plt.xticks(rotation=45)
plt.show()
