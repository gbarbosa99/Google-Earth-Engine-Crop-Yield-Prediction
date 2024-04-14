locations = list(metrics_dict.keys())
rmse_values = [metrics_dict[loc]['R-squared'] for loc in locations]

plt.figure(figsize=(10, 6))
bars = plt.bar(locations, rmse_values, color='skyblue')
for bar in bars:
  yval = bar.get_height()
  plt.text(bar.get_x() + bar.get_width()/2, yval + 0.005, round(yval, 3), ha='center', va='bottom')

plt.xlabel('Location')
plt.ylabel('R-squared')
plt.title('R-squared for NDVI Predictions Across Locations')
plt.xticks(rotation=45)
plt.show()
