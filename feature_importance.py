for location_id, importances in feature_importances_dict.items():
    # Sort features for consistent plotting
    features, values = zip(*sorted(importances.items(), key=lambda x: x[1], reverse=True))

    plt.figure(figsize=(10, 6))
    plt.bar(features, values)
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title(f'Feature Importance in Predicting NDVI for {location_id}')
    plt.xticks(rotation=45)  # Rotate feature names for better readability
    plt.show()
