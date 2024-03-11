import json
import folium
import pandas as pd

def extract_lat_lon(geojson_str):
    geojson = json.loads(geojson_str)
    if geojson['type'] == 'Point':
        # Extract latitude and longitude for Point geometry
        lon, lat = geojson['coordinates']
        return lat, lon
    return None, None

# Apply the function to each row in the DataFrame
aggregated_df[['latitude', 'longitude']] = aggregated_df.apply(
    lambda row: pd.Series(extract_lat_lon(row['.geo'])), axis=1)

# Creating a map centered around the average latitude and longitude
map_center = [aggregated_df['latitude'].mean(), aggregated_df['longitude'].mean()]
m = folium.Map(location=map_center, zoom_start=5)

# Adding points for each spot in aggregated_df
for _, row in aggregated_df.iterrows():
    # Define popup content
    popup_content = f"""
    <table style='width:200px'>
        <tr>
            <th style='text-align:left;'><strong>Location</strong></th>
            <td>{row['latitude']:.4f}, {row['longitude']:.4f}</td>
        </tr>
        <tr>
            <th style='text-align:left;'><strong>Actual NDVI</strong></th>
            <td>{row['NDVI']:.2f}</td>
        </tr>
        <tr>
            <th style='text-align:left;'><strong>Predicted NDVI</strong></th>
            <td>{row['Predicted_NDVI']:.2f}</td>
        </tr>
    </table>
    """
    popup = folium.Popup(popup_content, max_width=250)

    # Check if predicted NDVI is within one standard deviation of the actual NDVI (Might remove this and use a different metric)
    ndvi_diff = abs(row['NDVI'] - row['Predicted_NDVI'])
    color = 'blue' if ndvi_diff <= ndvi_std else 'red'

    # Create CircleMarker with dynamic color
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=5,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.7,
        popup=popup,
    ).add_to(m)

# Display the map
m
