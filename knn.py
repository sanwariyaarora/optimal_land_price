import requests
import numpy as np
from sklearn.neighbors import NearestNeighbors

# Use your actual Google Maps API key
API_KEY = 'AIzaSyCFv8iVw9cqEsRxaPMmOAzfNudkikCX9gY'

# Define facilities to search for with weights
facility_types = {
    'hospital': 0.2,
    'school': 0.1,
    'park': 0.05,  # The stray '+' after 0.05 is removed
    'shopping_mall': 0.1,
    'restaurant': 0.05,
    'train_station': 0.1,
    'bus_station': 0.1,
    'subway_station': 0.1
}


# Additional facility types with higher search radius
connectivity_types = {
    'airport': 0.2,
    'train_station': 0.1
}

# Function to fetch facilities from Google Maps API
def get_nearby_facilities(lat, lng, facility_type, radius=500):
    url = f"https://maps.googleapis.com/maps/api/place/nearbysearch/json?location={lat},{lng}&radius={radius}&type={facility_type}&key={API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json().get('results', [])
    else:
        raise Exception(f"API request failed with status code {response.status_code}: {response.text}")

# Fetch all facilities in a given area and calculate weighted score based on proximity and count
def fetch_and_score_facilities(lat, lng, radius=500):
    total_weight = sum(facility_types.values()) + sum(connectivity_types.values())
    total_score = 0

    # Fetch local facilities (within 500 meters)
    for facility, weight in facility_types.items():
        facilities = get_nearby_facilities(lat, lng, facility_type=facility, radius=radius)
        facility_count = len(facilities)
        if facility_count > 0:
            facility_coords = np.array(
                [[f['geometry']['location']['lat'], f['geometry']['location']['lng']] for f in facilities])
            user_location = np.array([[lat, lng]])
            knn = NearestNeighbors(n_neighbors=1)
            knn.fit(facility_coords)
            distances, _ = knn.kneighbors(user_location)
            closest_distance = np.min(distances)
            scaled_distance = (closest_distance / radius)
            distance_score = max(0, 100 - scaled_distance * 100)
            normalized_count = min(facility_count / 5, 1)
            score = (distance_score * 0.8 + normalized_count * 0.2) * weight
            total_score += score

    # Fetch connectivity facilities (airport, train stations) with larger radius
    for facility, weight in connectivity_types.items():
        larger_radius = 20000 if facility == 'airport' else 10000
        facilities = get_nearby_facilities(lat, lng, facility_type=facility, radius=larger_radius)
        if facilities:
            facility_coords = np.array(
                [[f['geometry']['location']['lat'], f['geometry']['location']['lng']] for f in facilities])
            user_location = np.array([[lat, lng]])
            knn = NearestNeighbors(n_neighbors=1)
            knn.fit(facility_coords)
            distances, _ = knn.kneighbors(user_location)
            closest_distance = np.min(distances)
            scaled_distance = (closest_distance / larger_radius)
            distance_score = max(0, 100 - scaled_distance * 100)
            score = distance_score * weight
            total_score += score

    final_score = int(total_score / total_weight if total_weight != 0 else 0)
    return max(0, min(100, final_score))

# Flask compatibility: Define the predict function to return proximity score
def predict(lat, lng):
    try:
        proximity_score = fetch_and_score_facilities(lat, lng, radius=500)
        return {
            "proximity_score": proximity_score
        }
    except Exception as e:
        return {
            "error": f"Failed to calculate proximity score: {e}"
        }
