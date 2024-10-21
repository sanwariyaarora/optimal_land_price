import requests
import math
import folium
from folium.plugins import MarkerCluster
import os
from sklearn.preprocessing import MinMaxScaler


def geocode_pincode(pincode, google_maps_api_key):
    url = f"https://maps.googleapis.com/maps/api/geocode/json?address={pincode}&key={google_maps_api_key}"
    response = requests.get(url).json()

    if response['status'] == 'OK':
        location = response['results'][0]['geometry']['location']
        latitude = location['lat']
        longitude = location['lng']
        return [latitude, longitude]
    else:
        return None


def calculate_distance(coords1, coords2):
    R = 6371  # Radius of the Earth in km
    lat1, lon1 = coords1
    lat2, lon2 = coords2
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c  # in km
    return distance


def price_change_score(area_coords, developments):
    score = 0
    nearby_developments = []

    for dev in developments:
        dist = calculate_distance(area_coords, dev["coords"])
        if dist < 25:  # Increase weight for areas within 25 km
            score += dev["impact"] * (25 - dist) / 25  # Normalize score based on proximity (closer = higher)
            nearby_developments.append(dev)  # Store the nearby developments

    score = min(max(score, 0), 100)  # Ensure score is between 0 and 100
    return score, nearby_developments


def create_map(center_coords, developments_within_radius):
    map_obj = folium.Map(location=center_coords, zoom_start=12)

    # Add marker for the center location
    folium.Marker(
        location=center_coords,
        popup="PIN Code Location",
        icon=folium.Icon(color="blue", icon="home")
    ).add_to(map_obj)

    # Create a MarkerCluster for development projects
    marker_cluster = MarkerCluster().add_to(map_obj)

    for dev in developments_within_radius:
        folium.Marker(
            location=dev["coords"],
            popup=f"{dev['name']} (Impact: {dev['impact']})",
            icon=folium.Icon(color="red", icon="info-sign")
        ).add_to(marker_cluster)

    map_file = "static/development_map.html"
    map_obj.save(map_file)

    return os.path.abspath(map_file)


developments = [
    {"name": "Chennai Metro - Madhavaram Station", "coords": [13.1501, 80.2240], "impact": 8},
    {"name": "Chennai Sea Bridge", "coords": [13.0922, 80.2921], "impact": 7},
    {"name": "Cuddalore Greenfield Port", "coords": [11.7447, 79.7685], "impact": 9},
    {"name": "Hosur Airport", "coords": [12.7402, 77.8292], "impact": 9},
    {"name": "Bangalore-Hosur Interstate Metro", "coords": [12.9716, 77.5946], "impact": 8},
    {"name": "SIPCOT Integrated Textile Park, Salem", "coords": [11.6643, 78.1460], "impact": 6},
    {"name": "Rameswaram to Talaimannar Ferry Service", "coords": [9.2886, 79.3129], "impact": 6},
    {"name": "Space Industrial & Propellant Park", "coords": [13.0827, 80.2707], "impact": 10},
    {"name": "TIDEL Park - Madurai", "coords": [9.9252, 78.1198], "impact": 7},
    {"name": "Neo TIDEL Park, Vellore", "coords": [12.9165, 79.1325], "impact": 7},
    {"name": "Chennai Peripheral Ring Road", "coords": [13.0827, 80.2707], "impact": 8},
    {"name": "Kalaignar International Convention Centre", "coords": [12.8072, 80.2416], "impact": 8},
    {"name": "Multi-modal Bus Terminus, Broadway", "coords": [13.0909, 80.2843], "impact": 6},
    {"name": "Outer Ring Road Expansion, Chennai", "coords": [12.9857, 80.2206], "impact": 7},
    {"name": "Hi-tech Film City, Poonamallee", "coords": [13.0391, 80.1067], "impact": 6},
    {"name": "Shenoy Nagar Park Restoration", "coords": [13.0750, 80.2270], "impact": 6},
    {"name": "Inclusive Park for Disabled Children", "coords": [13.0851, 80.2100], "impact": 7},
    {"name": "100 New Parks Across the City", "coords": [13.0830, 80.2707], "impact": 8},
    {"name": "Hitech The Crest Phase 2 Residential Plots", "coords": [13.0496, 80.1271], "impact": 6},
    {"name": "GP Homes Ethulia Project", "coords": [13.0826, 80.1462], "impact": 6},
    {"name": "Smart Classrooms for Chennai Schools", "coords": [13.0292, 80.2472], "impact": 7},
    {"name": "Pedestrian-Friendly Pondy Bazaar", "coords": [13.0458, 80.2470], "impact": 8},
    {"name": "Cycling Tracks along Buckingham Canal", "coords": [13.0562, 80.2486], "impact": 7},
    {"name": "Command and Control Center (CCC)", "coords": [13.0827, 80.2707], "impact": 9},
    {"name": "Chettinad Hospital Expansion", "coords": [12.8240, 80.0360], "impact": 8}
]


def predict(pincode, google_maps_api_key):
    coords = geocode_pincode(pincode, google_maps_api_key)
    if not coords:
        return {"error": "Location not found. Please enter a valid PIN code.", "price_score": 0, "map_file": None}

    price_score, nearby_developments = price_change_score(coords, developments)
    map_file = create_map(coords, nearby_developments)

    return {
        "price_score": price_score,
        "map_file": map_file
    }