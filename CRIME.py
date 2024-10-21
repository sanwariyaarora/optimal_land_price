import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv('crime_data.csv')  # Ensure the path is correct

# Function to calculate crime score for a district
def calculate_crime_score(district_name):
    # Check if the district exists in the dataset
    if district_name not in df['District'].values:
        return {
            "error": f"District '{district_name}' not found in the dataset."
        }

    # Select the data for the specific district
    district_data = df[df['District'] == district_name]

    # Features used for crime score calculation
    features = [
        'Death due to negligence relating to road accidents - I',
        'Hit and Run - I',
        'Other Accidents - I'
    ]

    # Convert relevant columns to numeric
    for feature in features:
        district_data[feature] = pd.to_numeric(district_data[feature], errors='coerce')

    # Clean the data (handle missing values or invalid entries)
    district_data = district_data[features].replace([np.inf, -np.inf], np.nan).fillna(0)

    if district_data.empty or district_data[features].sum().sum() == 0:
        return {
            "error": f"District '{district_name}' does not have sufficient data."
        }

    # Weighting of different crime types
    weights = {
        'Death due to negligence relating to road accidents - I': 0.5,
        'Hit and Run - I': 0.3,
        'Other Accidents - I': 0.2
    }

    # Calculate weighted crime score
    weighted_score = 0
    for feature in features:
        score_value = district_data[feature].values[0]
        weighted_score += score_value * weights[feature]

    # Scaling the score between 0 and 100 based on predefined maximum score
    max_possible_score = 1000  # Adjust this if needed
    scaled_score = (weighted_score / max_possible_score) * 100

    return {
        "district": district_name,
        "crime_score": round(scaled_score, 2)  # Round to 2 decimal places
    }

# Predict function to be called from app.py
def predict(district_name):
    result = calculate_crime_score(district_name)
    return result  # Return the result from calculate_crime_score
