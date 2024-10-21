import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import warnings

warnings.filterwarnings("ignore", category=ConvergenceWarning)


# Load and prepare disaster data
def load_and_prepare_data(filename):
    df = pd.read_csv(filename)
    df = df.set_index('Year')
    df = df.apply(pd.to_numeric, errors='coerce')
    return df


# Load datasets
all_disasters = load_and_prepare_data('all_disasters.csv')
earthquakes = load_and_prepare_data('Earthquakes.csv')
floods = load_and_prepare_data('Floods.csv')
cyclones = load_and_prepare_data('Cyclones.csv')

# Prepare the data
districts = all_disasters.columns
total_disasters = all_disasters.iloc[0].values

# Count occurrences of disasters
def count_disasters(df):
    return df.notna().sum().values


earthquake_count = count_disasters(earthquakes)
flood_count = count_disasters(floods)
cyclone_count = count_disasters(cyclones)

# Combine all disaster data
disaster_data = pd.DataFrame({
    'District': districts,
    'Total': total_disasters,
    'Earthquakes': earthquake_count,
    'Floods': flood_count,
    'Cyclones': cyclone_count
})


# Function to make predictions
def predict_disasters(data, steps=10):
    if len(data) <= 3:  # Not enough data points for ARIMA
        return np.full(steps, np.mean(data))

    try:
        model = ARIMA(data, order=(1, 1, 1))
        results = model.fit()
        forecast = results.forecast(steps=steps)
        return np.maximum(forecast.round().astype(int), 0)  # Ensure non-negative integers
    except:
        return np.full(steps, np.mean(data))


# Make predictions for each disaster type
predictions = {}
for disaster in ['Total', 'Earthquakes', 'Floods', 'Cyclones']:
    predictions[disaster] = predict_disasters(disaster_data[disaster])


# Calculate safety rating based on disaster data
def calculate_safety_rating(district):
    if district not in districts:
        raise ValueError(f"District '{district}' not found in disaster data.")

    district_data = disaster_data[disaster_data['District'] == district].iloc[0]
    total_disasters = district_data['Total']
    max_disasters = disaster_data['Total'].max()

    # Calculate base safety score (inverted, so fewer disasters = higher score)
    base_score = 100 - (total_disasters / max_disasters * 100)

    # Adjust for specific disaster types with penalties
    flood_penalty = 10 if district_data['Floods'] > 0 else 0
    cyclone_penalty = 9 if district_data['Cyclones'] > 0 else 0
    earthquake_penalty = 7 if district_data['Earthquakes'] > 0 else 0

    # Calculate final score
    safety_score = max(1, min(100, base_score - flood_penalty - cyclone_penalty - earthquake_penalty))

    return round(safety_score, 2)


# Flask compatibility: predict function to be used in the Flask app
def predict(district):
    if district not in districts:
        return {
            "error": f"District '{district}' not found in the disaster data."
        }

    # Calculate safety rating
    safety_rating = calculate_safety_rating(district)

    # Fetch disaster prediction for the next 10 years
    district_index = districts.get_loc(district)
    disaster_predictions = {disaster: predictions[disaster][district_index] for disaster in predictions}

    # Return both the safety rating and the predicted disaster numbers
    return {
        "safety_rating": safety_rating,
        "predictions": disaster_predictions
    }
