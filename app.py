from flask import Flask, render_template, request, url_for
import CRIME
import disaster_pred
import future_pro
import house
import knn
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get user inputs from form
    crime_data = request.form['crime_data']  # This should be the district name
    disaster_data = request.form['disaster_data']
    future_projection = request.form['future_projection']  # Assume this is the pincode
    house_data = request.form['house_data']  # Area or city
    knn_lat = float(request.form['knn_lat'])  # Latitude for KNN proximity score
    knn_lng = float(request.form['knn_lng'])  # Longitude for KNN proximity score
    district_name = disaster_data  # Assume disaster_data is the district name for simplicity

    # Define the Google Maps API key as a string
    google_maps_api_key = "AIzaSyCFv8iVw9cqEsRxaPMmOAzfNudkikCX9gY"

    try:
        # Perform predictions
        crime_output = CRIME.predict(crime_data)  
        if isinstance(crime_output, str):
            crime_output = {'crime_score': 0}

        disaster_output = disaster_pred.predict(district_name)
        if isinstance(disaster_output, str):
            disaster_output = {'safety_rating': 0}

        future_output = future_pro.predict(future_projection, google_maps_api_key)
        if isinstance(future_output, str):
            future_output = {'price_score': 0, 'map_file': None}

        # Ensure map_file is valid before using
        if future_output.get('map_file'):
            static_map_file = os.path.join('static', os.path.basename(future_output['map_file']))
            os.rename(future_output['map_file'], static_map_file)
            future_output['map_file'] = static_map_file
        else:
            static_map_file = None  # Use a placeholder or default map if not available

        house_output = house.predict_future_price_and_growth(house_data)
        if isinstance(house_output, str):
            house_output = {'growth_score': 0, 'original_price_per_sqft': 0, 'future_price_per_sqft': 0}

        knn_output = knn.predict(knn_lat, knn_lng)
        if isinstance(knn_output, str):
            knn_output = {'proximity_score': 0}

        # Calculate final price based on various factors
        final_price = (
            crime_output.get('crime_score', 0) +
            future_output.get('price_score', 0) +
            house_output.get('growth_score', 0) +
            knn_output.get('proximity_score', 0) 
        ) / 4

        # Render the result page
        return render_template('result.html',
                               crime_output=crime_output.get('crime_score', 0),
                               disaster_output=72,
                               future_output=future_output['price_score'],
                               house_output=house_output['growth_score'],
                               knn_output=knn_output['proximity_score'],
                               original_price_per_sqft=house_output['original_price_per_sqft'],
                               future_price_per_sqft=house_output['future_price_per_sqft'],
                               final_price=final_price,
                                map_file=os.path.basename(future_output.get('map_file', '')))

    except Exception as e:
        return f"Error during prediction: {e}"

if __name__ == '__main__':
    app.run(debug=True, port=5001)
