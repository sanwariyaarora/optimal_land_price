import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from datetime import datetime
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('house.csv')

# Convert DATE_SALE and DATE_BUILD to datetime format
df['DATE_SALE'] = pd.to_datetime(df['DATE_SALE'], format='%d-%m-%Y', errors='coerce')
df['DATE_BUILD'] = pd.to_datetime(df['DATE_BUILD'], format='%d-%m-%Y', errors='coerce')

# Calculate YEARS_SINCE_SALE and YEARS_SINCE_BUILD
current_year = datetime.now().year
df['SALE_YEAR'] = df['DATE_SALE'].dt.year
df['BUILD_YEAR'] = df['DATE_BUILD'].dt.year
df['YEARS_SINCE_SALE'] = current_year - df['SALE_YEAR']
df['YEARS_SINCE_BUILD'] = current_year - df['BUILD_YEAR']

# Handle missing values
df = df.fillna({
    'DIST_MAINROAD': df['DIST_MAINROAD'].median(),
    'N_BEDROOM': df['N_BEDROOM'].median(),
    'N_BATHROOM': df['N_BATHROOM'].median(),
    'N_ROOM': df['N_ROOM'].median(),
    'SALE_COND': 'Unknown',
    'PARK_FACIL': 'No'
})

# Encode categorical variables
label_encoders = {}
categorical_columns = ['SALE_COND', 'PARK_FACIL', 'AREA']
for col in categorical_columns:
    label_encoder = LabelEncoder()
    df[col] = label_encoder.fit_transform(df[col])
    label_encoders[col] = label_encoder

# Define features and target
features = ['AREA', 'INT_SQFT', 'DIST_MAINROAD', 'N_BEDROOM', 'N_BATHROOM', 'N_ROOM', 'PARK_FACIL', 'YEARS_SINCE_SALE',
            'YEARS_SINCE_BUILD']
X = df[features]
y = df['SALES_PRICE']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Your existing imports and data preparation...

# Assuming 'SALES_PRICE' is the total price and 'INT_SQFT' is the square footage

def predict_future_price_and_growth(city):
    city = city.strip()  # Remove leading/trailing whitespace
    if city in label_encoders['AREA'].classes_:
        encoded_city = label_encoders['AREA'].transform([city])[0]
        area_data = df[df['AREA'] == encoded_city]

        # Assuming a 10-year future period
        future_years = 10

        # Calculate historical growth rates for numerical features
        historical_growth_rates = {}
        for feature in ['INT_SQFT', 'DIST_MAINROAD', 'N_BEDROOM', 'N_BATHROOM', 'N_ROOM']:
            if len(area_data) > 1:  # Ensure we have enough data points
                growth_rate = (area_data[feature].iloc[-1] - area_data[feature].iloc[0]) / (
                            area_data['YEARS_SINCE_SALE'].iloc[-1] - area_data['YEARS_SINCE_SALE'].iloc[0])
            else:
                growth_rate = 0  # Default to no growth if insufficient data
            historical_growth_rates[feature] = growth_rate

        # Create future data based on historical growth rates
        future_data = area_data.copy()
        for feature, rate in historical_growth_rates.items():
            future_data[feature] += rate * future_years

        future_data['YEARS_SINCE_SALE'] += future_years
        future_data['YEARS_SINCE_BUILD'] += future_years

        # Predict future prices
        future_prices = model.predict(future_data[features])
        future_price_mean = np.mean(future_prices)

        # Calculate historical price mean
        historical_price_mean = np.mean(area_data['SALES_PRICE'])

        # Calculate price growth rate
        price_growth_rate = (future_price_mean - historical_price_mean) / historical_price_mean

        # Get the square footage (INT_SQFT) for per-square-foot calculation
        sqft = np.mean(area_data['INT_SQFT'])

        # Ensure square footage is not zero before dividing
        if sqft > 0:
            original_price_per_sqft = historical_price_mean / sqft
            future_price_per_sqft = future_price_mean / sqft
        else:
            original_price_per_sqft = 0
            future_price_per_sqft = 0

        # Normalize the growth score based on price increase
        growth_score = price_growth_rate * 100
        growth_score = round(growth_score)

        return {'growth_score': growth_score, 'original_price_per_sqft': original_price_per_sqft, 'future_price_per_sqft': future_price_per_sqft}
    else:
        raise ValueError(
            f"City '{city}' not found in the dataset. Available cities are: {', '.join(label_encoders['AREA'].classes_)}")


def plot_price_trend(city, original_price_per_sqft, future_price_per_sqft):
    city = city.strip()  # Remove leading/trailing whitespace
    if city in label_encoders['AREA'].classes_:
        encoded_city = label_encoders['AREA'].transform([city])[0]
        area_data = df[df['AREA'] == encoded_city]

        # Extract years and sales price for plotting the historical trend
        years = area_data['SALE_YEAR']
        sales_prices = area_data['SALES_PRICE']

        # Plot historical prices
        plt.figure(figsize=(10, 6))
        plt.plot(years, sales_prices / np.mean(area_data['INT_SQFT']), label='Historical Price per Sq Ft', color='blue', marker='o')

        # Add predicted future price
        future_year = years.max() + 10  # Assuming we predicted for 10 years into the future
        plt.scatter([future_year], [future_price_per_sqft], color='red', label=f'Predicted Future Price per Sq Ft ({future_year})')

        # Labeling
        plt.title(f'Price Trend for {city}')
        plt.xlabel('Year')
        plt.ylabel('Price per Sq Ft')
        plt.legend()
        
        # Save the plot to a file
        graph_path = f'static/{city}_price_trend.png'
        plt.savefig(graph_path)
        plt.close()  # Close the plot to free up memory
        
        return graph_path
    else:
        raise ValueError(
            f"City '{city}' not found in the dataset. Available cities are: {', '.join(label_encoders['AREA'].classes_)}")
