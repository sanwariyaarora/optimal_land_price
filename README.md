
# House Price Prediction and Growth Model

## Overview

This project uses historical real estate data to predict **future house prices** and calculate a **growth score** for various areas. The model is built using **Python** and employs a **Random Forest Regressor** to predict future prices based on features such as the area, number of rooms, proximity to main roads, and the age of the house. In addition, the model computes a **growth score** which indicates the potential price growth of properties in different areas over a defined future period (e.g., 10 years).

### Key Features:
- **Future Price Prediction**: Forecast house prices based on historical trends.
- **Growth Score Calculation**: A 0-100 score representing expected growth in property values.
- **Customizable Area Relevance**: Placeholder for user-defined relevance scores based on facilities like schools, hospitals, etc.
  
## Requirements

To run the project, you will need to install the following Python libraries:

```bash
pip install pandas
pip install numpy
pip install scikit-learn
```

If you want to visualize data, you can also install:

```bash
pip install matplotlib
pip install seaborn
```

## Installation

1. Clone the repository or download the files to your local machine.
   
2. Ensure you have the necessary dependencies installed by running the above commands.

3. Place the dataset `house.csv` in the same directory as the Python files.

4. The main file is `complete.py`, which runs the prediction and scoring process.

## File Structure

- **house.py**: Contains the data preprocessing and core logic for predicting future prices and calculating the growth score.
- **complete.py**: The main file that integrates the `house.py` functions and provides the interface for user input and output.
- **house.csv**: The dataset containing historical data on house sales (you need to provide this file).

## How to Use

1. Run the `complete.py` script:
   ```bash
   python complete.py
   ```

2. When prompted, enter the name of the area (city) you want predictions for:
   ```
   Enter name of the area: <Area Name>
   ```

3. The model will output:
   - **Predicted Future Price**: The estimated price of houses in the selected area 10 years from now.
   - **Growth Score**: A score (0-100) indicating the expected price growth potential in the area.
   - **Relevance Score**: Placeholder value, which can be customized to reflect other area-related factors.

### Sample Output:
```bash
Predicted Future Price for <Area Name> in 10 years: ₹5,500,000
Future Growth Score for <Area Name>: 82/100
Area Relevance Score for <Area Name>: 75/100
```

## Dataset Structure

The dataset (`house.csv`) should contain the following columns:

- **AREA**: Categorical value representing the city or area.
- **INT_SQFT**: The internal square footage of the property.
- **DIST_MAINROAD**: Distance from the main road.
- **N_BEDROOM**: Number of bedrooms.
- **N_BATHROOM**: Number of bathrooms.
- **N_ROOM**: Total number of rooms.
- **DATE_SALE**: Date of sale (in `dd-mm-yyyy` format).
- **DATE_BUILD**: Date when the house was built (in `dd-mm-yyyy` format).
- **SALES_PRICE**: The price at which the house was sold.
- **SALE_COND**: Sale condition (categorical).
- **PARK_FACIL**: Parking facility availability (categorical).

Ensure your data conforms to these specifications for accurate results.

## Customization

- The model includes a **placeholder for relevance score calculation**. This can be customized based on your specific criteria (e.g., proximity to schools, shopping malls, public transport).
- The prediction period is set to 10 years by default. You can modify this by adjusting the `future_years` variable in the `house.py` file.

## Contributing

Contributions to improve the model or add new features (e.g., additional prediction periods or factors like environmental scores) are welcome. Please create a pull request or submit an issue for discussion.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

Special thanks to the open-source community and the contributors of libraries such as **pandas**, **scikit-learn**, and **numpy** for making projects like this possible.

---

This **README.md** provides a clear overview of the project, its purpose, usage instructions, and customization options. You can adapt it further based on specific details related to your project.
