# Data-Science
# Titanic Survival Prediction Project

This project uses a linear regression model to predict the likelihood of a Titanic passenger's survival based on various features such as passenger class, age, number of siblings/spouses aboard, and fare paid. The analysis provides insights into survival factors in similar disaster scenarios.

## Project Structure


### Files
- **`data/titanic_original.csv`**: Dataset file with Titanic passenger data (assumes a `survived` column indicating passenger survival).
- **`models/titanic_survival_model.pkl`**: Saved linear regression model for survival prediction.
- **`main.py`**: Main script that:
  - Loads and preprocesses the dataset.
  - Defines the independent variables (features) and dependent variable (target).
  - Trains a linear regression model.
  - Saves the trained model.
  - Predicts survival probability for hypothetical data.
- **`requirements.txt`**: Lists the required Python packages.

## Requirements

To run this project, you need Python 3.x and the following libraries:
- `pandas`
- `scikit-learn`
- `joblib`

Install the required packages by running:
## How to Run the Project

git clone https://github.com/your_username/Titanic_Analysis.git
cd Titanic_Analysis

##Customizing Predictions
# Example hypothetical passenger data
hypothetical_passenger = pd.DataFrame({
    'pclass': [1],    # First class
    'age': [28],      # 28 years old
    'sibsp': [0],     # No siblings/spouses aboard
    'fare': [100]     # Fare of 100
})
