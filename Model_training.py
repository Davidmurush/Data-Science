import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib

# Step 1: Load and preprocess the data
data_path = "/content/titanic_original.csv"  # Path to the uploaded file
data = pd.read_csv(data_path)

# Step 2: Data preprocessing
data = data.dropna(subset=['survived'])  # Drop rows where target is missing
data = data.fillna(0)  # Fill other missing values with 0

# Select features and target variable
X = data[['pclass', 'age', 'sibsp', 'fare']]
y = data['survived']  # Dependent variable (survival prediction)

# Step 3: Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=45)

# Step 4: Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error on Test Set:", mse)

# Step 5: Save the model for future use
model_path = "titanic_survival_model.pkl"
joblib.dump(model, model_path)

# Step 6: Make a prediction with hypothetical data
hypothetical_passenger = pd.DataFrame({
    'pclass': [1],    # First class
    'age': [28],      # 28 years old
    'sibsp': [0],     # No siblings/spouses aboard
    'fare': [100]     # Fare of 100
})

# Predict survival
survival_prediction = model.predict(hypothetical_passenger)
print(f"Predicted survival probability for the hypothetical passenger: {survival_prediction[0]}")
