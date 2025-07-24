import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset (replace with your own dataset if needed)
# Example: Advertising dataset
data = pd.read_csv("advertising.csv")  # Columns: TV, Radio, Newspaper, Sales

# Display first few rows
print("Sample Data:\n", data.head())

# Define features and target
X = data[['TV', 'Radio', 'Newspaper']]
y = data['Sales']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print("Mean Squared Error:", mse)
print("R² Score:", r2)

# Optional: Show coefficients
coefficients = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
print("\nFeature Coefficients:\n", coefficients)
