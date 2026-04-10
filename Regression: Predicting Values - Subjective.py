#Task1
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 1. Create Synthetic Dataset
np.random.seed(42)
n_samples = 60

area_sqft = np.random.randint(500, 3500, n_samples)
num_bedrooms = np.random.randint(1, 6, n_samples)
age_years = np.random.randint(0, 30, n_samples)

# Formula for price: 0.05*area + 10*bedrooms - 0.5*age + noise
price_lakhs = (0.05 * area_sqft) + (10 * num_bedrooms) - (0.5 * age_years) + np.random.normal(0, 5, n_samples)

df = pd.DataFrame({
    'area_sqft': area_sqft,
    'num_bedrooms': num_bedrooms,
    'age_years': age_years,
    'price_lakhs': price_lakhs
})

# 2. Build Multiple Linear Regression Model
X = df[['area_sqft', 'num_bedrooms', 'age_years']]
y = df['price_lakhs']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# 3. Print Coefficients and Intercept
print(f"Intercept: {model.intercept_:.2f}")
for feature, coef in zip(X.columns, model.coef_):
    print(f"Coefficient for {feature}: {coef:.4f}")

# 4. Display First 5 Actual vs Predicted
y_pred = model.predict(X_test)
comparison_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}).head(5)
print("\nFirst 5 Actual vs Predicted values:")
print(comparison_df)


#Task2
# Calculate Metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"\nMAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

# Task 2 Comment:
# MAE represents the average absolute error in price (lakhs), making it easy to interpret.
# RMSE penalizes larger errors more heavily than MAE, highlighting outliers.
# R² indicates the proportion of variance explained by the model; a value close to 1 suggests a strong fit.


#Task3
# Compute Residuals
residuals = y_test - y_pred

# Plotting Histogram of Residuals
plt.figure(figsize=(8, 5))
plt.hist(residuals, bins=10, color='skyblue', edgecolor='black', alpha=0.7)
plt.axvline(0, color='red', linestyle='--')
plt.title('Distribution of Residuals')
plt.xlabel('Residual Value (Actual - Predicted)')
plt.ylabel('Frequency')
plt.show()
