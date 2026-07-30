import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

df = pd.read_csv("clean_with_returns.csv", index_col=0, parse_dates=True)

# X = input (what we predict FROM), y = target (what we predict)
X = df[["Oil_return"]]   # double brackets = keep it as a table, not a plain list
y = df["INR_return"]

# Split: 80% train, 20% test. random_state fixes the split so it's reproducible.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the unseen test data
y_pred = model.predict(X_test)

# Evaluate honestly
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"Slope (coefficient): {model.coef_[0]:.4f}")
print(f"Intercept: {model.intercept_:.6f}")
print(f"R² score: {r2:.4f}")
print(f"Mean Absolute Error: {mae:.6f}")