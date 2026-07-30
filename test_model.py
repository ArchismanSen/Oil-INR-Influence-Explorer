import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv("clean_with_returns.csv", index_col=0, parse_dates=True)

X = df[["Oil_return"]]
y = df["INR_return"]

# Retrain on ALL the data this time (not just 80%) since we're done evaluating
# and now just want the model's real-world best guess
model = LinearRegression()
model.fit(X, y)

# Try your own inputs here — oil return as a decimal (2% = 0.02, -5% = -0.05)
test_values = [0.02, -0.02, 0.05, -0.05, 0.10]

print("If oil moves by this much in a day, model predicts INR moves by:\n")
for oil_move in test_values:
    predicted_inr = model.predict([[oil_move]])[0]
    print(f"Oil: {oil_move*100:+.1f}%  ->  Predicted INR: {predicted_inr*100:+.4f}%")