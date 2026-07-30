import pandas as pd

merged = pd.read_csv("clean_merged.csv", index_col=0, parse_dates=True)

# Daily % change for each series
merged["INR_return"] = merged["INR"].pct_change()
merged["Oil_return"] = merged["Oil"].pct_change()

# Drop the first row (it has no previous day to compare to, so it's NaN)
merged = merged.dropna()

print(merged.head())
print("\nShape after adding returns:", merged.shape)

merged.to_csv("clean_with_returns.csv")