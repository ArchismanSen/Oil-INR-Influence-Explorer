import pandas as pd

# Load both files
inr = pd.read_csv("raw_inr.csv", index_col=0, parse_dates=True)
oil = pd.read_csv("raw_oil.csv", index_col=0, parse_dates=True)

# Keep only the Close price column, rename for clarity
inr_close = inr[["Close"]].rename(columns={"Close": "INR"})
oil_close = oil[["Close"]].rename(columns={"Close": "Oil"})

# Merge on matching dates only
merged = inr_close.join(oil_close, how="inner")

print("Merged shape:", merged.shape)
print(merged.head())
print(merged.tail())

merged.to_csv("clean_merged.csv")