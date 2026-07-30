import yfinance as yf

# Download rupee/dollar exchange rate
inr = yf.download("USDINR=X", start="2020-01-01", end="2026-07-23")
inr.columns = inr.columns.get_level_values(0)   # flatten the header
inr.to_csv("raw_inr.csv")
print("INR data saved:", inr.shape)

# Download Brent crude oil prices
oil = yf.download("BZ=F", start="2020-01-01", end="2026-07-23")
oil.columns = oil.columns.get_level_values(0)   # flatten the header
oil.to_csv("raw_oil.csv")
print("Oil data saved:", oil.shape)