import pandas as pd
from scipy.stats import pearsonr

df = pd.read_csv("clean_with_returns.csv", index_col=0, parse_dates=True)

lags_to_test = [0, 1, 2, 3, 5, 7, 10]

print(f"{'Lag':>5} | {'Correlation':>12} | {'p-value':>10}")
print("-" * 35)

for lag in lags_to_test:
    # Shift oil return forward by 'lag' days, so today's oil lines up with INR 'lag' days later
    shifted_oil = df["Oil_return"].shift(lag)

    # Combine into a temp dataframe and drop rows with no match (start of series)
    temp = pd.DataFrame({
        "Oil_shifted": shifted_oil,
        "INR_return": df["INR_return"]
    }).dropna()

    corr, p_value = pearsonr(temp["Oil_shifted"], temp["INR_return"])
    print(f"{lag:>5} | {corr:>12.4f} | {p_value:>10.4f}")