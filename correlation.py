import pandas as pd
from scipy.stats import pearsonr, spearmanr

df = pd.read_csv("clean_with_returns.csv", index_col=0, parse_dates=True)

# Pearson correlation - measures linear relationship
pearson_corr, pearson_p = pearsonr(df["INR_return"], df["Oil_return"])

# Spearman correlation - measures monotonic relationship (doesn't assume straight-line)
spearman_corr, spearman_p = spearmanr(df["INR_return"], df["Oil_return"])

print(f"Pearson correlation:  {pearson_corr:.4f}  (p-value: {pearson_p:.4f})")
print(f"Spearman correlation: {spearman_corr:.4f}  (p-value: {spearman_p:.4f})")