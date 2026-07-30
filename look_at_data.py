import pandas as pd

inr = pd.read_csv("raw_inr.csv")
oil = pd.read_csv("raw_oil.csv")

print("---- INR DATA ----")
print(inr.head())      # first 5 rows
print(inr.tail())      # last 5 rows

print("\n---- OIL DATA ----")
print(oil.head())
print(oil.tail())