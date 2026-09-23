import pandas as pd

df = pd.read_csv("data/data_ecommerce_customer_churn.csv")

print(df.head())
print(df.shape)
print(df.columns)
print(df.isnull().sum())
print("Duplicate rows:", df.duplicated().sum())
print(df.dtypes)