
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("AB_NYC_2019.csv")

print("Initial Shape:", df.shape)
print(df.info())

"""misssing data"""

missing_counts = df.isnull().sum()
print("\nMissing Values:\n", missing_counts)

# Fill missing 'reviews_per_month' with 0
if 'reviews_per_month' in df.columns:
    df['reviews_per_month'] = df['reviews_per_month'].fillna(0)

# Drop rows with missing essential info
df = df.dropna(subset=['name', 'host_id', 'latitude', 'longitude'])

"""dup removing"""

df.drop_duplicates(inplace=True)
print("After duplicate removal:", df.shape)

"""standardization"""

if 'name' in df.columns:
    df['name'] = df['name'].str.strip().str.title()

if 'neighbourhood_group' in df.columns:
    df['neighbourhood_group'] = df['neighbourhood_group'].str.strip().str.title()

"""Outlier Detection & Treatment"""

if 'price' in df.columns:
    q_low = df['price'].quantile(0.01)
    q_high = df['price'].quantile(0.99)
    df = df[(df['price'] >= q_low) & (df['price'] <= q_high)]

"""saving"""

df.to_csv("AB_NYC_2019_cleaned.csv", index=False)

"""visualization"""

if 'neighbourhood_group' in df.columns and 'price' in df.columns:
    plt.figure(figsize=(8,6))
    sns.boxplot(x='neighbourhood_group', y='price', data=df)
    plt.title("Price Distribution by Neighbourhood Group (Cleaned Data)")
    plt.show()
