"""
STAGE 1: DATA LOADING & PREPARATION
Simple script to load, clean, and prepare data for segmentation
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("STAGE 1: DATA LOADING & PREPARATION")
print("=" * 70)

# Load the data
df = pd.read_csv('marketing_campaign.csv', sep='\t', quotechar=None, quoting=3)

# Clean column names (remove quotes)
df.columns = df.columns.str.replace('"', '')

print(f"\nDataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# Select features for segmentation
# These are the meaningful customer attributes
features_to_use = [
    'Year_Birth',           # Customer age (derived)
    'Income',               # Annual income
    'Kidhome',              # Kids at home
    'Teenhome',             # Teens at home
    'Recency',              # Days since last purchase
    'MntWines',             # Spending on wines
    'MntFruits',            # Spending on fruits
    'MntMeatProducts',      # Spending on meat
    'MntFishProducts',      # Spending on fish
    'MntSweetProducts',     # Spending on sweets
    'MntGoldProds',         # Spending on gold products
    'NumDealsPurchases',    # Purchases with discount
    'NumWebPurchases',      # Purchases via web
    'NumCatalogPurchases',  # Purchases via catalog
    'NumStorePurchases',    # Purchases in store
    'NumWebVisitsMonth',    # Website visits per month
    'AcceptedCmp1',         # Campaign 1 response
    'AcceptedCmp2',         # Campaign 2 response
    'AcceptedCmp3',         # Campaign 3 response
    'AcceptedCmp4',         # Campaign 4 response
    'AcceptedCmp5',         # Campaign 5 response
    'Complain',             # Complaints made
    'Z_CostContact',        # Cost of contact
    'Z_Revenue',            # Revenue per customer
    'Response',             # Response to recent campaign
]

# Create a copy of selected features
X = df[features_to_use].copy()

print(f"\nFeatures selected: {len(features_to_use)}")
print(f"Data shape for segmentation: {X.shape}")

# Handle missing values BEFORE converting to numeric
print(f"\nHandling missing values...")
for col in X.columns:
    if X[col].isnull().sum() > 0 or (X[col] == '').sum() > 0:
        X[col] = pd.to_numeric(X[col], errors='coerce')
        X[col].fillna(X[col].median(), inplace=True)

print(f"Missing values per column:")
missing = X.isnull().sum()
if missing.sum() > 0:
    print(missing[missing > 0])
    # Fill any remaining with median
    for col in X.columns:
        if X[col].isnull().sum() > 0:
            median_val = X[col].median()
            if pd.isna(median_val):
                median_val = 0
            X[col].fillna(median_val, inplace=True)
else:
    print("No missing values!")

# Convert all to numeric (safe conversion)
for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors='coerce')

# Fill any conversion errors with median
X = X.fillna(X.median())

print(f"\nData types:")
print(X.dtypes)

# Calculate derived feature: Age
current_year = 2014  # Data is from 2012-2014
X['Age'] = current_year - X['Year_Birth']

# Calculate total spending
X['TotalSpending'] = (X['MntWines'] + X['MntFruits'] + X['MntMeatProducts'] +
                      X['MntFishProducts'] + X['MntSweetProducts'] + X['MntGoldProds'])

# Calculate total purchases
X['TotalPurchases'] = (X['NumWebPurchases'] + X['NumCatalogPurchases'] + X['NumStorePurchases'])

# Calculate total campaigns accepted (fill NaN as 0 first)
X['Response'] = X['Response'].fillna(0)
X['CampaignsAccepted'] = (X['AcceptedCmp1'] + X['AcceptedCmp2'] + X['AcceptedCmp3'] +
                          X['AcceptedCmp4'] + X['AcceptedCmp5'] + X['Response'])

print(f"\nDerived features created: Age, TotalSpending, TotalPurchases, CampaignsAccepted")

# Check statistics
print(f"\nData Statistics (first 5 features):")
print(X[features_to_use[:5]].describe())

print(f"\nFeature ranges:")
for col in X.columns:
    print(f"{col:25s}: min={X[col].min():10.2f}, max={X[col].max():10.2f}")

# Save cleaned data
X.to_csv('02_cleaned_data.csv', index=False)
print(f"\nCleaned data saved to: 02_cleaned_data.csv")

print("\n" + "=" * 70)
print("STAGE 1 COMPLETE")
print("=" * 70)
