# notebook/feature_engineering.py

import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the cleaned dataset
df = pd.read_csv('data/cleaned_customer_churn.csv')

# Drop irrelevant columns
df.drop('customerID', axis=1, inplace=True)

# Map target column
df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})

# Create tenure groups
# Create tenure groups and encode them numerically
df['tenure_group'] = pd.cut(df['tenure'], bins=[0, 12, 24, 48, 60, 72], 
                            labels=[0, 1, 2, 3, 4])
df['tenure_group'] = df['tenure_group'].astype(int)

# Encode binary categorical columns
binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 
               'PaperlessBilling']

for col in binary_cols:
    df[col] = df[col].map({'Yes': 1, 'No': 0, 'Female': 0, 'Male': 1})

# Encode remaining categorical columns with LabelEncoder
cat_cols = df.select_dtypes(include='object').columns

le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# Define features (X) and target (y)
X = df.drop('Churn', axis=1)
y = df['Churn']

# Check output shapes
print("✅ Features shape:", X.shape)
print("🎯 Target shape:", y.shape)

# Optionally save preprocessed data
X.to_csv('data/features.csv', index=False)
y.to_csv('data/target.csv', index=False)
