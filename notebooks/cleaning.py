import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('data/Customer_Churn.csv')

# Basic info
print("\n🔍 First 5 rows:")
print(df.head())

print("\n📋 Dataset Info:")
print(df.info())

print("\n🧮 Summary Statistics:")
print(df.describe(include='all'))

print("\n🧼 Missing Values:")
print(df.isnull().sum())

# Check data types
print("\n🔢 Data Types:")
print(df.dtypes)

# Check unique values in each column
print("\n🔍 Unique values per column:")
for col in df.columns:
    print(f"{col}: {df[col].nunique()}")

# Strip whitespace in column names (if any)
df.columns = df.columns.str.strip()

# Convert TotalCharges to numeric (some values may be empty strings)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Check again for nulls
print("\n🧼 Missing after type conversion:")
print(df.isnull().sum())

# Drop rows with nulls (or you can fill if very few)
df.dropna(inplace=True)

# Set plot style
sns.set(style="whitegrid")

# 1. Churn distribution
sns.countplot(data=df, x='Churn')
plt.title('Churn Distribution')
plt.show()

# 2. Churn vs Contract type
sns.countplot(data=df, x='Contract', hue='Churn')
plt.title('Churn by Contract Type')
plt.show()

# 3. Monthly Charges vs Churn
sns.boxplot(data=df, x='Churn', y='MonthlyCharges')
plt.title('Monthly Charges by Churn Status')
plt.show()

# 4. Correlation heatmap (numeric only)
corr = df.corr(numeric_only=True)
plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# Create tenure bins
df['tenure_group'] = pd.cut(df['tenure'], bins=[0, 12, 24, 48, 60, 72], 
                            labels=['0-12', '13-24', '25-48', '49-60', '61-72'])

sns.countplot(data=df, x='tenure_group', hue='Churn')
plt.title('Churn by Tenure Group')
plt.show()

sns.countplot(data=df, x='InternetService', hue='Churn')
plt.title('Churn by Internet Service Type')
plt.show()

plt.figure(figsize=(10,4))
sns.countplot(data=df, y='PaymentMethod', hue='Churn')
plt.title('Churn by Payment Method')
plt.show()

# Drop rows with missing TotalCharges
df.dropna(subset=['TotalCharges'], inplace=True)

df.reset_index(drop=True, inplace=True)

df.to_csv('data/cleaned_customer_churn.csv', index=False)
