import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv("Electric_Vehicle_Population_Data.csv")

# Step 1: Handle missing values
# Check for missing values
print(df.isnull().sum())

# Drop unnecessary columns
df = df.drop(columns=['Unnamed: 3', 'Unnamed: 14'], errors='ignore')

# Fill missing values without using inplace=True
df['Country'] = df['Country'].fillna(df['Country'].mode()[0])
df['City'] = df['City'].fillna(df['City'].mode()[0])
df['Postal Code'] = df['Postal Code'].fillna(df['Postal Code'].mode()[0])
df['Legislative District'] = df['Legislative District'].fillna(df['Legislative District'].mode()[0])
df['Electric Utility'] = df['Electric Utility'].fillna(df['Electric Utility'].mode()[0])

# Step 2: Handle outliers
# Use IQR to detect outliers in 'Electric Range' and 'Base MSRP'
Q1 = df[['Electric Range', 'Base MSRP']].quantile(0.25)
Q3 = df[['Electric Range', 'Base MSRP']].quantile(0.75)
IQR = Q3 - Q1

# Remove outliers
df = df[~((df[['Electric Range', 'Base MSRP']] < (Q1 - 1.5 * IQR)) | (df[['Electric Range', 'Base MSRP']] > (Q3 + 1.5 * IQR))).any(axis=1)]

# Step 3: Normalize or scale features
# Select features for scaling
features_to_scale = ['Electric Range', 'Base MSRP']
scaler = StandardScaler()

# Fit and transform the scaler
df[features_to_scale] = scaler.fit_transform(df[features_to_scale])

# Step 4: Split the data into training and testing sets
# Assume you want to predict 'Base MSRP' as the target variable
X = df.drop(columns=['Base MSRP'])  # Features
y = df['Base MSRP']                  # Target variable

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Output the shapes of the resulting datasets
print("Training set shape:", X_train.shape, y_train.shape)
print("Testing set shape:", X_test.shape, y_test.shape)

# Check for missing values
print(df.isnull().sum())