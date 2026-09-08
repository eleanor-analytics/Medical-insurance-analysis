
    """
U.S. Medical Insurance Cost Analysis
Author: Eleanor Bryan
Date: 2026
Purpose: Explore factors affecting medical insurance costs and build a predictive model.
"""

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


# 1. LOAD AND EXPLORE DATA

# Load the dataset
df = pd.read_csv('insurance.csv')

# Display basic information
print("=== DATA OVERVIEW ===")
print(f"Shape: {df.shape}")
print("\nFirst 5 rows:")
print(df.head())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\nSummary statistics:")
print(df.describe())


# 2. DATA CLEANING & PREPARATION


# No missing values found, but let's ensure categorical variables are properly formatted
df['sex'] = df['sex'].astype('category')
df['smoker'] = df['smoker'].astype('category')
df['region'] = df['region'].astype('category')

# Create dummy variables for categorical columns
df_encoded = pd.get_dummies(df, columns=['sex', 'smoker', 'region'], drop_first=True)

print("\nEncoded column names:")
print(df_encoded.columns.tolist())


# 3. EXPLORATORY DATA ANALYSIS

# Set up visualisation style
sns.set_style('whitegrid')
plt.figure(figsize=(12, 8))

# Distribution of charges
plt.subplot(2, 3, 1)
sns.histplot(df['charges'], bins=30, kde=True)
plt.title('Distribution of Insurance Charges')
plt.xlabel('Charges ($)')

# Charges by smoking status
plt.subplot(2, 3, 2)
sns.boxplot(x='smoker', y='charges', data=df)
plt.title('Charges by Smoking Status')
plt.xlabel('Smoker')
plt.ylabel('Charges ($)')

# Charges by region
plt.subplot(2, 3, 3)
sns.boxplot(x='region', y='charges', data=df)
plt.title('Charges by Region')
plt.xlabel('Region')
plt.ylabel('Charges ($)')

# Charges by sex
plt.subplot(2, 3, 4)
sns.boxplot(x='sex', y='charges', data=df)
plt.title('Charges by Sex')
plt.xlabel('Sex')
plt.ylabel('Charges ($)')

# Correlation heatmap
plt.subplot(2, 3, 5)
correlation = df_encoded.corr()
sns.heatmap(correlation[['charges']].sort_values(by='charges', ascending=False), 
            annot=True, cmap='coolwarm', cbar=False)
plt.title('Correlation with Charges')

# BMI vs Charges by smoking status
plt.subplot(2, 3, 6)
sns.scatterplot(x='bmi', y='charges', hue='smoker', data=df, alpha=0.6)
plt.title('BMI vs Charges by Smoking Status')
plt.xlabel('BMI')
plt.ylabel('Charges ($)')

plt.tight_layout()
plt.savefig('eda_visualisations.png', dpi=300)
plt.show()


# 4. KEY INSIGHTS


print("\n=== KEY INSIGHTS ===")
print(f"Average charge: ${df['charges'].mean():,.2f}")
print(f"Median charge: ${df['charges'].median():,.2f}")
print(f"Maximum charge: ${df['charges'].max():,.2f}")

# Average charges by smoker status
smoker_avg = df.groupby('smoker')['charges'].mean()
print(f"\nAverage charges - Smoker: ${smoker_avg['yes']:,.2f}")
print(f"Average charges - Non-smoker: ${smoker_avg['no']:,.2f}")
print(f"Difference: ${smoker_avg['yes'] - smoker_avg['no']:,.2f}")

# Average charges by region
region_avg = df.groupby('region')['charges'].mean()
print("\nAverage charges by region:")
for region, avg in region_avg.items():
    print(f"  {region}: ${avg:,.2f}")


# 5. PREDICTIVE MODELLING


# Define features (X) and target (y)
X = df_encoded.drop('charges', axis=1)
y = df_encoded['charges']

# Split into training and testing sets (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")

# Train linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("\n=== MODEL PERFORMANCE ===")
print(f"R-squared: {r2_score(y_test, y_pred):.4f}")
print(f"Mean Absolute Error: ${mean_absolute_error(y_test, y_pred):,.2f}")

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
})
feature_importance = feature_importance.sort_values('Coefficient', ascending=False)
print("\nTop 5 most influential features on charges:")
print(feature_importance.head(5))


# 6. SAMPLE PREDICTIONS


# Create a function to predict charges
def predict_charges(age, bmi, children, smoker, sex, region):
    """
    Predict insurance charges based on patient characteristics.
    """
    # Create a DataFrame with the same columns as X
    input_data = pd.DataFrame(0, index=[0], columns=X.columns)
    
    # Fill in numeric values
    input_data['age'] = age
    input_data['bmi'] = bmi
    input_data['children'] = children
    
    # Fill in categorical variables
    if smoker.lower() == 'yes':
        input_data['smoker_yes'] = 1
    if sex.lower() == 'male':
        input_data['sex_male'] = 1
    if region.lower() == 'northwest':
        input_data['region_northwest'] = 1
    elif region.lower() == 'southeast':
        input_data['region_southeast'] = 1
    elif region.lower() == 'southwest':
        input_data['region_southwest'] = 1
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    return prediction

# Test the function
sample_patient = {
    'age': 30,
    'bmi': 27.5,
    'children': 1,
    'smoker': 'no',
    'sex': 'male',
    'region': 'southeast'
}

predicted_charge = predict_charges(**sample_patient)
print(f"\n=== SAMPLE PREDICTION ===")
print(f"Patient: {sample_patient}")
print(f"Predicted charge: ${predicted_charge:,.2f}")

print("\n=== ANALYSIS COMPLETE ===")