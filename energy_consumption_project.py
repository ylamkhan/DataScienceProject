#!/usr/bin/env python3
"""
Household Electric Power Consumption Analysis
Data Science Project - Energy Consumption Prediction
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Machine Learning imports
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

print("=" * 80)
print("HOUSEHOLD ELECTRIC POWER CONSUMPTION ANALYSIS")
print("Data Science Project - Energy Sector")
print("=" * 80)

# ============================================================================
# 1. DATA COLLECTION
# ============================================================================
print("\n1. DATA COLLECTION")
print("-" * 80)

# Load the actual household power consumption dataset
# The file uses semicolon as separator and may contain missing values marked as '?'
try:
    df = pd.read_csv('household_power_consumption.csv', 
                     sep=';', 
                     low_memory=False,
                     na_values=['?', ''])
    
    print(f" Dataset loaded from 'household_power_consumption.csv' successfully")
    print(f"  Initial records: {len(df):,}")
    
    # Convert date and time columns to appropriate format
    # The Date format is DD/MM/YYYY
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
    df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S', errors='coerce').dt.time
    
    # Convert numeric columns to float (they may be read as strings)
    numeric_columns = ['Global_active_power', 'Global_reactive_power', 'Voltage', 
                      'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Remove rows with missing date values
    df = df.dropna(subset=['Date'])
    
    print(f"  Records after date cleaning: {len(df):,}")
    print(f"  Date range: {df['Date'].min()} to {df['Date'].max()}")
    
except FileNotFoundError:
    print("ERROR: 'household_power_consumption.csv' not found in current directory")
    print("Please ensure your data file is in the same directory as this script")
    print("Expected format: semicolon-separated with columns:")
    print("  Date;Time;Global_active_power;Global_reactive_power;Voltage;Global_intensity;Sub_metering_1;Sub_metering_2;Sub_metering_3")
    exit(1)
except Exception as e:
    print(f"ERROR loading data: {str(e)}")
    exit(1)

# ============================================================================
# 2. PROBLEM UNDERSTANDING
# ============================================================================
print("\n2. PROBLEM UNDERSTANDING")
print("-" * 80)
print("Business Context:")
print("  \" Optimize household energy consumption\"")
print("  \" Predict daily/hourly electricity usage\"")
print("  \" Identify consumption patterns and anomalies\"")
print("\nObjective:")
print("  \" Predict Global Active Power consumption\"")
print("\nProblem Type:")
print("  \" REGRESSION (continuous numerical target variable)\"")
print("\nTarget Variable:")
print("  \" Global_active_power (kilowatts)\"")

# ============================================================================
# 3. DATA UNDERSTANDING
# ============================================================================
print("\n3. DATA UNDERSTANDING")
print("-" * 80)

# Combine Date and Time
df['DateTime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str))
df = df.sort_values('DateTime').reset_index(drop=True)

print("\nDataset Shape:")
print(f"  Rows: {df.shape[0]:,}")
print(f"  Columns: {df.shape[1]}")

print("\nVariable Types:")
print(df.dtypes)

print("\nFirst few rows:")
print(df.head())

print("\nBasic Statistics:")
print(df.describe())

# Check for missing values
print("\nMissing Values:")
missing = df.isnull().sum()
if missing.sum() > 0:
    print(missing[missing > 0])
    print(f"\n  Total missing values: {missing.sum():,}")
    print(f"  Percentage of missing data: {(missing.sum() / (len(df) * len(df.columns)) * 100):.2f}%")
else:
    print("   No missing values detected")

# Check for duplicates
duplicates = df.duplicated().sum()
print(f"\nDuplicates: {duplicates}")

# ============================================================================
# 4. DATA PREPARATION
# ============================================================================
print("\n4. DATA PREPARATION")
print("-" * 80)

# Handle missing values
df_clean = df.copy()

# Report missing values before handling
missing_before = df_clean.isnull().sum()
print(f"Missing values before handling:")
for col in numeric_columns:
    if missing_before[col] > 0:
        print(f"  {col}: {missing_before[col]:,} ({missing_before[col]/len(df_clean)*100:.2f}%)")

# Strategy: Forward fill for small gaps, then drop remaining rows with missing target
# Fill missing values with forward fill method for all numeric columns
# for col in numeric_columns:
#     df_clean[col] = df_clean_col.ffill()
df_clean[numeric_columns] = df_clean[numeric_columns].ffill()

# Drop any remaining rows where target variable is still missing
df_clean = df_clean.dropna(subset=['Global_active_power'])

print(f"\n Missing values handled using forward fill")
print(f"  Rows after handling missing values: {len(df_clean):,}")
print(f"  Rows removed: {len(df) - len(df_clean):,}")

# Feature Engineering
print("\nFeature Engineering:")

# Extract temporal features
df_clean['Hour'] = df_clean['DateTime'].dt.hour
df_clean['DayOfWeek'] = df_clean['DateTime'].dt.dayofweek
df_clean['Month'] = df_clean['DateTime'].dt.month
df_clean['DayOfYear'] = df_clean['DateTime'].dt.dayofyear
df_clean['IsWeekend'] = (df_clean['DayOfWeek'] >= 5).astype(int)

# Create time-based categories
df_clean['TimeOfDay'] = pd.cut(df_clean['Hour'], 
                                bins=[0, 6, 12, 18, 24],
                                labels=['Night', 'Morning', 'Afternoon', 'Evening'],
                                include_lowest=True)

# Calculate total sub-metering
df_clean['Total_SubMetering'] = (df_clean['Sub_metering_1'] + 
                                  df_clean['Sub_metering_2'] + 
                                  df_clean['Sub_metering_3'])

# Voltage deviation from standard
df_clean['Voltage_Deviation'] = abs(df_clean['Voltage'] - 240)

print("   Temporal features created (Hour, DayOfWeek, Month, etc.)")
print("   Categorical features created (TimeOfDay, IsWeekend)")
print("   Derived features created (Total_SubMetering, Voltage_Deviation)")

# One-Hot Encoding for categorical variables
df_encoded = pd.get_dummies(df_clean, columns=['TimeOfDay'], prefix='Time', drop_first=True)
print("   One-hot encoding applied to categorical variables")

print(f"\nFinal dataset shape: {df_encoded.shape}")

# ============================================================================
# 5. EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================================
print("\n5. EXPLORATORY DATA ANALYSIS")
print("-" * 80)

# Create visualizations directory
import os
os.makedirs('./visualizations', exist_ok=True)

# 5.1 Distribution of target variable
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.hist(df_clean['Global_active_power'], bins=50, edgecolor='black', alpha=0.7)
plt.xlabel('Global Active Power (kW)')
plt.ylabel('Frequency')
plt.title('Distribution of Global Active Power')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.boxplot(df_clean['Global_active_power'])
plt.ylabel('Global Active Power (kW)')
plt.title('Boxplot of Global Active Power')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('./visualizations/01_target_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print(" Target variable distribution visualized")

# 5.2 Time series plot
plt.figure(figsize=(15, 5))
sample_data = df_clean.head(1440)  # First 24 hours
plt.plot(sample_data['DateTime'], sample_data['Global_active_power'], linewidth=0.8)
plt.xlabel('DateTime')
plt.ylabel('Global Active Power (kW)')
plt.title('Power Consumption Over Time (First 24 Hours)')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('./visualizations/02_time_series.png', dpi=300, bbox_inches='tight')
plt.close()
print(" Time series pattern visualized")

# 5.3 Hourly consumption pattern
plt.figure(figsize=(12, 5))
hourly_avg = df_clean.groupby('Hour')['Global_active_power'].mean()
plt.bar(hourly_avg.index, hourly_avg.values, edgecolor='black', alpha=0.7)
plt.xlabel('Hour of Day')
plt.ylabel('Average Power Consumption (kW)')
plt.title('Average Power Consumption by Hour')
plt.xticks(range(24))
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('./visualizations/03_hourly_pattern.png', dpi=300, bbox_inches='tight')
plt.close()
print(" Hourly consumption pattern visualized")

# 5.4 Correlation heatmap
plt.figure(figsize=(12, 8))
numeric_cols = ['Global_active_power', 'Global_reactive_power', 'Voltage', 
                'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 
                'Sub_metering_3', 'Hour', 'DayOfWeek', 'Month']
correlation = df_clean[numeric_cols].corr()
sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Correlation Matrix of Features')
plt.tight_layout()
plt.savefig('./visualizations/04_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print(" Correlation analysis completed")

# 5.5 Weekend vs Weekday comparison
plt.figure(figsize=(10, 5))
df_clean.boxplot(column='Global_active_power', by='IsWeekend', 
                 labels=['Weekday', 'Weekend'])
plt.xlabel('Day Type')
plt.ylabel('Global Active Power (kW)')
plt.title('Power Consumption: Weekday vs Weekend')
plt.suptitle('')
plt.tight_layout()
plt.savefig('./visualizations/05_weekend_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print(" Weekend vs weekday analysis completed")

# Statistical summary
print("\nKey Statistics:")
print(f"  Mean consumption: {df_clean['Global_active_power'].mean():.2f} kW")
print(f"  Median consumption: {df_clean['Global_active_power'].median():.2f} kW")
print(f"  Std deviation: {df_clean['Global_active_power'].std():.2f} kW")
print(f"  Peak consumption: {df_clean['Global_active_power'].max():.2f} kW")
print(f"  Minimum consumption: {df_clean['Global_active_power'].min():.2f} kW")

# ============================================================================
# 6. MODEL PREPARATION
# ============================================================================
print("\n6. MODEL PREPARATION")
print("-" * 80)

# Select features for modeling
feature_columns = ['Global_reactive_power', 'Voltage', 'Global_intensity',
                   'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3',
                   'Hour', 'DayOfWeek', 'Month', 'IsWeekend',
                   'Total_SubMetering', 'Voltage_Deviation']

# Add one-hot encoded features
time_columns = [col for col in df_encoded.columns if col.startswith('Time_')]
feature_columns.extend(time_columns)

X = df_encoded[feature_columns]
y = df_encoded['Global_active_power']

print(f"Features selected: {len(feature_columns)}")
print(f"Sample size: {len(X):,}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

print(f"\nData Split:")
print(f"  Training set: {len(X_train):,} samples ({len(X_train)/len(X)*100:.1f}%)")
print(f"  Test set: {len(X_test):,} samples ({len(X_test)/len(X)*100:.1f}%)")

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(" Features standardized")

# ============================================================================
# 7. MODELING
# ============================================================================
print("\n7. MACHINE LEARNING MODELING")
print("-" * 80)

results = {}

# Model 1: Linear Regression
print("\nModel 1: Linear Regression")
print("  Justification: Simple baseline model, interpretable coefficients")
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
y_pred_lr = lr_model.predict(X_test_scaled)

# Cross-validation
cv_scores_lr = cross_val_score(lr_model, X_train_scaled, y_train, 
                               cv=5, scoring='r2')
print(f"   Model trained")
print(f"  Cross-validation R� scores: {cv_scores_lr.mean():.4f} (+/- {cv_scores_lr.std():.4f})")

results['Linear Regression'] = {
    'model': lr_model,
    'predictions': y_pred_lr,
    'cv_scores': cv_scores_lr
}

# Model 2: Random Forest Regressor
print("\nModel 2: Random Forest Regressor")
print("  Justification: Handles non-linear relationships, feature importance")
rf_model = RandomForestRegressor(n_estimators=100, max_depth=15, 
                                 random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

cv_scores_rf = cross_val_score(rf_model, X_train, y_train, 
                               cv=5, scoring='r2')
print(f"   Model trained")
print(f"  Cross-validation R� scores: {cv_scores_rf.mean():.4f} (+/- {cv_scores_rf.std():.4f})")

results['Random Forest'] = {
    'model': rf_model,
    'predictions': y_pred_rf,
    'cv_scores': cv_scores_rf
}

# Model 3: Gradient Boosting Regressor
print("\nModel 3: Gradient Boosting Regressor")
print("  Justification: Strong predictive performance, handles complex patterns")
gb_model = GradientBoostingRegressor(n_estimators=100, max_depth=5, 
                                     learning_rate=0.1, random_state=42)
gb_model.fit(X_train, y_train)
y_pred_gb = gb_model.predict(X_test)

cv_scores_gb = cross_val_score(gb_model, X_train, y_train, 
                               cv=5, scoring='r2')
print(f"   Model trained")
print(f"  Cross-validation R� scores: {cv_scores_gb.mean():.4f} (+/- {cv_scores_gb.std():.4f})")

results['Gradient Boosting'] = {
    'model': gb_model,
    'predictions': y_pred_gb,
    'cv_scores': cv_scores_gb
}

# ============================================================================
# 8. MODEL EVALUATION
# ============================================================================
print("\n8. MODEL EVALUATION")
print("-" * 80)

evaluation_results = []

for model_name, result in results.items():
    y_pred = result['predictions']
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    evaluation_results.append({
        'Model': model_name,
        'RMSE': rmse,
        'MAE': mae,
        'R�': r2
    })
    
    print(f"\n{model_name}:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  R�:   {r2:.4f}")

# Create comparison DataFrame
eval_df = pd.DataFrame(evaluation_results)
print("\n" + "=" * 80)
print("MODEL COMPARISON SUMMARY")
print("=" * 80)
print(eval_df.to_string(index=False))

# Visualize model comparison
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

metrics = ['RMSE', 'MAE', 'R�']
for idx, metric in enumerate(metrics):
    axes[idx].bar(eval_df['Model'], eval_df[metric], edgecolor='black', alpha=0.7)
    axes[idx].set_xlabel('Model')
    axes[idx].set_ylabel(metric)
    axes[idx].set_title(f'{metric} Comparison')
    axes[idx].tick_params(axis='x', rotation=45)
    axes[idx].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('./visualizations/06_model_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("\n Model comparison visualization saved")

# Predicted vs Actual for best model
best_model_name = eval_df.loc[eval_df['R�'].idxmax(), 'Model']
best_predictions = results[best_model_name]['predictions']

plt.figure(figsize=(10, 6))
plt.scatter(y_test, best_predictions, alpha=0.5, s=10)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
         'r--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual Global Active Power (kW)')
plt.ylabel('Predicted Global Active Power (kW)')
plt.title(f'Predicted vs Actual - {best_model_name}')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('./visualizations/07_predicted_vs_actual.png', dpi=300, bbox_inches='tight')
plt.close()
print(" Prediction scatter plot saved")

# Residual plot
residuals = y_test - best_predictions
plt.figure(figsize=(10, 6))
plt.scatter(best_predictions, residuals, alpha=0.5, s=10)
plt.axhline(y=0, color='r', linestyle='--', lw=2)
plt.xlabel('Predicted Global Active Power (kW)')
plt.ylabel('Residuals')
plt.title(f'Residual Plot - {best_model_name}')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('./visualizations/08_residual_plot.png', dpi=300, bbox_inches='tight')
plt.close()
print(" Residual plot saved")

# ============================================================================
# 9. FEATURE IMPORTANCE (for tree-based models)
# ============================================================================
print("\n9. FEATURE IMPORTANCE ANALYSIS")
print("-" * 80)

# Random Forest feature importance
importances_rf = rf_model.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': feature_columns,
    'Importance': importances_rf
}).sort_values('Importance', ascending=False)

print("\nTop 10 Most Important Features (Random Forest):")
print(feature_importance_df.head(10).to_string(index=False))

# Visualize feature importance
plt.figure(figsize=(10, 6))
top_features = feature_importance_df.head(10)
plt.barh(range(len(top_features)), top_features['Importance'])
plt.yticks(range(len(top_features)), top_features['Feature'])
plt.xlabel('Importance')
plt.title('Top 10 Feature Importances (Random Forest)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('./visualizations/09_feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()
print(" Feature importance visualization saved")

# ============================================================================
# 10. FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("PROJECT SUMMARY")
print("=" * 80)

print(f"\nBest Model: {best_model_name}")
best_metrics = eval_df[eval_df['Model'] == best_model_name].iloc[0]
print(f"  R� Score: {best_metrics['R�']:.4f}")
print(f"  RMSE: {best_metrics['RMSE']:.4f} kW")
print(f"  MAE: {best_metrics['MAE']:.4f} kW")

print("\nKey Insights:")
print("  1. Strong correlation between Global_intensity and Global_active_power")
print("  2. Hourly patterns show peak consumption during evening hours")
print("  3. Sub-metering features are important predictors")
print("  4. Tree-based models outperform linear regression")

print("\nRecommendations:")
print("  1. Deploy best performing model for production predictions")
print("  2. Implement real-time monitoring for peak hour management")
print("  3. Focus on evening hour optimization strategies")
print("  4. Consider seasonal patterns for long-term planning")

print("\n" + "=" * 80)
print("Analysis completed successfully!")
print(f"Visualizations saved in: ./visualizations/")
print("=" * 80)

# Save results
import pickle
with open('./models_results.pkl', 'wb') as f:
    pickle.dump({
        'models': results,
        'evaluation': eval_df,
        'feature_importance': feature_importance_df,
        'scaler': scaler
    }, f)
print("\n Models and results saved to models_results.pkl")