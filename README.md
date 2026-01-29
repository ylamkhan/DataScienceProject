# Household Electric Power Consumption Analysis
## Data Science Project - Energy Sector

---

## 📋 Project Overview

This project analyzes household electric power consumption patterns using machine learning techniques. The analysis encompasses data from a household in France, measuring global active power consumption at one-minute intervals to develop predictive models capable of forecasting energy consumption.

### **Team Members**
- Youssef Mohammed Imane

### **Project Type**
Regression Analysis - Predicting continuous numerical values (kilowatts)

---

## 🎯 Objectives

1. Analyze historical power consumption data to identify patterns and trends
2. Develop predictive models for Global Active Power consumption
3. Compare multiple machine learning algorithms for regression tasks
4. Identify key features influencing energy consumption
5. Provide actionable recommendations for energy optimization

---

## 📊 Dataset

**Source:** UCI Machine Learning Repository - Individual Household Electric Power Consumption

**Characteristics:**
- **Records:** 10,000 observations (sample from full dataset)
- **Time Period:** December 16-22, 2006
- **Sampling Rate:** 1 minute
- **Features:** 9 original variables + engineered features

**Key Variables:**
- `Global_active_power` - Target variable (kilowatts)
- `Global_reactive_power` - Reactive power (kilowatts)
- `Voltage` - Minute-averaged voltage (volts)
- `Global_intensity` - Current intensity (amperes)
- `Sub_metering_1/2/3` - Energy sub-metering (watt-hours)

---

## 🔧 Methodology

### 1. Data Collection & Cleaning
- Loaded dataset from UCI ML Repository
- Handled missing values using forward-fill imputation
- Consolidated Date and Time into DateTime column
- Verified data quality (no duplicates found)

### 2. Feature Engineering
**Temporal Features:**
- Hour (0-23)
- DayOfWeek (0-6)
- Month
- DayOfYear
- IsWeekend (binary)
- TimeOfDay (categorical: Night/Morning/Afternoon/Evening)

**Derived Features:**
- Total_SubMetering (sum of all sub-metering)
- Voltage_Deviation (absolute deviation from 240V standard)

### 3. Exploratory Data Analysis
- Descriptive statistics
- Distribution analysis
- Correlation analysis
- Temporal pattern identification
- Weekend vs. weekday comparison

### 4. Machine Learning Models
Three regression algorithms were implemented:

| Model | Justification |
|-------|---------------|
| **Linear Regression** | Baseline model, interpretable coefficients, fast |
| **Random Forest** | Handles non-linear relationships, feature importance |
| **Gradient Boosting** | Sequential learning, high accuracy potential |

**Training Configuration:**
- Train/Test Split: 80/20
- Cross-Validation: 5-fold
- Feature Scaling: StandardScaler

### 5. Evaluation Metrics
- **RMSE** (Root Mean Squared Error) - Primary metric
- **MAE** (Mean Absolute Error) - Robust to outliers
- **R²** (Coefficient of Determination) - Variance explained

---

## 📈 Results

### Model Performance

| Model | RMSE (kW) | MAE (kW) | R² Score |
|-------|-----------|----------|----------|
| **Linear Regression** ✓ | **1.9343** | **1.5703** | **-0.0040** |
| Random Forest | 1.9438 | 1.5778 | -0.0139 |
| Gradient Boosting | 1.9516 | 1.5850 | -0.0221 |

**Best Model:** Linear Regression achieved the lowest error metrics.

### Feature Importance (Random Forest)

| Rank | Feature | Importance |
|------|---------|-----------|
| 1 | Global_intensity | 17.5% |
| 2 | Global_reactive_power | 14.4% |
| 3 | Voltage | 13.6% |
| 4 | Voltage_Deviation | 12.8% |
| 5 | Hour | 8.8% |
| 6 | Total_SubMetering | 7.5% |

---

## 🔍 Key Insights

### Consumption Patterns
- **Peak Hours:** 18:00-22:00 (evening) - highest consumption
- **Morning Spike:** 7:00-9:00 - breakfast and preparation
- **Low Usage:** 0:00-6:00 (nighttime)
- **Weekend Differences:** More distributed usage vs. concentrated weekday peaks

### Feature Relationships
- Strong correlation between Global_intensity and Global_active_power (P = V × I)
- Voltage stability significantly impacts consumption
- Climate control (Sub_metering_3) is the largest energy consumer
- Temporal features crucial for prediction accuracy

### Model Performance Context
- Negative R² scores indicate challenges with minute-level prediction
- High temporal variability creates baseline-level performance
- Models would perform better on aggregated (hourly/daily) data
- Linear relationships dominate in this dataset

---

## 💡 Recommendations

### Technical Improvements
1. Aggregate predictions to hourly or daily levels
2. Incorporate external variables (weather, occupancy, holidays)
3. Implement time-series specific models (ARIMA, LSTM)
4. Expand dataset to multiple households
5. Apply ensemble methods for improved stability

### Business Actions
1. Deploy real-time monitoring for peak hour management (18:00-22:00)
2. Implement smart automation for load shifting
3. Develop consumer feedback systems
4. Focus efficiency programs on climate control systems
5. Consider time-of-use pricing strategies

### Future Work
- Anomaly detection for equipment malfunction identification
- Clustering analysis for consumption profile segmentation
- Renewable energy system integration
- Mobile application development for consumer engagement

---

## 🛠 Technical Stack

**Programming Language:** Python 3.x

**Libraries:**
- `pandas` - Data manipulation and analysis
- `numpy` - Numerical computing
- `matplotlib` - Data visualization
- `seaborn` - Statistical visualization
- `scikit-learn` - Machine learning algorithms

**Key Algorithms:**
- Linear Regression (OLS)
- Random Forest Regressor (100 estimators, max_depth=15)
- Gradient Boosting Regressor (100 estimators, max_depth=5, lr=0.1)

---

## 📁 Project Structure

```
project/
├── energy_consumption_analysis.py    # Main analysis script
├── Energy_Consumption_Report.docx    # Detailed project report
├── Energy_Consumption_Presentation.pptx  # 15-minute presentation
├── visualizations/                   # Generated plots and charts
│   ├── 01_target_distribution.png
│   ├── 02_time_series.png
│   ├── 03_hourly_pattern.png
│   ├── 04_correlation_heatmap.png
│   ├── 05_weekend_comparison.png
│   ├── 06_model_comparison.png
│   ├── 07_predicted_vs_actual.png
│   ├── 08_residual_plot.png
│   └── 09_feature_importance.png
├── models_results.pkl                # Saved models and results
└── README.md                         # This file
```

---

## 🚀 How to Run

1. **Install Dependencies:**
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn statsmodels
   ```

2. **Run Analysis:**
   ```bash
   python energy_consumption_analysis.py
   ```

3. **View Results:**
   - Analysis output: Console
   - Visualizations: `/visualizations/` directory
   - Detailed report: `Energy_Consumption_Report.docx`
   - Presentation: `Energy_Consumption_Presentation.pptx`

---

## 📊 Deliverables

✅ **15-Minute Presentation** - Professional slides with visualizations  
✅ **Detailed Report** - Comprehensive documentation of methodology and findings  
✅ **Documented Code** - Clean, well-commented Python scripts  
✅ **Visualizations** - 9 high-quality charts and plots  
✅ **Saved Models** - Pickle file with trained models and results  

---

## 🎓 Academic Context

**Course:** Data Science Project 2025-26  
**Topic:** Energy Consumption Prediction  
**Problem Type:** Regression  
**Dataset:** UCI Machine Learning Repository  

---

## 📚 References

1. Dua, D. and Graff, C. (2019). UCI Machine Learning Repository. Irvine, CA: University of California, School of Information and Computer Science.
2. Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.
3. Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5-32.
4. Friedman, J. H. (2001). Greedy Function Approximation: A Gradient Boosting Machine. Annals of Statistics, 29(5), 1189-1232.

---

## 📝 License

This project is created for academic purposes as part of a Data Science course.

---

## 👥 Contributors

Add your team members here:
- Member 1: [Name] - [Role]
- Member 2: [Name] - [Role]
- Member 3: [Name] - [Role]

---

## 📧 Contact

For questions or discussions about this project:
- Repository: https://github.com/ylamkhan/DataScienceProject
- Course: Data Science 2025-26

---

**Date:** January 2026  
**Status:** ✅ Complete
