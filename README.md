# 🍷 Wine Quality Prediction - Data Mining Project

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-v4.0-orange.svg)](https://lightgbm.readthedocs.io/)
[![CatBoost](https://img.shields.io/badge/CatBoost-v1.2-yellow.svg)](https://catboost.ai/)
[![Dataset](https://img.shields.io/badge/Dataset-Kaggle-blue.svg)](https://www.kaggle.com/datasets/jhalls/wine-dataset-270k-records)

Welcome to the **Wine Quality Prediction** repository. This project applies advanced data mining techniques to a real-world dataset to predict wine quality ratings (`points`) based on various features such as price, origin, and sommelier descriptions. 

---

## 1. Problem Identification & Dataset Selection

### 📊 Dataset Overview
- **Source:** Kaggle ([Wine Dataset - 270k records](https://www.kaggle.com/datasets/jhalls/wine-dataset-270k-records))
- **Total Records:** 129,971 (after initial import)
- **Total Attributes:** 13
- **Data Mining Task:** Prediction (Regression)

### 🎯 Objective
The primary objective of this analysis is to build a highly accurate predictive regression model to estimate wine `points` (quality rating on an 80-100 scale). By doing so, we aim to uncover which features (e.g., price, region, variety, or textual reviews) hold the most predictive power regarding wine quality.

### 🗂️ Attribute Measurement Scales
Based on the raw dataset exploration, here is the measurement scale for each attribute:

| Attribute | Data Type | Measurement Scale | Description |
| :--- | :--- | :--- | :--- |
| **country** | Categorical | Nominal | The country that the wine is from. |
| **description** | NLP (Free Text) | Nominal | A few sentences from a sommelier describing the wine. |
| **designation** | Categorical | Nominal | The vineyard within the winery where the grapes are sourced. |
| **points** | Numerical | Interval | The number of points WineEnthusiast rated the wine (Target). |
| **price** | Numerical | Ratio | The cost for a bottle of the wine. |
| **province** | Categorical | Nominal | The province or state that the wine is from. |
| **region_1** | Categorical | Nominal | The wine growing area in a province or state. |
| **region_2** | Categorical | Nominal | A more specific region specified within a wine growing area. |
| **taster_name** | Categorical | Nominal | Name of the person who tasted and reviewed the wine. |
| **taster_twitter_handle** | Categorical | Nominal | Twitter handle for the person who tasted and reviewed the wine. |
| **title** | NLP (Free Text) | Nominal | The title of the wine review. |
| **variety** | Categorical | Nominal | The type of grapes used to make the wine. |
| **winery** | Categorical | Nominal | The winery that made the wine. |

---

## 2. Data Preprocessing & Exploratory Data Analysis (EDA)

A robust preprocessing pipeline was engineered to ensure high-quality model inputs, strictly guided by feature importance and missing data thresholds.

### 🧹 Missing Value Handling
Features with massive missing percentages and low overall importance (e.g., `region_2` at 61.1%, `taster_name` at 20.2%) were dropped permanently. For our most critical feature, `price` (missing ~6.9%), rows were dropped to prevent injecting artificial relationships into the target.

> **Visual 1: Missing Values Overview** > *Note: Replace the path below with your actual image path* > <img src="assets/missing_values_hexagram.png" alt="Missing Values Hexagram" width="800"/>

### 🕵️ Outlier Detection & Skewness
We utilized box plots to identify extreme outliers in numerical columns. `price` exhibited a severe positive skew, heavily impacted by ultra-premium wines.

> **Visual 2: Extreme Value Handling** > *Note: Replace the path below with your actual image path* > <img src="assets/box_plots_outliers.png" alt="Box Plot for extreme value handling of price and points" width="800"/>

### 📈 Normalization & Distribution
To correct the heavy right-skew of the `price` feature, a `log1p()` transformation was applied. This mathematically stabilized the variance and significantly improved its linear correlation with the `points` target.

> **Visual 3: Data Distribution (Price & Points)** > *Note: Replace the path below with your actual image path* > <img src="assets/data_distribution.png" alt="Data Distribution for both price and points" width="800"/>

> **Visual 4: Price Log-Transformation Impact** > *Note: Replace the path below with your actual image path* > <img src="assets/price_log1p_comparison.png" alt="Price distribution before and after log1p" width="800"/>

### 🧩 Feature Engineering & Encoding Strategy
- **Low Cardinality (`country`, `taster_twitter_handle`):** One-Hot Encoding.
- **High Cardinality (`province`, `variety`, `region_1`):** Target Encoding via Mean Mapping.
- **Ultra-High Cardinality (`winery`, `designation`):** Frequency Encoding.
- **Free Text (`description`, `title`):** TF-IDF Vectorization.

> **Visual 5: Feature Correlation Heatmap** > *Note: Replace the path below with your actual image path* > <img src="assets/feature_correlation_heatmap.png" alt="Feature Correlation Heatmap" width="600"/>

---

## 3. Model Building & Performance Evaluation

Two advanced Gradient Boosting algorithms were implemented: **LightGBM** and **CatBoost**. Model performance was strictly evaluated using a multi-split approach (50:50, 60:40, 70:30, 80:20, 90:10).

### 🏆 Predictive Model Evaluation Matrix

| No | Fold Split | Algorithm 1: LightGBM <br> MAE | MSE | RMSE | R² | Algorithm 2: CatBoost <br> MAE | MSE | RMSE | R² |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **50:50** | 1.3461 | 3.1116 | 1.7640 | 0.6640 | 1.1281 | 2.1179 | 1.4553 | 0.7713 |
| 2 | **60:40** | 1.3222 | 3.0366 | 1.7426 | 0.6724 | 1.1161 | 2.0632 | 1.4364 | 0.7774 |
| 3 | **70:30** | 1.3018 | 2.9667 | 1.7224 | 0.6803 | 1.1066 | 2.0294 | 1.4246 | 0.7813 |
| 4 | **80:20** | 1.2802 | 2.9153 | 1.7074 | 0.6873 | 1.1057 | 2.0244 | 1.4228 | 0.7829 |
| 5 | **90:10** | 1.2678 | 2.8742 | 1.6953 | 0.6937 | 1.0989 | 2.0044 | 1.4158 | 0.7864 |
| **🌟** | **Best Fold By Metrics** | **1.2678** | **2.8742** | **1.6953** | **0.6937** | **1.0989** | **2.0044** | **1.4158** | **0.7864** |

### 🥇 Best Model Selection (Overall)
**Winner: CatBoost (90:10 Split)**
CatBoost natively handles categorical features and proved to be remarkably superior for this dataset. Across every single split, CatBoost consistently outperformed the highly-tuned LightGBM model. Even CatBoost's weakest split (50:50 with R²: 0.7713) beat LightGBM's absolute peak performance (90:10 with R²: 0.6937). 

### 💡 Insight Analysis
Both models exhibited ideal theoretical behavior: as the volume of training data increased, error metrics (MAE, RMSE) steadily decreased while the variance explained (R²) climbed. The gap between RMSE and MAE in the CatBoost model is minimal (~0.31 points), confirming that the model is stable and does not suffer from massive, wild outlier guesses.

---

## 4. Conclusion & Actionable Recommendations
1. **Price is Paramount:** The log-transformed price shows a strong correlation (`0.610`) with the target points. Wine quality and market price are deeply intertwined.
2. **Text Yields High Value:** Utilizing TF-IDF on sommelier descriptions successfully fed rich NLP signals to the tabular trees, vastly improving R².
3. **Deployment Strategy:** Due to its robust native handling of raw categorical text and superior R² metric, the finalized `catboost_wine_model.cbm` is highly recommended for production deployment over LightGBM.

---
### 👥 Project Team
* **Thone Dra**
* **Lay Pyay Hnin**
* **Soe Sandi Ag**

*Assignment for CCS23213 Data Mining*
