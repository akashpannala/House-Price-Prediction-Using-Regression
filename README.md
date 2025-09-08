# 🏡 House Price Prediction – Kaggle (Advanced Regression Techniques)

## 📌 Overview
This project is based on the **House Prices: Advanced Regression Techniques** Kaggle competition.  
The goal is to predict house prices using various regression techniques after proper data preprocessing and feature engineering.  

Final Score: **0.18663 RMSLE**  

---

##  Workflow Summary

### 1. Import Libraries
- Loaded essential libraries: `numpy`, `pandas`, `scikit-learn` preprocessing tools, and `LinearRegression`.

### 2. Load Data
- Read training data (`train.csv`) and test data (`test.csv`) into pandas DataFrames.

### 3. Prepare Data
- Separated target variable (`SalePrice`) from training data.  
- Concatenated train and test features (excluding `SalePrice`) for unified preprocessing.

### 4. Handle Missing Values
- Defined `fill_none_categories()` to handle missing values:  
  - Filled missing categorical features with `"None"`.  
  - Filled numeric features with median, zero, or related column values.

### 5. Encode Features
- Defined `correct_encoding()`:  
  - Mapped ordinal quality features to integers.  
  - Applied one-hot encoding to categorical variables using `pd.get_dummies`.

### 6. Final Missing Value Handling
- Filled any remaining missing values with zero.

### 7. Split Data Back
- Split the preprocessed dataset back into training and test sets (based on original training set length).

### 8. Train Model
- Trained a **Linear Regression model** on the processed training data and target variable.

### 9. Predict
- Predicted house prices for the test set.

### 10. Save Submission
- Created a submission file:  
  - DataFrame with test IDs and predicted prices.  
  - Exported as `submission.csv` (no index).  
