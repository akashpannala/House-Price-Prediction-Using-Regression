import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def load_data():
    df = pd.read_csv('datasets/train.csv')
    tdf = pd.read_csv("datasets/test.csv")
    return df, tdf

def fill_none_categories(df):
    categorical_none_features = [
        'PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 
        'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
        'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 
        'BsmtFinType2', 'MasVnrType'
    ]
    
    for feature in categorical_none_features:
        if feature in df.columns:
            df[feature] = df[feature].fillna('None')
    
    if 'MasVnrArea' in df.columns:
        df['MasVnrArea'] = df['MasVnrArea'].fillna(0)
    if 'GarageYrBlt' in df.columns:
        df['GarageYrBlt'] = df['GarageYrBlt'].fillna(df['YearBuilt'])
    if 'LotFrontage' in df.columns:
        df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].median())
    if 'Electrical' in df.columns:
        df['Electrical'] = df['Electrical'].fillna('SBrkr')
    
    return df

def correct_encoding(df):
    df_encoded = df.copy()
    quality_features = ['ExterQual', 'KitchenQual', 'BsmtQual', 'HeatingQC', 'GarageQual']
    quality_map = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
    
    for feature in quality_features:
        if feature in df.columns:
            df_encoded[f'{feature}_Encoded'] = df_encoded[feature].map(quality_map).fillna(0)
    
    remaining_categorical = df_encoded.select_dtypes(include=['object']).columns
    df_encoded = pd.get_dummies(df_encoded, columns=remaining_categorical, drop_first=True)
    df_encoded = df_encoded.drop(columns=quality_features, errors='ignore')
    return df_encoded

def train_and_save_model():
    print("Training model...")
    df, tdf = load_data()
    df_price = df['SalePrice'] 
    df_all = pd.concat([df.drop('SalePrice', axis=1), tdf], ignore_index=True)
    
    df_all = fill_none_categories(df_all)
    df_all = correct_encoding(df_all)
    df_all = df_all.fillna(0)
    
    n_train = len(df)
    df_train = df_all.iloc[:n_train, :]
    df_test = df_all.iloc[n_train:, :]
    
    # Scale the features
    scaler = StandardScaler()
    df_train_scaled = scaler.fit_transform(df_train)
    df_test_scaled = scaler.transform(df_test)
    
    # Train model on scaled data
    lr_model = LinearRegression()
    lr_model.fit(df_train_scaled, df_price)
    
    # Save model, scaler, and feature names
    joblib.dump(lr_model, 'house_price_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(df_train.columns.tolist(), 'feature_columns.pkl')
    
    print("Model trained and saved successfully!")
    print(f"Number of features: {df_train.shape[1]}")
    
    return lr_model, scaler, df_train.columns.tolist()

if __name__ == "__main__":
    train_and_save_model()
