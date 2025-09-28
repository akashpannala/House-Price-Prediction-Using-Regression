# prediction_interface.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from model_training import fill_none_categories, correct_encoding

class HousePricePredictor:
    def __init__(self):
        self.model = None
        self.feature_columns = None
        self.load_model()
    
    def load_model(self):
        try:
            self.model = joblib.load('house_price_model.pkl')
            self.feature_columns = joblib.load('feature_columns.pkl')
            st.success("Model loaded successfully!")
        except FileNotFoundError:
            st.error("Model not found. Please run model_training.py first.")
            st.stop()
    
    def create_input_features(self):
        """Create a dictionary with all possible features set to 0"""
        return {col: 0 for col in self.feature_columns}
    
    def preprocess_user_input(self, user_input):
        """Preprocess user input to match training data format"""
        # Convert to DataFrame
        input_df = pd.DataFrame([user_input])
        
        # Apply the same preprocessing as training
        input_df = fill_none_categories(input_df)
        input_df = correct_encoding(input_df)
        input_df = input_df.fillna(0)
        
        # Ensure all training columns exist
        for col in self.feature_columns:
            if col not in input_df.columns:
                input_df[col] = 0
        
        # Reorder columns to match training
        input_df = input_df[self.feature_columns]
        
        return input_df

def main():
    st.set_page_config(page_title="House Price Predictor", layout="wide")
    st.title("🏠 House Price Prediction Interface")
    st.markdown("Enter the property details below to get a price prediction.")
    
    # Initialize predictor
    predictor = HousePricePredictor()
    
    # Create sidebar for navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Choose Mode", 
                                   ["Quick Prediction", "Detailed Prediction"])
    
    if app_mode == "Quick Prediction":
        quick_prediction(predictor)
    else:
        detailed_prediction(predictor)

def quick_prediction(predictor):
    st.header("Quick Prediction")
    st.markdown("Enter basic property information for a quick estimate.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        overall_qual = st.slider("Overall Quality (1-10)", 1, 10, 5)
        gr_liv_area = st.number_input("Living Area (sq ft)", 0, 10000, 1500)
        total_bsmt_sf = st.number_input("Basement Area (sq ft)", 0, 5000, 0)
    
    with col2:
        year_built = st.number_input("Year Built", 1800, 2023, 2000)
        year_remod_add = st.number_input("Remodel Year", 1800, 2023, 2000)
        garage_cars = st.slider("Garage Cars Capacity", 0, 5, 2)
    
    with col3:
        full_bath = st.slider("Full Bathrooms", 0, 5, 2)
        bedroom_abv_gr = st.slider("Bedrooms", 0, 10, 3)
        tot_rms_abv_grd = st.slider("Total Rooms", 0, 15, 6)
    
    # Create base input
    user_input = predictor.create_input_features()
    
    # Update with user values
    user_input['OverallQual'] = overall_qual
    user_input['GrLivArea'] = gr_liv_area
    user_input['TotalBsmtSF'] = total_bsmt_sf
    user_input['YearBuilt'] = year_built
    user_input['YearRemodAdd'] = year_remod_add
    user_input['GarageCars'] = garage_cars
    user_input['FullBath'] = full_bath
    user_input['BedroomAbvGr'] = bedroom_abv_gr
    user_input['TotRmsAbvGrd'] = tot_rms_abv_grd
    
    if st.button("Predict Price", type="primary"):
        try:
            processed_input = predictor.preprocess_user_input(user_input)
            prediction = predictor.model.predict(processed_input)[0]
            
            st.success(f"### Predicted House Price: **${prediction:,.2f}**")
            
            # Display feature importance
            show_feature_importance(predictor, processed_input)
            
        except Exception as e:
            st.error(f"Error making prediction: {e}")

def detailed_prediction(predictor):
    st.header("Detailed Prediction")
    st.markdown("Provide detailed information for more accurate prediction.")
    
    # Create tabs for different feature categories
    tab1, tab2, tab3, tab4 = st.tabs(["Property Details", "Interior Features", 
                                     "Exterior Features", "Location & Garage"])
    
    user_input = predictor.create_input_features()
    
    with tab1:
        st.subheader("Basic Property Information")
        col1, col2 = st.columns(2)
        
        with col1:
            user_input['LotArea'] = st.number_input("Lot Area (sq ft)", 0, 100000, 10000)
            user_input['LotFrontage'] = st.number_input("Lot Frontage (ft)", 0, 500, 100)
            user_input['OverallQual'] = st.slider("Overall Quality", 1, 10, 5)
            user_input['OverallCond'] = st.slider("Overall Condition", 1, 10, 5)
        
        with col2:
            user_input['YearBuilt'] = st.number_input("Year Built", 1800, 2023, 2000)
            user_input['YearRemodAdd'] = st.number_input("Remodel Year", 1800, 2023, 2000)
            user_input['MasVnrArea'] = st.number_input("Masonry Veneer Area", 0, 5000, 0)
    
    with tab2:
        st.subheader("Interior Features")
        col1, col2 = st.columns(2)
        
        with col1:
            user_input['GrLivArea'] = st.number_input("Above Grade Living Area", 0, 10000, 1500)
            user_input['TotalBsmtSF'] = st.number_input("Total Basement Area", 0, 5000, 0)
            user_input['1stFlrSF'] = st.number_input("1st Floor Area", 0, 5000, 1000)
            user_input['2ndFlrSF'] = st.number_input("2nd Floor Area", 0, 5000, 500)
        
        with col2:
            user_input['FullBath'] = st.slider("Full Bathrooms", 0, 5, 2)
            user_input['HalfBath'] = st.slider("Half Bathrooms", 0, 3, 1)
            user_input['BedroomAbvGr'] = st.slider("Bedrooms", 0, 10, 3)
            user_input['KitchenAbvGr'] = st.slider("Kitchens", 0, 5, 1)
            user_input['TotRmsAbvGrd'] = st.slider("Total Rooms", 0, 15, 6)
    
    with tab3:
        st.subheader("Exterior Features")
        col1, col2 = st.columns(2)
        
        with col1:
            # Exterior quality
            exter_qual = st.selectbox("Exterior Quality", 
                                    ["None", "Po", "Fa", "TA", "Gd", "Ex"])
            quality_map = {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
            user_input['ExterQual_Encoded'] = quality_map[exter_qual]
            
            # House style
            house_style = st.selectbox("House Style", [
                "1Story", "1.5Fin", "1.5Unf", "2Story", "2.5Fin", 
                "2.5Unf", "SFoyer", "SLvl"
            ])
            user_input[f'HouseStyle_{house_style}'] = 1
        
        with col2:
            # Roof style
            roof_style = st.selectbox("Roof Style", [
                "Flat", "Gable", "Gambrel", "Hip", "Mansard", "Shed"
            ])
            user_input[f'RoofStyle_{roof_style}'] = 1
            
            # Foundation type
            foundation = st.selectbox("Foundation Type", [
                "BrkTil", "CBlock", "PConc", "Slab", "Stone", "Wood"
            ])
            user_input[f'Foundation_{foundation}'] = 1
    
    with tab4:
        st.subheader("Garage & Location")
        col1, col2 = st.columns(2)
        
        with col1:
            user_input['GarageCars'] = st.slider("Garage Car Capacity", 0, 5, 2)
            user_input['GarageArea'] = st.number_input("Garage Area (sq ft)", 0, 2000, 500)
            garage_qual = st.selectbox("Garage Quality", 
                                     ["None", "Po", "Fa", "TA", "Gd", "Ex"])
            user_input['GarageQual_Encoded'] = quality_map[garage_qual]
        
        with col2:
            # Neighborhood (simplified)
            neighborhood = st.selectbox("Neighborhood", [
                "Blmngtn", "Blueste", "BrDale", "BrkSide", "ClearCr", "CollgCr",
                "Crawfor", "Edwards", "Gilbert", "IDOTRR", "MeadowV", "Mitchel",
                "NAmes", "NPkVill", "NWAmes", "NoRidge", "NridgHt", "OldTown",
                "SWISU", "Sawyer", "SawyerW", "Somerst", "StoneBr", "Timber", "Veenker"
            ])
            user_input[f'Neighborhood_{neighborhood}'] = 1
    
    if st.button("Predict Detailed Price", type="primary"):
        try:
            processed_input = predictor.preprocess_user_input(user_input)
            prediction = predictor.model.predict(processed_input)[0]
            
            st.success(f"### Predicted House Price: **${prediction:,.2f}**")
            
            # Show additional information
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Predicted Price", f"${prediction:,.0f}")
            with col2:
                st.metric("Price per Sq Ft", f"${prediction/user_input['GrLivArea']:,.0f}" 
                         if user_input['GrLivArea'] > 0 else "N/A")
            with col3:
                st.metric("Property Age", f"{2023 - user_input['YearBuilt']} years")
            
        except Exception as e:
            st.error(f"Error making prediction: {e}")

def show_feature_importance(predictor, processed_input):
    """Show which features most influenced the prediction"""
    st.subheader("Prediction Insights")
    
    # Get coefficients and feature names
    coefficients = predictor.model.coef_
    feature_names = processed_input.columns
    
    # Create importance dataframe
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'coefficient': coefficients,
        'value': processed_input.iloc[0].values
    })
    
    # Calculate contribution
    importance_df['contribution'] = importance_df['coefficient'] * importance_df['value']
    importance_df = importance_df.sort_values('contribution', key=abs, ascending=False)
    
    # Show top 5 positive and negative contributors
    top_positive = importance_df.nlargest(5, 'contribution')
    top_negative = importance_df.nsmallest(5, 'contribution')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Top Positive Influences:**")
        for _, row in top_positive.iterrows():
            if abs(row['contribution']) > 0:
                st.write(f"{row['feature']}: +${row['contribution']:,.0f}")
    
    with col2:
        st.write("**Top Negative Influences:**")
        for _, row in top_negative.iterrows():
            if abs(row['contribution']) > 0:
                st.write(f"{row['feature']}: ${row['contribution']:,.0f}")

if __name__ == "__main__":
    main()