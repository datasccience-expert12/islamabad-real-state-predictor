import streamlit as st
import pandas as pd
import joblib

# --- 1. SETUP PAGE CONFIGURATION ---
st.set_page_config(page_title="Islamabad House Predictor", page_icon="🏠")

# --- 2. LOAD THE SAVED MODEL ---
# We use @st.cache_resource so it doesn't reload the model on every click (Faster)
@st.cache_resource
def load_model():
    model = joblib.load('house_price_model.pkl')
    model_columns = joblib.load('model_columns.pkl')
    return model, model_columns

model, model_columns = load_model()

# --- 3. THE SIDEBAR (User Inputs) ---
st.sidebar.header("🏡 Enter House Details")

# Sector Selection (Must match the columns you trained on)
sector = st.sidebar.selectbox(
    "Select Sector",
    ["F-10", "DHA Phase 2", "Bahria Town", "G-13"]
)

# Size Input
size_marla = st.sidebar.slider("Size (Marla)", min_value=3, max_value=40, value=10)

# Bedrooms & Bathrooms
bedrooms = st.sidebar.number_input("Bedrooms", min_value=1, max_value=10, value=3)
bathrooms = st.sidebar.number_input("Bathrooms", min_value=1, max_value=10, value=3)

# --- 4. MAIN PAGE DESIGN ---
st.title("🇵🇰 Islamabad Real Estate Estimator")
st.markdown("""
This AI model predicts fair market prices for houses in Islamabad 
based on **Location, Size, and Condition**.
""")

st.write("---") # A horizontal line

# --- 5. PREDICTION LOGIC ---
if st.button("Predict Price 🚀"):
    # Convert Marla to Sqft (Because your model learned on Sqft)
    area_sqft = size_marla * 225
    
    # Create the input data dictionary
    input_data = {
        'bedrooms': [bedrooms],
        'bathrooms': [bathrooms],
        'area_sqft': [area_sqft],
        'location_Bahria Town': [0],
        'location_DHA Phase 2': [0],
        'location_F-10': [0],
        'location_G-13': [0]
    }
    
    # Set the chosen sector to 1
    sector_col = f"location_{sector}"
    if sector_col in input_data:
        input_data[sector_col] = [1]
        
    # Convert to DataFrame and align columns
    query_df = pd.DataFrame(input_data)
    query_df = query_df.reindex(columns=model_columns, fill_value=0)
    
    # Get Prediction
    prediction = model.predict(query_df)
    
    # Display Result
    st.success(f"💰 Estimated Price: PKR {prediction[0]:,.0f}")
    st.info(f"📍 Location: {sector} | 📏 Size: {size_marla} Marla")

# --- 6. FOOTER ---
st.write("---")
st.caption("Built with Python, Scikit-Learn & Streamlit for proptech companies by Engr.jASIM")