import streamlit as st
import pandas as pd
import json

@st.cache_data
def load_data():
    """
    Loads the data from the compressed CSV.
    Use caching, this runs only ONCE.
    Subsequent calls return the data from memory instantly.
    """
    try:
        # Load compressed XZ file directly
        df = pd.read_csv('data/SGJobData.csv.xz', compression='xz')

        # Apply the function to the 'categories' column to create 'category_name'
        df['category_name'] = df['categories'].apply(extract_category_name)

        # --- CLEANING STEP ---
        # Keep rows where salary is greater than 0 AND less than or equal to 40,000
        df = df[(df['average_salary'] > 0) & (df['average_salary'] <= 40000)]

        return df
    
    except FileNotFoundError:
        st.error("File not found! Please ensure 'data/SGJobData.csv.xz' exists.")
        return pd.DataFrame()
    
# Define a function to extract the category name from the JSON string
def extract_category_name(category_json):
    if pd.isna(category_json):
        return None
    try:
        # Assuming category_json is a string representation of a JSON array
        categories_list = json.loads(category_json)
        if categories_list and isinstance(categories_list, list) and len(categories_list) > 0:
            # Assuming the first item in the list contains the 'category' field
            if 'category' in categories_list[0]:
                return categories_list[0]['category']
        return None
    except json.JSONDecodeError:
        # Handle cases where the string is not a valid JSON
        return None
    except Exception as e:
        # Catch any other unexpected errors during processing
        return None

