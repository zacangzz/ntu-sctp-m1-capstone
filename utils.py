import streamlit as st
import pandas as pd
import json

@st.cache_data
def load_data(filepath="data/SGJobData.csv.xz"):
    """
    Loads the data from the compressed CSV.
    Use caching, this runs only ONCE.
    Subsequent calls return the data from memory instantly.
    """
    try:
        # Load csv file
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        return pd.DataFrame()

    # Apply the function to the 'categories' column to create 'category_name'
    df['category_name'] = df['categories'].apply(extract_category_name)

    # --- CLEANING STEP ---
    # Keep rows where salary is greater than 0 AND less than or equal to 40,000
    df = df[(df['average_salary'] > 0) & (df['average_salary'] <= 40000)]

    # Add Competition Metrics
    df['competition_ratio'] = df.apply(
        lambda x: x['metadata_totalNumberJobApplication'] / x['numberOfVacancies'] 
        if x['numberOfVacancies'] > 0 else 0, axis=1
    )

    # Add Conversion Metrics
    df['conversion_rate'] = df.apply(
        lambda x: (x['metadata_totalNumberJobApplication'] / x['metadata_totalNumberOfView']) * 100 
        if x['metadata_totalNumberOfView'] > 0 else 0, axis=1
    )

    return df
    
    
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

