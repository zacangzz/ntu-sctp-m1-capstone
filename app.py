import streamlit as st
import utils
from slides import overview, supply_demand, competition, engagement, volatility, conclusion

# --- Configuration Constants ---
# Dictionary mapping display names to module references
SLIDES = {
    "1. Overview": overview,
    "2. Supply vs. Demand": supply_demand,
    "3. Competition Analysis": competition,
    "4. Engagement Funnel": engagement,
    "5. Conclusion": conclusion
}

# Page Config
st.set_page_config(page_title="SG Job Market Insights", layout="wide")

# --- Data Loading ---
@st.cache_data
def get_data():
    return utils.load_data()

try:
    df = get_data()
    if df.empty:
        st.error("Data could not be loaded. Please check 'data/SGJobData.csv.xz'.")
        st.stop()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", list(SLIDES.keys()))

# --- Main Rendering Logic ---
# This replaces the long if/elif chain
if selection in SLIDES:
    SLIDES[selection].render(df)