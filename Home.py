import streamlit as st
import utils 


st.set_page_config(page_title="SG Job Market Dashboard", layout="wide")

st.title("🇸🇬 SG Job Market Overview")

# Load data
df = utils.load_data()

st.markdown(f"""
### Dataset Overview
We are analyzing **{len(df):,} job postings**.

**Key Metrics Tracked:**
* **Categories:** {df['category_name'].nunique()} unique industries
* **Salary Range:** ${df['average_salary'].min():,} - ${df['average_salary'].max():,}
* **Engagement:** Tracking views vs. applications
""")

st.info("👈 Select a slide from the sidebar to view detailed analytics.")