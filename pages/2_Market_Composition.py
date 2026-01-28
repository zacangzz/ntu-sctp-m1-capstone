import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import utils

st.title("🏢 Market Composition")
st.subheader("Slide 2: Job Position Breakdown")

df = utils.load_data()

if not df.empty:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # --- CHART 1: Top 10 Jobs by Category ---
    # 1. Identify the Top 10 Categories
    top_10_categories = df['category_name'].value_counts().nlargest(10).index
    
    # 2. Filter data for the plot  
    df_top_10 = df[df['category_name'].isin(top_10_categories)]

    sns.countplot(data=df_top_10, 
                  y='category_name', 
                  hue='category_name',
                  legend=False,
                  order=top_10_categories, # Force order to Top 10
                  palette='viridis', 
                  ax=ax1)
    
    ax1.set_title("Top 10 Industries by Job Volume")
    ax1.set_xlabel("Job Count")
    ax1.set_ylabel("Industry")

    # --- CHART 2: Position Levels ---
    # (Usually fewer than 10, so we display all, or you can apply similar logic)
    sns.countplot(data=df, 
                  x='positionLevels', 
                  hue='positionLevels',
                  legend=False,
                  palette='magma', 
                  ax=ax2)
    
    ax2.set_title("Distribution of Position Levels")
    ax2.set_xlabel("Position Level")
    ax2.tick_params(axis='x', rotation=45)
    
    sns.despine()
    st.pyplot(fig)