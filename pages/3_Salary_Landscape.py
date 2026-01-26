import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import utils

st.title("💰 Salary Landscape")
st.subheader("Slide 3: Salary by Position Level")

df = utils.load_data()

if not df.empty:
    fig, ax = plt.subplots(figsize=(14, 6))

    # Boxplot: Salary distribution
    # FIX: Assigned x variable ('positionLevels') to hue, and set legend=False
    sns.boxplot(data=df, 
                x='positionLevels', 
                y='average_salary', 
                hue='positionLevels', 
                legend=False,         
                palette='coolwarm', 
                ax=ax)
    
    ax.set_title("Monthly Salary Distribution by Seniority")
    ax.set_xlabel("Position Level")
    ax.set_ylabel("Average Salary (SGD)")
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.tick_params(axis='x', rotation=45)

    
    sns.despine()
    st.pyplot(fig)
