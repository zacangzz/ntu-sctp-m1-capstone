import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

def render(df):
    st.title("🇸🇬 Job Market Health")
    st.markdown("A high-level snapshot of current market conditions.")
    
    # Top Level Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Vacancies", f"{df['numberOfVacancies'].sum():,}")
    m2.metric("Total Applications", f"{df['metadata_totalNumberJobApplication'].sum():,}")
    m3.metric("Avg. Market Salary", f"${df['average_salary'].mean():,.0f}")
    
    repost_rate = df[df['metadata_repostCount'] > 0].shape[0] / len(df)
    m4.metric("Avg. Repost Rate", f"{repost_rate:.1%}")
    
    st.divider()
    
    st.subheader("Top 15 Sectors by Vacancy Volume")
    
    # Ensure necessary columns exist before plotting
    if 'category_name' in df.columns and 'numberOfVacancies' in df.columns:
        # 1. Group by category and sum vacancies
        sector_dist = df.groupby('category_name')['numberOfVacancies'].sum().reset_index()
        
        # 2. Sort descending and take top 15
        top_15_sectors = sector_dist.sort_values('numberOfVacancies', ascending=False).head(15)
        
        # Seaborn Plot
        fig, ax = plt.subplots(figsize=(10, 7))
        
        # Switch to horizontal bar plot for better readability of 15 items
        sns.barplot(
            data=top_15_sectors, 
            y='category_name',      
            x='numberOfVacancies',
            hue='category_name',    # Assign y variable to hue
            legend=False,           # Disable the legend
            palette='viridis', 
            ax=ax,
            orient='h'
        )
        ax.set_xlabel("Number of Vacancies")
        ax.set_ylabel("") # Clear the y-label as category names are self-explanatory
        
        st.pyplot(fig)
    else:
        st.warning("Insufficient data to display sector distribution.")