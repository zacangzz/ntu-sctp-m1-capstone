import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def render(df):
    st.header("Supply (Vacancies) vs. Demand (Applications)")
    st.markdown("Comparing the volume of open roles against seeker interest for the top active sectors.")
    st.divider()
    
    # --- DATA PROCESSING FOR SIMPLIFICATION ---
    if 'category_name' in df.columns:
        # 1. Aggregate data by category
        agg = df.groupby('category_name')[['numberOfVacancies', 'metadata_totalNumberJobApplication']].sum().reset_index()
        
        # 2. Calculate Total Volume to find the "Top 15" most active sectors
        agg['total_volume'] = agg['numberOfVacancies'] + agg['metadata_totalNumberJobApplication']
        
        # 3. Sort by volume and slice top 15 to remove clutter
        top_15 = agg.sort_values('total_volume', ascending=False).head(15)
        
        # 4. Calculate Gaps for Analysis Points
        # Gap > 0: More Apps than Jobs (Candidate Competition / Employer Market)
        # Gap < 0: More Jobs than Apps (Talent Shortage / Candidate Market)
        top_15['gap'] = top_15['metadata_totalNumberJobApplication'] - top_15['numberOfVacancies']
        
        # Identify extremes for the insights panel
        highest_demand_gap = top_15.loc[top_15['gap'].idxmax()] # Hardest to get (Most competition)
        highest_supply_gap = top_15.loc[top_15['gap'].idxmin()] # Hardest to fill (Talent shortage)
        
    else:
        st.warning("Data missing 'category_name'. Cannot render chart.")
        return

    col1, col2 = st.columns([3, 1])
    
    with col1:
        # --- DUAL AXIS CHART ---
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        # Bar Plot for Supply (Vacancies)
        sns.barplot(
            data=top_15, 
            x='category_name', 
            y='numberOfVacancies', 
            color='skyblue', 
            alpha=0.7, 
            ax=ax1,
            label='Vacancies (Supply)'
        )
        ax1.set_ylabel('Vacancies (Blue Bars)', color='tab:blue', fontsize=10)
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax1.set_xlabel("")
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
        
        # Line Plot for Demand (Applications)
        ax2 = ax1.twinx()
        sns.lineplot(
            data=top_15, 
            x='category_name', 
            y='metadata_totalNumberJobApplication', 
            color='red', 
            marker='o', 
            linewidth=2, 
            ax=ax2,
            label='Applications (Demand)'
        )
        ax2.set_ylabel('Applications (Red Line)', color='tab:red', fontsize=10)
        ax2.tick_params(axis='y', labelcolor='tab:red')
        
        # Grid and Title
        ax1.grid(axis='y', linestyle='--', alpha=0.5)
        ax1.set_title("Top 15 Sectors: Supply vs. Demand", fontsize=12)
        
        st.pyplot(fig)
    
    with col2:
        st.subheader("Analysis Points")
        
        # Dynamic Insight 1: Talent Shortage (High Supply, Low Demand)
        # Check if the gap is actually negative (more jobs than apps)
        if highest_supply_gap['gap'] < 0:
            st.markdown(f"**📉 Talent Shortage in '{highest_supply_gap['category_name']}'**")
            st.caption(f"There are {int(abs(highest_supply_gap['gap'])):,} more open jobs than applicants. This is a **Candidate's Market**.")
        else:
             st.markdown(f"**📉 Lowest Competition in '{highest_supply_gap['category_name']}'**")
             st.caption("This sector has the fewest applicants per role relative to others.")

        # Dynamic Insight 2: High Competition (High Demand, Low Supply)
        st.markdown(f"**🔥 High Competition in '{highest_demand_gap['category_name']}'**")
        st.caption(f"Applications exceed vacancies by {int(highest_demand_gap['gap']):,}. This is an **Employer's Market**.")
                
        # General Observation
        st.markdown("**💡 Strategic Note**")
        st.caption("Large gaps between the Red Line (Interest) and Blue Bars (Openings) indicate market inefficiency. Focus recruiting efforts where the Blue Bars are significantly higher.")