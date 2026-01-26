import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import utils

st.title("👁️ Engagement Metrics")
st.subheader("Slide 4: Views vs. Applications (Aggregated)")

df = utils.load_data()

if not df.empty:
    # --- DATA AGGREGATION ---
    # Group by Position Level to find averages
    agg_df = df.groupby('positionLevels', observed=True).agg({
        'metadata_totalNumberOfView': 'mean',
        'metadata_totalNumberJobApplication': 'mean',
        'title': 'count'  # We count jobs to size the bubbles
    }).reset_index()

    # Rename for clarity
    agg_df.columns = ['Position Level', 'Avg Views', 'Avg Applications', 'Job Count']

    # --- PLOTTING ---
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create a Bubble Chart (Scatter with 'size' parameter)
    sns.scatterplot(
        data=agg_df,
        x='Avg Views',
        y='Avg Applications',
        hue='Position Level',
        size='Job Count',     # Bubble size based on number of jobs
        sizes=(200, 2000),    # Range of bubble sizes
        palette='viridis',
        alpha=0.7,
        ax=ax
    )

    # Add text labels to each bubble for clarity
    for i in range(agg_df.shape[0]):
        ax.text(
            agg_df['Avg Views'][i], 
            agg_df['Avg Applications'][i] + 0.5, # Offset text slightly up
            agg_df['Position Level'][i], 
            horizontalalignment='center', 
            size='medium', 
            color='black', 
            weight='semibold'
        )

    ax.set_title("Average Engagement Performance by Seniority")
    ax.set_xlabel("Average Views per Job")
    ax.set_ylabel("Average Applications per Job")
    
    # Move legend outside
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1), title="Seniority")
    
    # Add a grid for easier reading
    ax.grid(True, linestyle='--', alpha=0.5)
    sns.despine()

    st.pyplot(fig)

    st.info("""
    **How to read this chart:**
    * **X-Axis (Right):** Jobs that get more visibility (Views).
    * **Y-Axis (Up):** Jobs that get more actual applicants.
    * **Bubble Size:** Represents the total volume of jobs in the market for that level.
    """)