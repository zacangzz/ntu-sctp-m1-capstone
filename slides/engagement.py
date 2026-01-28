import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def render(df):
    st.header("Engagement Analysis by Position Level")
    st.markdown("Evaluating how different seniority levels convert viewer attention into actual applications.")
    st.divider()
    
    # --- DATA PROCESSING ---
    if 'positionLevels' in df.columns:
        # Explode list-like columns if necessary (handling potential string representations of lists)
        # If your data is already clean strings, this line just passes through.
        df_exploded = df.explode('positionLevels')
        
        # Aggregate metrics
        agg = df_exploded.groupby('positionLevels').agg({
            'metadata_totalNumberOfView': 'mean',
            'metadata_totalNumberJobApplication': 'mean',
            'numberOfVacancies': 'sum'
        }).reset_index()
        
        # Calculate conversion rate
        agg['conversion_rate'] = (agg['metadata_totalNumberJobApplication'] / agg['metadata_totalNumberOfView']) * 100
        
        # Filter out rows with zero views to avoid errors
        agg = agg[agg['metadata_totalNumberOfView'] > 0]
        
    else:
        st.warning("Column 'positionLevels' not found in dataset.")
        return

    col1, col2 = st.columns([3, 1])
    
    with col1:
        # --- REVISED CHART ---
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Scatter plot
        sns.scatterplot(
            data=agg, 
            x='metadata_totalNumberOfView', 
            y='metadata_totalNumberJobApplication', 
            size='numberOfVacancies',
            hue='conversion_rate',
            palette='coolwarm',
            sizes=(50, 400), # Reduced size for clarity
            alpha=0.9,
            ax=ax
        )
        
        # 15% Efficiency Benchmark Line
        max_view = agg['metadata_totalNumberOfView'].max() * 1.1
        max_app = agg['metadata_totalNumberJobApplication'].max() * 1.1
        
        # Draw line (y = 0.15 * x)
        x_vals = np.linspace(0, max_view, 100)
        y_vals = x_vals * 0.15
        ax.plot(x_vals, y_vals, color='gray', linestyle='--', alpha=0.5, label='15% Conversion Target')
        
        # Label Annotations (with slight offset)
        for i in range(agg.shape[0]):
            row = agg.iloc[i]
            ax.text(
                row['metadata_totalNumberOfView'] + 1, 
                row['metadata_totalNumberJobApplication'], 
                row['positionLevels'], 
                fontsize=9, 
                weight='semibold',
                alpha=0.8
            )

        # Formatting
        ax.set_title("Engagement Efficiency by Seniority")
        ax.set_xlabel("Avg. Views per Posting")
        ax.set_ylabel("Avg. Applications per Posting")
        ax.set_xlim(left=0, right=max_view)
        ax.set_ylim(bottom=0, top=max_app)
        
        # Fixed Legend Position
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, title="Conversion %")
        plt.tight_layout() # Ensures legend fits in the image
        
        st.pyplot(fig)
    
    with col2:
        st.subheader("Analysis Points")
        
        # Identify key data points for dynamic text
        senior_role = agg.loc[agg['positionLevels'] == 'Senior Management'].iloc[0] if 'Senior Management' in agg['positionLevels'].values else agg.iloc[0]
        junior_role = agg.loc[agg['positionLevels'].str.contains('Junior|Fresh', case=False, na=False)].sort_values('metadata_totalNumberOfView').iloc[0] if any(agg['positionLevels'].str.contains('Junior|Fresh')) else agg.iloc[-1]

        st.markdown(f"**🚀 Executive Power:**")
        st.caption(f"**Senior Management** roles are outliers, generating the highest views ({senior_role['metadata_totalNumberOfView']:.0f}) and applications ({senior_role['metadata_totalNumberJobApplication']:.0f}) per post. High visibility correlates with high intent here.")
        
        st.markdown("---")
        
        st.markdown("**📉 The 'Junior' Squeeze:**")
        st.caption("Entry-level and Junior roles cluster at the bottom-left. Despite the large workforce size, individual postings receive significantly less engagement, suggesting market saturation.")
        
        st.markdown("---")
        
        st.markdown("**🎯 Efficiency Gap:**")
        st.caption("Most roles fall below the dashed 15% line. 'Middle Management' is the closest to bridging this gap, indicating a balanced supply/demand ratio.")