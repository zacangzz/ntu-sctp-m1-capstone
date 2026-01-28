import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

def render(df):
    st.header("Volatility: Fresh vs. Stagnant Listings")
    st.divider()
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if 'is_repost' in df.columns:
            repost_counts = df['is_repost'].value_counts()
            
            # Using Matplotlib Pie (Seaborn doesn't have a native Pie chart)
            fig, ax = plt.subplots(figsize=(8, 8))
            colors = sns.color_palette('pastel')[0:2]
            
            ax.pie(
                repost_counts, 
                labels=repost_counts.index, 
                autopct='%1.1f%%', 
                startangle=90, 
                colors=colors,
                explode=(0.05, 0) # Explode the first slice
            )
            ax.set_title("Distribution of Fresh vs. Reposted Jobs")
            st.pyplot(fig)
        else:
            st.warning("Repost data not available.")
            
    with col2:
        st.subheader("Analysis Points")
        st.markdown("• **Churn:** A large slice for 'Reposts' suggests difficulty in filling roles.")
        st.markdown("• **Stagnation:** Persistent reposts often indicate a disconnect between employer expectations and market reality.")