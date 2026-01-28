import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

def render(df):
    st.header("Competition Index: The 'Hottest' Roles")
    st.divider()
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Top 10 most competitive roles
        top_roles = df.groupby('title')['competition_ratio'].mean().nlargest(10).reset_index()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.barplot(
            data=top_roles, 
            y='title', 
            x='competition_ratio', 
            palette='magma', 
            orient='h',
            ax=ax
        )
        ax.set_title("Top 10 Roles by Applicants per Vacancy")
        ax.set_xlabel("Average Applicants per Role")
        ax.set_ylabel("")
        
        st.pyplot(fig)
        
    with col2:
        st.subheader("Analysis Points")
        st.markdown("**Salary Sensitivity:**")
        st.caption("High advertised salaries (e.g., General Manager >$16k) trigger the highest competition (>850 applicants/role).")
        
        st.markdown("---")

        st.markdown("**Flexibility is King:**")
        st.caption("Keywords like 'Work From Home' and 'Part-Time' dominate the top list, indicating lifestyle benefits drive volume as much as pay.")
        
        st.markdown("---")

        st.markdown("**Screening Bottleneck:**")
        st.caption("With 500+ applicants for Admin/Data Entry roles, manual review is impossible. Automated CV parsing is critical here.")