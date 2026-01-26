import streamlit as st
import utils

st.title("🏁 Conclusion")
st.subheader("Slide 5: Key Takeaways")

df = utils.load_data()

if not df.empty:
    # Calculate most common position
    top_pos = df['positionLevels'].mode()[0]
    avg_sal_senior_mgmt = df[df['positionLevels'] == 'Senior Management']['average_salary'].mean()
    
    st.markdown(f"""
    **Market Insights:**
    1.  **Hiring Volume:** The market is currently dominated by **{top_pos}** roles.
    2.  **Executive Pay:** The average salary for Senior Management-level positions is **${avg_sal_senior_mgmt:,.0f}**.
    3.  **Engagement:** Junior roles tend to receive higher application volumes per view compared to senior roles (Slide 4).
    """)