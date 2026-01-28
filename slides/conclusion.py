import streamlit as st

def render(df):
    st.title("Strategic Recommendations")
    st.markdown("### Executive Summary")
    st.markdown("""
    Our analysis reveals a polarized market: **Senior & Flexible roles** are over-subscribed, 
    while **Operational sectors (F&B, Retail)** face a critical talent shortage.
    """)
    
    st.divider()
    
    # Create 3 columns for the main strategies
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.error("🚨 CRITICAL: The Supply Gap")
        st.markdown("**Observation:**")
        st.caption("F&B and Sales/Retail have massive vacancy volumes (Blue Bars) but low applicant interest (Red Line).")
        
        st.markdown("**Strategic Pivot:**")
        st.write("Stop relying on organic traffic. These roles need **Salary Benchmarking** or **Sign-on Bonuses** to compete.")
        
    with c2:
        st.success("💎 OPPORTUNITY: The 'Flex' Factor")
        st.markdown("**Observation:**")
        st.caption("Roles mentioning 'Part-Time' or 'WFH' dominate the Top 10 list, rivaling 'General Manager' in popularity.")
        
        st.markdown("**Strategic Pivot:**")
        st.write("Reword hard-to-fill Job Descriptions. Swap rigid language for **flexible keywords** (e.g., '4-hour shifts', 'Hybrid').")
        
    with c3:
        st.info("⚙️ OPTIMIZATION: Funnel Segmentation")
        st.markdown("**Observation:**")
        st.caption("Senior Management roles perform best per-post. Entry-level posts are diluted and low-performing.")
        
        st.markdown("**Strategic Pivot:**")
        st.write("**Senior Roles:** Implement AI screening tools to handle volume.\n\n**Entry Roles:** Increase ad spend to boost visibility.")

    st.divider()

    # Final "Next Steps" metrics
    st.subheader("Immediate Next Steps")
    
    m1, m2, m3 = st.columns(3)
    m1.metric(label="1. Audit F&B Descriptions", value="High Priority", delta="-Urgent")
    m2.metric(label="2. Implement WFH Tags", value="Medium Priority", delta="Quick Win")
    m3.metric(label="3. Sponsor Entry-Level Ads", value="Low Priority", delta="Long-term")