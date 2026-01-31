import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import os
from statsmodels.tsa.arima.model import ARIMA
from datetime import timedelta

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Workforce Intelligence Portal", layout="wide", initial_sidebar_state="collapsed")

# --- CUSTOM CSS ---
# Optimizes metric display for readability and handles long text strings
st.markdown("""
    <style>
    .metric-container {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        height: 140px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .metric-label {
        font-size: 15px;
        color: #5f6368;
        margin-bottom: 8px;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-value {
        font-size: 24px;
        font-weight: 700;
        color: #1a73e8;
        word-wrap: break-word;
        line-height: 1.2;
    }
    </style>
    """, unsafe_allow_html=True)

# --- DATA LOADING & PROCESSING ---
@st.cache_data
def load_data(file_path):
    """
    Robust data loader for SGJobData.csv.xz with error handling and caching.
    """
    if not os.path.exists(file_path):
        # Fallback for demonstration if specific file path varies
        if os.path.exists('SGJobsample_100000.csv.xz'):
            file_path = 'SGJobsample_100000.csv.xz'
        else:
            st.error(f"⚠️ Critical Error: Data file not found at `{file_path}`. Please ensure the file is in the root directory.")
            return pd.DataFrame()
    
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return pd.DataFrame()

    # 1. Normalize JSON Categories
    def parse_category(x):
        try:
            if pd.isna(x): return []
            # Handle potential single-quote JSON format
            data = json.loads(x.replace("'", '"'))
            return [item['category'] for item in data]
        except: return []

    if 'categories' in df.columns:
        df['category_list'] = df['categories'].apply(parse_category)
    else:
        st.error("Column 'categories' missing from dataset.")
        return pd.DataFrame()

    # 2. Date Parsing
    df['metadata_newPostingDate'] = pd.to_datetime(df['metadata_newPostingDate'], errors='coerce')
    df = df.dropna(subset=['metadata_newPostingDate'])
    df['month_year'] = df['metadata_newPostingDate'].dt.to_period('M').astype(str)

    # 3. Salary Calculation & Cleaning
    # If average_salary doesn't exist, calculate from min/max
    if 'average_salary' not in df.columns:
        df['average_salary'] = (df['salary_minimum'] + df['salary_maximum']) / 2
    
    # Filter Salary Outliers (Keep 1st to 99th percentile)
    q_low = df["average_salary"].quantile(0.01)
    q_hi = df["average_salary"].quantile(0.99)
    df = df[(df["average_salary"] >= q_low) & (df["average_salary"] <= q_hi)]

    # 4. Experience Bracket Categorization
    df['experience_bracket'] = pd.cut(
        df['minimumYearsExperience'], 
        bins=[-1, 2, 5, 10, 100], 
        labels=['0-2 yrs (Entry)', '3-5 yrs (Mid)', '6-10 yrs (Senior)', '10+ yrs (Lead)']
    )
    
    return df

# --- INITIALIZE APP ---
FILE_PATH = 'data/SGJobData.csv.xz'
df_raw = load_data(FILE_PATH)

if df_raw.empty:
    st.stop()

# Prepare exploded dataset for sector-level analysis (one row per category)
df_exploded = df_raw.explode('category_list')

# Calculate Reporting Period
min_date = df_raw['metadata_newPostingDate'].min().strftime('%d %b %Y')
max_date = df_raw['metadata_newPostingDate'].max().strftime('%d %b %Y')

# --- DASHBOARD HEADER ---
st.title("🎓 Workforce Intelligence Portal")
st.markdown(f"**Data Reporting Period:** {min_date} to {max_date}")
st.markdown("Leveraging labour market data to align educational curriculum with structural industry demand.")

# --- TABS LAYOUT ---
tabs = st.tabs([
    "📈 Executive Summary", 
    "📊 Market Demand", 
    "🎓 Graduate Pipeline", 
    "⚙️ Operational Intelligence",
    "🏗️ Structural Analysis",
    "🔮 Hiring Forecast"
])

# ==========================================
# TAB 1: EXECUTIVE SUMMARY
# ==========================================
with tabs[0]:
    st.header("Strategic Outlook")
    
    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    
    # Metrics Calculations
    total_openings = int(df_raw['numberOfVacancies'].sum())
    top_sector = df_exploded['category_list'].mode()[0] if not df_exploded.empty else "N/A"
    median_pay = df_raw['average_salary'].median()
    entry_level_rate = (df_raw['minimumYearsExperience'] <= 2).mean() * 100

    # Custom Metric Cards
    with col1:
        st.markdown(f"""<div class='metric-container'><div class='metric-label'>Total Job Openings</div><div class='metric-value'>{total_openings:,}</div></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class='metric-container'><div class='metric-label'>Top Industry Sector</div><div class='metric-value'>{top_sector}</div></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class='metric-container'><div class='metric-label'>Typical Monthly Salary</div><div class='metric-value'>${median_pay:,.0f}</div></div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class='metric-container'><div class='metric-label'>Entry-Level Roles</div><div class='metric-value'>{entry_level_rate:.1f}%</div></div>""", unsafe_allow_html=True)

    st.subheader("Strategic Recommendations")
    st.info("""
    * **Resource Allocation:** Shift faculty resources toward the "Top Industry Sector" identified above, as it currently anchors the region's hiring volume.
    * **Curriculum Focus:** With {entry_level_rate:.1f}% of roles open to fresh graduates, ensure Foundation/Diploma programs prioritize "Work-Ready" skills to capture this immediate demand.
    """, icon="💡")

# ==========================================
# TAB 2: MARKET DEMAND
# ==========================================
with tabs[1]:
    st.header("Industry Volume & Role Demand")
    c1, c2 = st.columns(2)

    with c1:
        # Chart: Total Job Openings by Industry (Top 10)
        # Aggregating by sector sum of vacancies
        sector_vol = df_exploded.groupby('category_list')['numberOfVacancies'].sum().sort_values(ascending=True).tail(10).reset_index()
        
        fig1 = px.bar(sector_vol, x='numberOfVacancies', y='category_list', orientation='h',
                      title="Top 10 Sectors by Total Job Openings",
                      color='numberOfVacancies', color_continuous_scale='Blues',
                      labels={'numberOfVacancies': 'Total Openings', 'category_list': 'Industry'},
                      template="plotly_white")
        fig1.update_traces(hovertemplate="<b>%{y}</b><br>Openings: %{x:,}")
        fig1.update_layout(showlegend=False)
        st.plotly_chart(fig1, use_container_width=True)
        
        st.markdown("""
        **Key Takeaways:**
        1.  **Volume Drivers:** Identifies the "Big 10" industries that provide the statistical majority of employment opportunities for graduates.
        2.  **Placement Strategy:** Programs aligned with these sectors offer the highest probability of rapid student placement upon graduation.
        """)

    with c2:
        # Chart: Top 10 Most Demanded Job Titles
        # Aggregating by exact job title
        title_vol = df_raw.groupby('title')['numberOfVacancies'].sum().sort_values(ascending=True).tail(10).reset_index()
        
        fig2 = px.bar(title_vol, x='numberOfVacancies', y='title', orientation='h',
                      title="Top 10 Most Demanded Job Titles",
                      color='numberOfVacancies', color_continuous_scale='Greens',
                      labels={'numberOfVacancies': 'Total Openings', 'title': 'Job Title'},
                      template="plotly_white")
        fig2.update_traces(hovertemplate="<b>%{y}</b><br>Openings: %{x:,}")
        fig2.update_layout(showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)
        
        st.markdown("""
        **Key Takeaways:**
        1.  **Curriculum Specificity:** Highlights specific roles (e.g., "Accountant" vs "Finance Manager") to inform module-level learning outcomes.
        2.  **Career Counseling:** Provides career advisors with concrete data on which specific job titles are most abundant in the current market.
        """)

# ==========================================
# TAB 3: GRADUATE PIPELINE
# ==========================================
with tabs[2]:
    st.header("Entry Barriers & ROI Benchmarks")
    c3, c4 = st.columns(2)

    with c3:
        # Chart: Experience Needed by Industry (Top 10 Vol)
        # Filter for top sectors to keep chart readable
        top_sectors_list = df_exploded['category_list'].value_counts().head(10).index.tolist()
        df_top_exp = df_exploded[df_exploded['category_list'].isin(top_sectors_list)]
        
        exp_counts = df_top_exp.groupby(['category_list', 'experience_bracket']).size().reset_index(name='Jobs')
        
        fig3 = px.bar(exp_counts, x='category_list', y='Jobs', color='experience_bracket',
                      title="Experience Requirements (Top 10 Sectors)",
                      labels={'category_list': 'Industry', 'Jobs': 'Job Count', 'experience_bracket': 'Experience'},
                      template="plotly_white", barmode='stack')
        fig3.update_traces(hovertemplate="<b>%{x}</b><br>%{data.name}<br>Count: %{y:,}")
        st.plotly_chart(fig3, use_container_width=True)
        
        st.markdown("""
        **Key Takeaways:**
        1.  **Accessibility Check:** Identifies "Green" sectors with large Entry-Level (0-2 yrs) portions, ideal for fresh graduate pipelines.
        2.  **Upskilling Needs:** Highlights sectors dominated by Senior roles, signaling a need for Executive Education or Advanced Certificate programs.
        """)

    with c4:
        # Chart: Lollipop Chart for Salary Benchmarking (Top 10 Sectors)
        # Calculate median salary per sector
        sal_bench = df_exploded.groupby('category_list')['average_salary'].median().sort_values(ascending=True).tail(15).reset_index()
        
        fig4 = go.Figure()
        # Draw lines
        fig4.add_trace(go.Scatter(
            x=sal_bench['average_salary'],
            y=sal_bench['category_list'],
            mode='markers',
            marker=dict(color='#1a73e8', size=12),
            name='Median Salary'
        ))
        # Draw shapes for sticks
        shapes = []
        for i, row in sal_bench.iterrows():
            shapes.append(dict(
                type="line",
                x0=0, y0=i, x1=row['average_salary'], y1=i,
                line=dict(color="#e0e0e0", width=2)
            ))
        
        fig4.update_layout(
            title="Typical Monthly Salary Benchmark (Top 15 Sectors)",
            shapes=shapes,
            xaxis=dict(title="Median Monthly Salary ($)", tickformat="$,.0f"),
            yaxis=dict(title="Industry"),
            template="plotly_white",
            showlegend=False
        )
        st.plotly_chart(fig4, use_container_width=True)
        
        st.markdown("""
        **Key Takeaways:**
        1.  **ROI Demonstration:** Provides clear evidence of financial returns for students considering high-value specializations.
        2.  **Pricing Strategy:** Helps institutions align tuition fees with the expected earning potential of graduates in specific fields.
        """)

# ==========================================
# TAB 4: OPERATIONAL INTELLIGENCE
# ==========================================
with tabs[3]:
    st.header("Hiring Scale & Competition")
    c5, c6 = st.columns(2)

    with c5:
        # Chart: Bulk Hiring Signal (Treemap)
        # Metric: Vacancies per Job Ad
        bulk_data = df_exploded.groupby('category_list').agg({
            'numberOfVacancies': 'sum',
            'metadata_jobPostId': 'count'
        }).reset_index()
        bulk_data['hiring_scale'] = bulk_data['numberOfVacancies'] / bulk_data['metadata_jobPostId']
        
        fig5 = px.treemap(
            bulk_data, 
            path=['category_list'], 
            values='numberOfVacancies',
            color='hiring_scale',
            color_continuous_scale='RdBu',
            title="Large-Scale Hiring Opportunities (Bulk Hiring Signal)",
            labels={'numberOfVacancies': 'Total Openings', 'hiring_scale': 'Vacancies per Ad'}
        )
        fig5.update_traces(hovertemplate="<b>%{label}</b><br>Total Openings: %{value:,}<br>Hiring Scale: %{color:.1f} jobs/ad")
        st.plotly_chart(fig5, use_container_width=True)
        
        st.markdown("""
        **Key Takeaways:**
        1.  **Partnership Targets:** Dark blue areas indicate sectors where employers hire in "Batches" (high scale), ideal for institutional MOUs.
        2.  **Efficiency:** Prioritizing these sectors allows for placing entire cohorts of students with fewer corporate partners.
        """)

    with c6:
        # Chart: Interest vs. Competition (Scatter)
        # X: Views, Y: Applications
        comp_data = df_raw.copy()
        # Cap outliers for better visualization
        q_view = comp_data['metadata_totalNumberOfView'].quantile(0.95)
        q_app = comp_data['metadata_totalNumberJobApplication'].quantile(0.95)
        comp_data = comp_data[(comp_data['metadata_totalNumberOfView'] < q_view) & (comp_data['metadata_totalNumberJobApplication'] < q_app)]
        
        fig6 = px.scatter(
            comp_data, 
            x='metadata_totalNumberOfView', 
            y='metadata_totalNumberJobApplication',
            color='experience_bracket',
            title="Interest vs. Competition Level",
            labels={'metadata_totalNumberOfView': 'Job Views', 'metadata_totalNumberJobApplication': 'Applications'},
            template="plotly_white",
            opacity=0.6
        )
        fig6.update_layout(legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig6, use_container_width=True)
        
        st.markdown("""
        **Key Takeaways:**
        1.  **Hidden Gems:** Roles with low views but low applications (bottom-left) represent uncrowded entry points for graduates.
        2.  **High Noise:** Roles with high views and high applications (top-right) require students to have standout portfolios to compete.
        """)

# ==========================================
# TAB 5: STRUCTURAL ANALYSIS
# ==========================================
with tabs[4]:
    st.header("Workforce Composition")
    c7, c8 = st.columns(2)

    with c7:
        # Chart: Demand by Career Stage (No Legend)
        seniority_counts = df_raw['positionLevels'].value_counts().reset_index()
        seniority_counts.columns = ['Level', 'Count']
        
        fig7 = px.bar(seniority_counts, x='Count', y='Level', orientation='h',
                      title="Demand by Career Stage",
                      color='Level',
                      template="plotly_white")
        fig7.update_layout(showlegend=False)
        fig7.update_traces(hovertemplate="<b>%{y}</b><br>Count: %{x:,}")
        st.plotly_chart(fig7, use_container_width=True)
        
        st.markdown("""
        **Key Takeaways:**
        1.  **Pipeline Gaps:** Visualizes the "Missing Middle" or leadership gaps in the local economy.
        2.  **Program Calibration:** Ensures the institution isn't over-producing Entry-level candidates for a market that primarily needs Managers.
        """)

    with c8:
        # Chart: Job Contract Breakdown (No Legend)
        emp_counts = df_raw['employmentTypes'].value_counts().reset_index()
        emp_counts.columns = ['Type', 'Total']
        
        fig8 = px.pie(emp_counts, values='Total', names='Type', hole=0.5,
                      title="Job Contract Breakdown",
                      template="plotly_white")
        fig8.update_layout(showlegend=False)
        fig8.update_traces(textinfo='percent+label')
        st.plotly_chart(fig8, use_container_width=True)
        
        st.markdown("""
        **Key Takeaways:**
        1.  **Gig Economy Readiness:** A high percentage of Contract/Temporary roles necessitates teaching freelancing and self-management skills.
        2.  **Stability Indicator:** A dominance of Permanent roles suggests a stable, long-term employment market for graduates.
        """)

# ==========================================
# TAB 6: HIRING DEMAND FORECAST
# ==========================================
with tabs[5]:
    st.header("🔮 Hiring Demand Forecast (Next 6 Months)")
    
    # 1. Data Preparation
    time_series = df_raw.set_index('metadata_newPostingDate').resample('M').size().reset_index()
    time_series.columns = ['Date', 'Volume']
    
    # Check data sufficiency
    if len(time_series) < 6:
        st.warning("⚠️ Insufficient historical data points (minimum 6 months required) to run ARIMA models.")
    else:
        try:
            # 2. Forecasting (ARIMA)
            model = ARIMA(time_series['Volume'], order=(1,1,1))
            model_fit = model.fit()
            
            steps = 6
            forecast_result = model_fit.get_forecast(steps=steps)
            forecast_mean = forecast_result.predicted_mean
            forecast_ci = forecast_result.conf_int(alpha=0.05) # 95% CI
            
            # Create Forecast Dates
            last_date = time_series['Date'].iloc[-1]
            forecast_dates = [last_date + pd.DateOffset(months=x) for x in range(1, steps+1)]
            
            # 3. Visualization: Forecast with Risk Corridor
            fig_f = go.Figure()
            
            # Historical Line
            fig_f.add_trace(go.Scatter(
                x=time_series['Date'], y=time_series['Volume'],
                mode='lines+markers', name='Historical Demand',
                line=dict(color='#1a73e8', width=2)
            ))
            
            # Risk Corridor (Confidence Interval)
            fig_f.add_trace(go.Scatter(
                x=forecast_dates + forecast_dates[::-1],
                y=forecast_ci.iloc[:, 1].tolist() + forecast_ci.iloc[:, 0].tolist()[::-1],
                fill='toself',
                fillcolor='rgba(232, 26, 26, 0.1)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip",
                name='95% Risk Corridor'
            ))
            
            # Forecast Line
            fig_f.add_trace(go.Scatter(
                x=forecast_dates, y=forecast_mean,
                mode='lines+markers', name='Forecasted Volume',
                line=dict(color='#d93025', width=2, dash='dash')
            ))
            
            fig_f.update_layout(
                title="Hiring Demand Forecast with Risk Corridor",
                xaxis_title="Timeline", yaxis_title="Job Postings",
                template="plotly_white",
                legend=dict(orientation="h", y=1.1)
            )
            st.plotly_chart(fig_f, use_container_width=True)
            
            st.markdown("""
            **Key Takeaways:**
            1.  **Risk Assessment:** The "Risk Corridor" (shaded red) shows the uncertainty level. A narrow corridor implies a stable market safe for long-term program investment.
            2.  **Trend Direction:** The dashed line predicts whether the total addressable market for graduates is expanding or contracting in the near term.
            """)

            st.divider()
            
            # 4. Advanced Metrics: Momentum & Velocity
            c_mom, c_vel = st.columns(2)
            
            with c_mom:
                # Chart: Sector Momentum (% Market Share over time)
                # Take top 5 sectors
                top_5 = df_exploded['category_list'].value_counts().head(5).index
                mom_df = df_exploded[df_exploded['category_list'].isin(top_5)]
                # Group by Month and Sector
                mom_pivot = mom_df.groupby(['month_year', 'category_list']).size().reset_index(name='count')
                # Calculate percentage share per month
                mom_pivot['total_month'] = mom_pivot.groupby('month_year')['count'].transform('sum')
                mom_pivot['share'] = (mom_pivot['count'] / mom_pivot['total_month']) * 100
                
                fig_mom = px.area(mom_pivot, x='month_year', y='share', color='category_list',
                                  title="Sector Momentum (% Market Share)",
                                  labels={'share': 'Market Share (%)', 'month_year': 'Month', 'category_list': 'Sector'},
                                  template="plotly_white")
                st.plotly_chart(fig_mom, use_container_width=True)
                
                st.markdown("""
                **Key Takeaways:**
                1.  **Dominance Shifts:** Visualizes if a top sector is losing relevance (shrinking area) to emerging competitors.
                2.  **Resource Planning:** Helps faculty heads predict which departments will need more budget based on share growth.
                """)

            with c_vel:
                # Chart: Job Role Velocity (Growth % of last month vs previous)
                # Get last 2 months
                sorted_months = sorted(df_raw['month_year'].unique())
                if len(sorted_months) >= 2:
                    last_m = sorted_months[-1]
                    prev_m = sorted_months[-2]
                    
                    # Filter data
                    df_vel = df_raw[df_raw['month_year'].isin([last_m, prev_m])]
                    vel_pivot = df_vel.groupby(['title', 'month_year']).size().unstack(fill_value=0)
                    
                    # Filter for roles with at least 10 postings to avoid noise
                    vel_pivot = vel_pivot[vel_pivot.sum(axis=1) > 10]
                    
                    # Calculate growth
                    if last_m in vel_pivot.columns and prev_m in vel_pivot.columns:
                        vel_pivot['growth_pct'] = ((vel_pivot[last_m] - vel_pivot[prev_m]) / vel_pivot[prev_m]) * 100
                        # Get top 10 fastest growing
                        top_vel = vel_pivot.sort_values('growth_pct', ascending=False).head(10).reset_index()
                        
                        fig_vel = px.bar(top_vel, x='growth_pct', y='title', orientation='h',
                                         title=f"Job Role Velocity (Growth %: {prev_m} to {last_m})",
                                         labels={'growth_pct': 'MoM Growth %', 'title': 'Job Role'},
                                         color='growth_pct', color_continuous_scale='Tealgrn',
                                         template="plotly_white")
                        fig_vel.update_layout(showlegend=False)
                        st.plotly_chart(fig_vel, use_container_width=True)
                        
                        st.markdown("""
                        **Key Takeaways:**
                        1.  **Early Warning System:** Identifies "Fast-Moving" roles that are suddenly spiking in demand, often signaling a new technology trend.
                        2.  **Niche Opportunities:** Highlights specific job titles to target for "Just-in-Time" short courses or bootcamps.
                        """)
                else:
                    st.info("Insufficient monthly data to calculate velocity.")

        except Exception as e:
            st.error(f"Forecasting Engine Error: {e}")
            st.markdown("Fallback: Please ensure sufficient historical data (months) exists for time-series analysis.")