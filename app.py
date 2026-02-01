import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import os
import gc
from statsmodels.tsa.arima.model import ARIMA
from datetime import timedelta

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Workforce Intelligence Portal", layout="wide", initial_sidebar_state="collapsed")

# --- CUSTOM CSS ---
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

# --- OPTIMIZED DATA LOADER ---
@st.cache_data(show_spinner="Processing Market Intelligence...", ttl=3600)
def load_aggregated_data(file_path):
    """
    Loads raw data, performs all heavy aggregations and forecasting ONCE, 
    and returns a lightweight dictionary of summary tables.
    """
    # 1. Load Raw Data
    if not os.path.exists(file_path):
        if os.path.exists('SGJobsample_100000.csv.xz'):
            file_path = 'SGJobsample_100000.csv.xz'
        else:
            return None

    try:
        # Load specific columns to save memory if possible, or all if dynamic
        df = pd.read_csv(file_path)
    except Exception:
        return None

    # 2. Data Cleaning & Type Optimization (Crucial for Memory)
    def parse_category(x):
        try:
            if pd.isna(x): return []
            data = json.loads(x.replace("'", '"'))
            return [item['category'] for item in data]
        except: return []

    if 'categories' in df.columns:
        df['category_list'] = df['categories'].apply(parse_category)
    
    # Date conversion
    df['metadata_newPostingDate'] = pd.to_datetime(df['metadata_newPostingDate'], errors='coerce')
    df = df.dropna(subset=['metadata_newPostingDate'])
    df['month_year'] = df['metadata_newPostingDate'].dt.to_period('M').astype(str)

    # Salary optimization
    if 'average_salary' not in df.columns:
        df['average_salary'] = (df['salary_minimum'] + df['salary_maximum']) / 2
    
    # Filter outliers and downcast types
    q_low = df["average_salary"].quantile(0.01)
    q_hi = df["average_salary"].quantile(0.99)
    df = df[(df["average_salary"] >= q_low) & (df["average_salary"] <= q_hi)]
    
    # Memory optimization: Convert object to category where applicable
    for col in ['employmentTypes', 'positionLevels', 'title']:
        if col in df.columns:
            df[col] = df[col].astype('category')

    # Experience bracket
    df['experience_bracket'] = pd.cut(
        df['minimumYearsExperience'], 
        bins=[-1, 2, 5, 10, 100], 
        labels=['0-2 yrs (Entry)', '3-5 yrs (Mid)', '6-10 yrs (Senior)', '10+ yrs (Lead)']
    )

    # 3. PRE-CALCULATE AGGREGATES (To discard raw dataframe later)
    output = {}
    
    # Scalars
    output['min_date'] = df['metadata_newPostingDate'].min().strftime('%d %b %Y')
    output['max_date'] = df['metadata_newPostingDate'].max().strftime('%d %b %Y')
    output['total_openings'] = int(df['numberOfVacancies'].sum())
    output['median_pay'] = df['average_salary'].median()
    output['entry_rate'] = (df['minimumYearsExperience'] <= 2).mean() * 100
    
    # Chart 7 & 8: Structural Analysis
    output['seniority_counts'] = df['positionLevels'].value_counts().reset_index()
    output['seniority_counts'].columns = ['Level', 'Count']
    output['emp_counts'] = df['employmentTypes'].value_counts().reset_index()
    output['emp_counts'].columns = ['Type', 'Total']

    # Chart 6: Comp vs View (Sampled for performance)
    # Cap outliers
    q_view = df['metadata_totalNumberOfView'].quantile(0.95)
    q_app = df['metadata_totalNumberJobApplication'].quantile(0.95)
    comp_df = df[(df['metadata_totalNumberOfView'] < q_view) & (df['metadata_totalNumberJobApplication'] < q_app)]
    # Downsample if too large for scatter plot
    if len(comp_df) > 5000:
        comp_df = comp_df.sample(5000)
    output['comp_data'] = comp_df[['metadata_totalNumberOfView', 'metadata_totalNumberJobApplication', 'experience_bracket']].copy()

    # Chart 2: Top Titles
    output['title_vol'] = df.groupby('title')['numberOfVacancies'].sum().sort_values(ascending=True).tail(10).reset_index()

    # Job Velocity (Last 2 months)
    sorted_months = sorted(df['month_year'].unique())
    output['velocity_data'] = None
    if len(sorted_months) >= 2:
        last_m, prev_m = sorted_months[-1], sorted_months[-2]
        df_vel = df[df['month_year'].isin([last_m, prev_m])]
        vel_pivot = df_vel.groupby(['title', 'month_year']).size().unstack(fill_value=0)
        vel_pivot = vel_pivot[vel_pivot.sum(axis=1) > 10]
        if last_m in vel_pivot.columns and prev_m in vel_pivot.columns:
            vel_pivot['growth_pct'] = ((vel_pivot[last_m] - vel_pivot[prev_m]) / vel_pivot[prev_m]) * 100
            output['velocity_data'] = vel_pivot.sort_values('growth_pct', ascending=False).head(10).reset_index()
            output['velocity_dates'] = (prev_m, last_m)

    # 4. EXPLOSION & SECTOR AGGREGATES
    # We explode momentarily to calculate sector stats, then delete the exploded frame
    df_exploded = df.explode('category_list')
    
    # Top Sector (Scalar)
    output['top_sector'] = df_exploded['category_list'].mode()[0] if not df_exploded.empty else "N/A"
    
    # Chart 1: Sector Volume
    output['sector_vol'] = df_exploded.groupby('category_list')['numberOfVacancies'].sum().sort_values(ascending=True).tail(10).reset_index()
    
    # Chart 3: Experience by Sector
    top_sectors_list = df_exploded['category_list'].value_counts().head(10).index.tolist()
    df_top_exp = df_exploded[df_exploded['category_list'].isin(top_sectors_list)]
    output['exp_counts'] = df_top_exp.groupby(['category_list', 'experience_bracket']).size().reset_index(name='Jobs')
    
    # Chart 4: Salary Benchmark
    output['sal_bench'] = df_exploded.groupby('category_list')['average_salary'].median().sort_values(ascending=True).tail(15).reset_index()
    
    # Chart 5: Bulk Hiring
    bulk_agg = df_exploded.groupby('category_list').agg({'numberOfVacancies': 'sum', 'metadata_jobPostId': 'count'}).reset_index()
    bulk_agg['hiring_scale'] = bulk_agg['numberOfVacancies'] / bulk_agg['metadata_jobPostId']
    output['bulk_data'] = bulk_agg
    
    # Momentum (Top 5 Sectors Share)
    top_5 = df_exploded['category_list'].value_counts().head(5).index
    mom_df = df_exploded[df_exploded['category_list'].isin(top_5)]
    mom_pivot = mom_df.groupby(['month_year', 'category_list']).size().reset_index(name='count')
    mom_pivot['total_month'] = mom_pivot.groupby('month_year')['count'].transform('sum')
    mom_pivot['share'] = (mom_pivot['count'] / mom_pivot['total_month']) * 100
    output['mom_data'] = mom_pivot

    # Cleanup exploded frame
    del df_exploded
    del df_top_exp
    del mom_df
    
    # 5. FORECASTING (Run once and cache result)
    time_series = df.set_index('metadata_newPostingDate').resample('M').size().reset_index()
    time_series.columns = ['Date', 'Volume']
    output['time_series'] = time_series
    
    output['forecast'] = None
    if len(time_series) >= 6:
        try:
            model = ARIMA(time_series['Volume'], order=(1,1,1))
            model_fit = model.fit()
            steps = 6
            forecast_result = model_fit.get_forecast(steps=steps)
            
            last_date = time_series['Date'].iloc[-1]
            forecast_dates = [last_date + pd.DateOffset(months=x) for x in range(1, steps+1)]
            
            output['forecast'] = {
                'dates': forecast_dates,
                'mean': forecast_result.predicted_mean.tolist(),
                'ci_lower': forecast_result.conf_int(alpha=0.05).iloc[:, 0].tolist(),
                'ci_upper': forecast_result.conf_int(alpha=0.05).iloc[:, 1].tolist()
            }
        except:
            pass # Fail gracefully
            
    # Final cleanup
    del df
    gc.collect()
    
    return output

# --- APP EXECUTION ---
FILE_PATH = 'data/SGJobData.csv.xz'
# Load lightweight dictionary instead of massive dataframe
data = load_aggregated_data(FILE_PATH)

if data is None:
    st.error("Data could not be loaded. Please ensure 'SGJobData.csv.xz' is in the directory.")
    st.stop()

# --- DASHBOARD HEADER ---
st.title("🎓 Workforce Intelligence Portal")
st.markdown(f"**Data Reporting Period:** {data['min_date']} to {data['max_date']}")
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
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""<div class='metric-container'><div class='metric-label'>Total Job Openings</div><div class='metric-value'>{data['total_openings']:,}</div></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class='metric-container'><div class='metric-label'>Top Industry Sector</div><div class='metric-value'>{data['top_sector']}</div></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class='metric-container'><div class='metric-label'>Typical Monthly Salary</div><div class='metric-value'>${data['median_pay']:,.0f}</div></div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class='metric-container'><div class='metric-label'>Entry-Level Roles</div><div class='metric-value'>{data['entry_rate']:.1f}%</div></div>""", unsafe_allow_html=True)

    st.subheader("Strategic Recommendations")
    st.info(f"""
    * **Resource Allocation:** Shift faculty resources toward **{data['top_sector']}**, as it currently anchors the region's hiring volume.
    * **Curriculum Focus:** With **{data['entry_rate']:.1f}%** of roles open to fresh graduates, ensure Foundation/Diploma programs prioritize "Work-Ready" skills to capture this immediate demand.
    """, icon="💡")

# ==========================================
# TAB 2: MARKET DEMAND
# ==========================================
with tabs[1]:
    st.header("Industry Volume & Role Demand")
    c1, c2 = st.columns(2)
    with c1:
        fig1 = px.bar(data['sector_vol'], x='numberOfVacancies', y='category_list', orientation='h',
                      title="Top 10 Sectors by Total Job Openings",
                      color='numberOfVacancies', color_continuous_scale='Blues',
                      labels={'numberOfVacancies': 'Total Openings', 'category_list': 'Industry'},
                      template="plotly_white")
        fig1.update_traces(hovertemplate="<b>%{y}</b><br>Openings: %{x:,}")
        fig1.update_layout(showlegend=False)
        st.plotly_chart(fig1, use_container_width=True)
        st.markdown("**Key Takeaways:**\n1. **Volume Drivers:** Identifies the 'Big 10' industries providing the majority of employment.\n2. **Placement Strategy:** Programs aligned here offer the highest probability of rapid student placement.")

    with c2:
        fig2 = px.bar(data['title_vol'], x='numberOfVacancies', y='title', orientation='h',
                      title="Top 10 Most Demanded Job Titles",
                      color='numberOfVacancies', color_continuous_scale='Greens',
                      labels={'numberOfVacancies': 'Total Openings', 'title': 'Job Title'},
                      template="plotly_white")
        fig2.update_traces(hovertemplate="<b>%{y}</b><br>Openings: %{x:,}")
        fig2.update_layout(showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown("**Key Takeaways:**\n1. **Curriculum Specificity:** Highlights specific roles to inform module-level learning outcomes.\n2. **Career Counseling:** Concrete data for advisors on abundant job titles.")

# ==========================================
# TAB 3: GRADUATE PIPELINE
# ==========================================
with tabs[2]:
    st.header("Entry Barriers & ROI Benchmarks")
    c3, c4 = st.columns(2)
    with c3:
        fig3 = px.bar(data['exp_counts'], x='category_list', y='Jobs', color='experience_bracket',
                      title="Experience Requirements (Top 10 Sectors)",
                      labels={'category_list': 'Industry', 'Jobs': 'Job Count', 'experience_bracket': 'Experience'},
                      template="plotly_white", barmode='stack')
        fig3.update_traces(hovertemplate="<b>%{x}</b><br>%{data.name}<br>Count: %{y:,}")
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown("**Key Takeaways:**\n1. **Accessibility Check:** Identifies 'Green' sectors ideal for fresh graduates.\n2. **Upskilling Needs:** Highlights sectors needing Executive Education.")

    with c4:
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=data['sal_bench']['average_salary'], y=data['sal_bench']['category_list'],
            mode='markers', marker=dict(color='#1a73e8', size=12), name='Median Salary'))
        shapes = [dict(type="line", x0=0, y0=i, x1=row['average_salary'], y1=i, line=dict(color="#e0e0e0", width=2)) 
                  for i, row in data['sal_bench'].iterrows()]
        fig4.update_layout(title="Typical Monthly Salary Benchmark (Top 15 Sectors)", shapes=shapes,
            xaxis=dict(title="Median Monthly Salary ($)", tickformat="$,.0f"), yaxis=dict(title="Industry"),
            template="plotly_white", showlegend=False)
        st.plotly_chart(fig4, use_container_width=True)
        st.markdown("**Key Takeaways:**\n1. **ROI Demonstration:** Evidence of financial returns for high-value specializations.\n2. **Pricing Strategy:** Aligns tuition fees with expected earning potential.")

# ==========================================
# TAB 4: OPERATIONAL INTELLIGENCE
# ==========================================
with tabs[3]:
    st.header("Hiring Scale & Competition")
    c5, c6 = st.columns(2)
    with c5:
        fig5 = px.treemap(data['bulk_data'], path=['category_list'], values='numberOfVacancies',
            color='hiring_scale', color_continuous_scale='RdBu', title="Large-Scale Hiring Opportunities",
            labels={'numberOfVacancies': 'Total Openings', 'hiring_scale': 'Vacancies per Ad'})
        fig5.update_traces(hovertemplate="<b>%{label}</b><br>Total Openings: %{value:,}<br>Hiring Scale: %{color:.1f} jobs/ad")
        st.plotly_chart(fig5, use_container_width=True)
        st.markdown("**Key Takeaways:**\n1. **Partnership Targets:** Blue areas indicate 'Batch Hiring' employers, ideal for MOUs.\n2. **Efficiency:** Place cohorts of students with fewer partners.")

    with c6:
        fig6 = px.scatter(data['comp_data'], x='metadata_totalNumberOfView', y='metadata_totalNumberJobApplication',
            color='experience_bracket', title="Interest vs. Competition Level",
            labels={'metadata_totalNumberOfView': 'Job Views', 'metadata_totalNumberJobApplication': 'Applications'},
            template="plotly_white", opacity=0.6)
        fig6.update_layout(legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig6, use_container_width=True)
        st.markdown("**Key Takeaways:**\n1. **Hidden Gems:** Low views/low applications represent uncrowded entry points.\n2. **High Noise:** High views/high applications require standout portfolios.")

# ==========================================
# TAB 5: STRUCTURAL ANALYSIS
# ==========================================
with tabs[4]:
    st.header("Workforce Composition")
    c7, c8 = st.columns(2)
    with c7:
        fig7 = px.bar(data['seniority_counts'], x='Count', y='Level', orientation='h', title="Demand by Career Stage",
                      color='Level', template="plotly_white")
        fig7.update_layout(showlegend=False)
        fig7.update_traces(hovertemplate="<b>%{y}</b><br>Count: %{x:,}")
        st.plotly_chart(fig7, use_container_width=True)
        st.markdown("**Key Takeaways:**\n1. **Pipeline Gaps:** Visualizes the 'Missing Middle' or leadership gaps.\n2. **Program Calibration:** Avoids over-producing Entry-level candidates for Manager-heavy markets.")

    with c8:
        fig8 = px.pie(data['emp_counts'], values='Total', names='Type', hole=0.5, title="Job Contract Breakdown", template="plotly_white")
        fig8.update_layout(showlegend=False)
        fig8.update_traces(textinfo='percent+label')
        st.plotly_chart(fig8, use_container_width=True)
        st.markdown("**Key Takeaways:**\n1. **Gig Economy:** High Contract roles necessitate freelancing skills.\n2. **Stability:** Permanent roles suggest stable long-term employment.")

# ==========================================
# TAB 6: HIRING FORECAST
# ==========================================
with tabs[5]:
    st.header("🔮 Hiring Demand Forecast (Next 6 Months)")
    
    if data['forecast']:
        # Forecast Chart
        fig_f = go.Figure()
        ts = data['time_series']
        fc = data['forecast']
        
        fig_f.add_trace(go.Scatter(x=ts['Date'], y=ts['Volume'], mode='lines+markers', name='Historical Demand', line=dict(color='#1a73e8', width=2)))
        fig_f.add_trace(go.Scatter(x=fc['dates'] + fc['dates'][::-1], y=fc['ci_upper'] + fc['ci_lower'][::-1],
            fill='toself', fillcolor='rgba(232, 26, 26, 0.1)', line=dict(color='rgba(255,255,255,0)'), hoverinfo="skip", name='95% Risk Corridor'))
        fig_f.add_trace(go.Scatter(x=fc['dates'], y=fc['mean'], mode='lines+markers', name='Forecasted Volume', line=dict(color='#d93025', width=2, dash='dash')))
        
        fig_f.update_layout(title="Hiring Demand Forecast with Risk Corridor", xaxis_title="Timeline", yaxis_title="Job Postings",
            template="plotly_white", legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig_f, use_container_width=True)
        st.markdown("**Key Takeaways:**\n1. **Risk Assessment:** Narrow red corridor implies stable market safe for investment.\n2. **Trend:** Dashed line predicts expansion or contraction of total market.")
    else:
        st.warning("Insufficient data for forecasting.")
        
    st.divider()
    c_mom, c_vel = st.columns(2)
    with c_mom:
        fig_mom = px.area(data['mom_data'], x='month_year', y='share', color='category_list',
            title="Sector Momentum (% Market Share)", labels={'share': 'Market Share (%)', 'month_year': 'Month', 'category_list': 'Sector'},
            template="plotly_white")
        st.plotly_chart(fig_mom, use_container_width=True)
        st.markdown("**Key Takeaways:**\n1. **Dominance Shifts:** Visualizes sectors losing relevance vs emerging competitors.\n2. **Resource Planning:** Predicts budget needs based on share growth.")

    with c_vel:
        if data['velocity_data'] is not None:
            prev, curr = data['velocity_dates']
            fig_vel = px.bar(data['velocity_data'], x='growth_pct', y='title', orientation='h',
                title=f"Job Role Velocity (Growth %: {prev} to {curr})", labels={'growth_pct': 'MoM Growth %', 'title': 'Job Role'},
                color='growth_pct', color_continuous_scale='Tealgrn', template="plotly_white")
            fig_vel.update_layout(showlegend=False)
            st.plotly_chart(fig_vel, use_container_width=True)
            st.markdown("**Key Takeaways:**\n1. **Early Warning:** Identifies roles spiking in demand (tech trends).\n2. **Niche Opps:** Targets for 'Just-in-Time' short courses.")
        else:
            st.info("Insufficient data for velocity analysis.")