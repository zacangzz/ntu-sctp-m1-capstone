import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import re
import os

# ==========================================
# 1. CONFIGURATION & STYLING
# ==========================================
st.set_page_config(page_title="SG Job Market Dashboard for Curriculum Design", layout="wide", page_icon="📊")

# Custom CSS with FORCED COLORS for visibility and Formatting
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .metric-value {
        font-size: 24px;
        font-weight: 700;
        color: #2c3e50 !important; 
    }
    .metric-label {
        font-size: 14px;
        color: #6c757d !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .insight-box {
        background-color: #e3f2fd; /* Light Blue */
        border-left: 5px solid #2196f3;
        padding: 15px;
        border-radius: 4px;
        margin-top: 10px;
        margin-bottom: 20px;
        height: 100%; 
    }
    .insight-text {
        color: #000000 !important; /* Force Black Text */
        font-size: 16px;
        line-height: 1.5;
    }
    .finding-title {
        font-weight: bold;
        color: #0d47a1 !important; /* Dark Blue Text */
        margin-bottom: 5px;
        font-size: 18px;
    }
    li {
        color: #000000 !important; /* Force List items Black */
        margin-bottom: 5px;
    }
    b {
        font-weight: 700;
        color: #000;
    }
    /* Dropdown / Selectbox text contrast */
    [data-testid="stSelectbox"] label, [data-testid="stSelectbox"] p {
        color: #1e293b !important;
    }
    [data-testid="stSelectbox"] div[data-baseweb="select"] {
        background-color: #ffffff !important;
        color: #1e293b !important;
    }
    [data-baseweb="select"] > div {
        background-color: #ffffff !important;
        color: #1e293b !important;
    }
    [data-baseweb="popover"] li, [data-baseweb="popover"] [role="option"] {
        background-color: #ffffff !important;
        color: #1e293b !important;
    }
    [data-baseweb="popover"] li:hover, [data-baseweb="popover"] [role="option"]:hover {
        background-color: #f1f5f9 !important;
        color: #0f172a !important;
    }
    option {
        background-color: #ffffff !important;
        color: #1e293b !important;
    }
</style>
""", unsafe_allow_html=True)

class Config:
    DATA_FILE = 'data/cleaned-sgjobdata.parquet'
    SKILL_FILE = 'data/cleaned-sgjobdata-category-withskills.parquet'
    CACHE_TTL = 3600

def _remove_outliers(df, col):
    """IQR-based outlier clipping (same logic as preprocess_data.remove_outliers)."""
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return np.where(df[col] > upper_bound, upper_bound,
                    np.where(df[col] < lower_bound, lower_bound, df[col]))


# ==========================================
# 2. DATA PROCESSING ENGINE
# ==========================================
class DataProcessor:
    @staticmethod
    def _parse_categories(cat_str):
        try:
            if isinstance(cat_str, str):
                return json.loads(cat_str.replace("'", '"'))
            return []
        except Exception:
            return []

    @staticmethod
    def _extract_category(val):
        if isinstance(val, dict):
            return val.get('category', val.get('name', str(val)))
        return str(val)

    @st.cache_data(ttl=Config.CACHE_TTL)
    def load_and_clean_data():
        if not os.path.exists(Config.DATA_FILE):
            st.error(f"🚨 Data file not found.")
            st.stop()

        df = pd.read_parquet(Config.DATA_FILE)
        # Align column names with app expectations
        rename_map = {}
        if 'title' in df.columns and 'jobtitle_cleaned' not in df.columns:
            rename_map['title'] = 'jobtitle_cleaned'
        if 'positionlevels' in df.columns and 'positionLevels' not in df.columns:
            rename_map['positionlevels'] = 'positionLevels'
        if rename_map:
            df = df.rename(columns=rename_map)
        # Use cleaned salary column
        if 'average_salary_cleaned' in df.columns:
            df['average_salary'] = df['average_salary_cleaned']
        if 'posting_date' in df.columns:
            df['posting_date'] = pd.to_datetime(df['posting_date'])
        elif 'posting_date' not in df.columns:
            # Parquet may lack date column; use placeholder for time-based charts
            df['posting_date'] = pd.Timestamp('2023-06-01')

        # 1. Derive Time-Based Columns
        df['month_year'] = df['posting_date'].dt.to_period('M').dt.to_timestamp()
        
        # 2. Explode Categories (One row per category)
        df['parsed_categories'] = df['categories'].apply(DataProcessor._parse_categories)
        df = df.explode('parsed_categories')
        df['category'] = df['parsed_categories'].apply(DataProcessor._extract_category)

        # 3. Handle Missing Values for Calculations
        df['num_vacancies'] = df['num_vacancies'].fillna(1) # Assume at least 1 vacancy if null
        df['num_applications'] = df['num_applications'].fillna(0)
        df['min_exp'] = df['min_exp'].fillna(0)
        # Outlier handling for min_exp (IQR-based, then clip to 0-15)
        if 'min_exp' in df.columns and not df.empty:
            df['min_exp'] = _remove_outliers(df, 'min_exp')
        df['min_exp'] = np.clip(df['min_exp'], 0, 15)

        # 4. Create Experience Segments (Used across multiple tabs)
        def categorize_exp(years):
            if years == 0: return '1. Fresh / Entry (0 yrs)'
            elif years <= 2: return '2. Junior (1-2 yrs)'
            elif years <= 5: return '3. Mid-Level (3-5 yrs)'
            elif years <= 8: return '4. Senior (6-8 yrs)'
            else: return '5. Lead / Expert (9+ yrs)'
        
        df['exp_segment'] = df['min_exp'].apply(categorize_exp)

        # Load skills if available (optional)
        if os.path.exists(Config.SKILL_FILE):
            skills_df = pd.read_parquet(Config.SKILL_FILE)
        else:
            skills_df = pd.DataFrame()

        return df, skills_df

# ==========================================
# 4. MAIN APP
# ==========================================
def main():
    # Load
    with st.spinner("Loading Data..."):
        df, skills_df = DataProcessor.load_and_clean_data()

    # Verify Data Integrity
    if df.empty:
        st.error("No valid data found after cleaning. Please check your CSV format.")
        st.stop()

    # Date Period
    if 'posting_date' in df.columns:
        min_d = df['posting_date'].min().strftime('%d %b %Y')
        max_d = df['posting_date'].max().strftime('%d %b %Y')
        period_str = f"{min_d} - {max_d}"
    else:
        period_str = "Date Unavailable"

    st.title("🎓 SG Job Market Dashboard for Curriculum Design")
    st.markdown("Aligning Curriculum with Real-Time Market Structure")
    st.write(f"**Data Period:** {period_str}")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Executive Summary", 
        "🏭 Sectoral Demand & Momentum", 
        "🛠️ Skill & Experience", 
        "🎓 Education Gap & Opportunity"
    ])

    # --- TAB 1: EXECUTIVE ---
    with tab1:
        st.subheader("High-Level Market Snapshot")

        # -----------------------------------------------------------
        # 1. DATA PREPARATION (METRICS)
        # -----------------------------------------------------------
        
        # 1.1 Ensure View Column Exists & is Numeric
        view_col = 'num_views'

        if view_col not in df.columns:
            df[view_col] = 0 # Create dummy if missing
            st.toast(f"⚠️ Column `{view_col}` not found. Views set to 0.", icon="ℹ️")
        else:
            # Force convert to numeric, turning errors/blanks into 0
            df[view_col] = pd.to_numeric(df[view_col], errors='coerce').fillna(0)

        # 1.2 HELPER: Calculate Stats for a specific metric
        def get_kpi_stats(df, count_method='sum', value_col='num_vacancies'):
            """
            count_method: 'sum' (for vacancies/views) or 'count' (for job posts)
            value_col: The column to sum (e.g., num_vacancies)
            """
            if count_method == 'sum':
                total_val = df[value_col].sum()
                # Top sector by sum of that value
                top_sector = df.groupby('category')[value_col].sum().idxmax()
                
                # Weighted Average Salary (More accurate for Vacancies/Views)
                if total_val > 0:
                    avg_sal = (df['average_salary'] * df[value_col]).sum() / total_val
                else:
                    avg_sal = 0
            else: # 'count' (Job Posts)
                total_val = len(df)
                # Top sector by count of rows
                top_sector = df['category'].value_counts().idxmax()
                # Simple Average Salary
                avg_sal = df['average_salary'].mean()

            return total_val, top_sector, avg_sal

        # CALC 1: Total Vacancies
        vac_total, vac_top, vac_sal = get_kpi_stats(df, 'sum', 'num_vacancies')
        
        # CALC 2: Total Job Posted (Rows)
        post_total, post_top, post_sal = get_kpi_stats(df, 'count', 'title')
        
        # CALC 3: Total Job Views (Using num_views)
        view_total, view_top, view_sal = get_kpi_stats(df, 'sum', view_col)

        # -----------------------------------------------------------
        # 2. VISUALIZATION (CUSTOM CARDS)
        # -----------------------------------------------------------
        
        st.markdown("""
        <style>
            .kpi-card {
                background-color: #f8f9fa;
                border-left: 5px solid #2E86C1;
                padding: 15px;
                border-radius: 5px;
                box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
                margin-bottom: 10px;
            }
            .kpi-title { font-size: 14px; color: #666; font-weight: bold; text-transform: uppercase;}
            .kpi-value { font-size: 28px; color: #2E86C1; font-weight: 800; margin: 5px 0;}
            .kpi-sub { font-size: 13px; color: #444; margin-top: 5px; line-height: 1.4;}
            .kpi-icon { font-size: 20px; float: right; }
        </style>
        """, unsafe_allow_html=True)

        kpi1, kpi2, kpi3 = st.columns(3)

        with kpi1:
            st.markdown(f"""
            <div class="kpi-card">
                <span class="kpi-icon">👥</span>
                <div class="kpi-title">Total Vacancies</div>
                <div class="kpi-value">{vac_total:,.0f}</div>
                <div class="kpi-sub">
                    <b>🏆 Top:</b> {vac_top}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with kpi2:
            st.markdown(f"""
            <div class="kpi-card">
                <span class="kpi-icon">📝</span>
                <div class="kpi-title">Total Job Posted</div>
                <div class="kpi-value">{post_total:,.0f}</div>
                <div class="kpi-sub">
                    <b>🏆 Top:</b> {post_top}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with kpi3:
            st.markdown(f"""
            <div class="kpi-card">
                <span class="kpi-icon">👁️</span>
                <div class="kpi-title">Total Job Views</div>
                <div class="kpi-value">{view_total:,.0f}</div>
                <div class="kpi-sub">
                    <b>🏆 Top:</b> {view_top}
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.divider()

        # -----------------------------------------------------------
        # 3. DYNAMIC CHART (TOP 10 SECTORS)
        # -----------------------------------------------------------
        
        c_head, c_opt = st.columns([3, 1])
        with c_head:
            st.markdown("#### 📊 Top 10 Sectors Breakdown")
        with c_opt:
            chart_metric = st.selectbox(
                "View By:", 
                ["Vacancies", "Job Posts", "Job Views"],
                index=0
            )

        # Logic to switch data based on selection
        if chart_metric == "Vacancies":
            metric_col = 'num_vacancies'
            agg_func = 'sum'
            bar_color = '#2E86C1' # Blue
            x_label = 'Total Vacancies'
        elif chart_metric == "Job Posts":
            metric_col = 'category' # Just for counting
            agg_func = 'count'
            bar_color = '#28B463' # Green
            x_label = 'Number of Posts'
        else: # Job Views
            metric_col = view_col # Use the metadata column
            agg_func = 'sum'
            bar_color = '#E67E22' # Orange
            x_label = 'Total Views'

        # Prepare Chart Data (exclude "Others" category)
        df_top10 = df[df['category'] != 'Others']
        if agg_func == 'sum':
            top_sectors_chart = df_top10.groupby('category')[metric_col].sum().sort_values(ascending=False).head(10).reset_index()
            top_sectors_chart.columns = ['category', 'Value']
        else:
            top_sectors_chart = df_top10['category'].value_counts().head(10).reset_index()
            top_sectors_chart.columns = ['category', 'Value']

        # Render Chart
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=top_sectors_chart['category'],
            x=top_sectors_chart['Value'],
            orientation='h',
            marker_color=bar_color,
            text=top_sectors_chart['Value'],
            texttemplate='%{text:.2s}',
            textposition='outside'
        ))

        fig.update_layout(
            yaxis={'categoryorder':'total ascending'},
            xaxis={'title': x_label},
            title=f"Top 10 Sectors by {chart_metric}",
            height=500,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig, use_container_width=True, key="executive_chart")

# --- TAB 2: SECTORAL DEMAND & MOMENTUM ---
    with tab2:
        st.subheader("🏭 Sectoral Demand & Momentum")
        st.markdown("Objective: Identify \"What\" to teach by tracking the velocity of industry needs.")
        
        col2_1, col2_2 = st.columns([1, 1])
        
        with col2_1:
            # 1. Market Share of Hiring (Vacancies)
            st.markdown("#### Market Share of Hiring")   
            st.caption("Which sectors have the highest volume of vacancies?")
            
            share_df = df.groupby('category')['num_vacancies'].sum().reset_index()
            # Sort and group small sectors
            share_df = share_df.sort_values('num_vacancies', ascending=False)
            top_share = share_df.head(15)
            others_val = share_df.iloc[15:]['num_vacancies'].sum()
            others_df = pd.DataFrame([['Others', others_val]], columns=['category', 'num_vacancies'])
            final_share = pd.concat([top_share, others_df])
            
            fig_share = px.pie(final_share, values='num_vacancies', names='category', hole=0.4,
                               color_discrete_sequence=px.colors.qualitative.Safe)
            # Make "Others" less prominent by pulling it out slightly
            if "Others" in final_share['category'].values:
                category_list = final_share['category'].tolist()
                others_position = category_list.index("Others")
                # Create pull array to pull "Others" slice out slightly (deselected effect)
                pull_values = [0] * len(final_share)
                pull_values[others_position] = 0.1  # Pull "Others" out by 10%
                fig_share.update_traces(pull=pull_values)
            st.plotly_chart(fig_share, use_container_width=True, key="market_share_chart")

        with col2_2:
            # 2. Role Prevalence Index (Job Titles)
            st.markdown("#### Role Prevalence Index")
            st.caption("Most frequently advertised job titles (Normalized 0-100)")
            
            # Add sector filter
            col_filter, col_dropdown = st.columns([1, 2])
            with col_filter:
                st.markdown("**Filter by Sector**")
            with col_dropdown:
                all_sectors = ['All'] + sorted(df['category'].unique().tolist())
                selected_sector_filter = st.selectbox("", all_sectors, key="role_sector_filter", label_visibility="collapsed")
            
            # Filter data based on sector selection
            if selected_sector_filter == 'All':
                filtered_df = df
            else:
                filtered_df = df[df['category'] == selected_sector_filter]
            
            title_counts = filtered_df['jobtitle_cleaned'].value_counts().reset_index()
            title_counts.columns = ['Job Title', 'Count']
            # Normalize to 100
            max_count = title_counts['Count'].max()
            title_counts['prevalence'] = (title_counts['Count'] / max_count) * 100
            
            fig_role = px.bar(title_counts.head(10), x='prevalence', y='Job Title', orientation='h',
                              color='prevalence', color_continuous_scale='Blues',
                              labels={'prevalence': 'Prevalence Index'})
            fig_role.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_role, use_container_width=True, key="role_prevalence_chart")

        # 3. Demand Velocity (Vacancy Growth)
        st.markdown("#### 📈 Demand Velocity (Vacancy Growth)")
        st.caption("Trend of vacancies over time. Steep lines = High Momentum.")
        
        # Top 10 sectors by vacancy volume
        velocity_df = df[df['category'] != 'Others']
        top_10_sectors = velocity_df.groupby('category')['num_vacancies'].sum().nlargest(10).index
        velocity_df = velocity_df[velocity_df['category'].isin(top_10_sectors)]
        velocity_df = velocity_df.groupby(['month_year', 'category'])['num_vacancies'].sum().reset_index()
        
        if len(velocity_df) > 1:
            fig_vel = px.line(velocity_df, x='month_year', y='num_vacancies', color='category',
                              markers=True, line_shape='spline',
                              title="Top 10 Sectors: Vacancy Trend Over Time",
                              labels={'month_year': 'Posting Date', 'num_vacancies': 'Vacancies', 'category': 'Sector'})
            st.plotly_chart(fig_vel, use_container_width=True, key="demand_velocity_chart")
        else:
            st.warning("Not enough data points for time-series velocity.")

        # 4. Bulk Hiring Map (sector × time heatmap)
        st.markdown("#### 🗺️ Bulk Hiring Map")
        st.caption("Vacancy hotspots by sector and time. Darker = higher bulk hiring intensity.")
        bulk_df = df[df['category'] != 'Others']
        top_sectors_bulk = bulk_df.groupby('category')['num_vacancies'].sum().nlargest(12).index
        bulk_filtered = bulk_df[bulk_df['category'].isin(top_sectors_bulk)]
        bulk_pivot = bulk_filtered.pivot_table(
            index='category', columns='month_year', values='num_vacancies', aggfunc='sum'
        ).fillna(0)
        fig_bulk = px.imshow(
            bulk_pivot, aspect='auto', color_continuous_scale='YlOrRd',
            labels=dict(x='Month', y='Sector', color='Vacancies')
        )
        st.plotly_chart(fig_bulk, use_container_width=True, key="bulk_hiring_map")

        # 5. Skills in High Demand
        st.markdown("#### High Demand Skills")
        st.caption("Top 10 skills by unique job postings over time.")
        
        if not skills_df.empty:
            # Ensure posting_date is datetime
            skills_df['posting_date'] = pd.to_datetime(skills_df['posting_date'], errors='coerce')
            skills_df = skills_df.dropna(subset=['posting_date'])
            
            # Extract year-month for grouping
            skills_df['year_month'] = skills_df['posting_date'].dt.to_period('M').astype(str)
            
            # Get available months sorted
            available_months = sorted(skills_df['year_month'].unique())
            
            if len(available_months) > 0:
                # Add sector filter
                skills_sectors = ['All'] + sorted(skills_df['category'].dropna().unique().tolist())
                col_skills_filter, col_skills_space = st.columns([1, 3])
                with col_skills_filter:
                    st.markdown("**Filter by Sector**")
                with col_skills_space:
                    selected_skills_sector = st.selectbox("", skills_sectors, key="skills_sector_filter", label_visibility="collapsed")
                
                # Filter by sector if selected
                skills_filtered = skills_df.copy()
                if selected_skills_sector != 'All':
                    skills_filtered = skills_filtered[skills_filtered['category'] == selected_skills_sector]
                
                # Prepare data for all months
                frames = []
                all_skills = set()
                
                for month in available_months:
                    month_data = skills_filtered[skills_filtered['year_month'] == month]
                    skill_counts = month_data.groupby('skill')['job_id'].nunique().reset_index(name='job_count')
                    skill_counts = skill_counts.sort_values('job_count', ascending=False).head(10)
                    
                    if not skill_counts.empty:
                        all_skills.update(skill_counts['skill'].tolist())
                        
                        frames.append(go.Frame(
                            data=[go.Bar(
                                y=skill_counts['skill'],
                                x=skill_counts['job_count'],
                                orientation='h',
                                marker=dict(color=skill_counts['job_count'], colorscale='Blues')
                            )],
                            name=month
                        ))
                
                # Get initial month data
                initial_month = available_months[0]
                initial_data = skills_filtered[skills_filtered['year_month'] == initial_month]
                initial_counts = initial_data.groupby('skill')['job_id'].nunique().reset_index(name='job_count')
                initial_counts = initial_counts.sort_values('job_count', ascending=False).head(10)
                
                if frames and not initial_counts.empty:
                    # Create figure with frames
                    chart_title = f'Top 10 Skills by Month: {initial_month}' if selected_skills_sector == 'All' else f'Top 10 Skills in {selected_skills_sector}'
                    
                    fig = go.Figure(
                        data=[go.Bar(
                            y=initial_counts['skill'],
                            x=initial_counts['job_count'],
                            orientation='h',
                            marker=dict(color=initial_counts['job_count'], colorscale='Blues')
                        )],
                        frames=frames
                    )
                    
                    fig.update_layout(
                        title=chart_title,
                        xaxis_title='Number of Unique Job Postings',
                        yaxis_title='Skill',
                        height=600,
                        yaxis={'categoryorder': 'total ascending'},
                        sliders=[{
                            'active': 0,
                            'steps': [
                                {
                                    'args': [[f.name], {'frame': {'duration': 300}, 'mode': 'immediate', 'transition': {'duration': 300}}],
                                    'label': f.name,
                                    'method': 'animate'
                                } for f in frames
                            ],
                            'transition': {'duration': 300}
                        }]
                    )
                    
                    st.plotly_chart(fig, use_container_width=True, key="skills_demand_chart")
                else:
                    st.info(f"No skills data available for {selected_skills_sector}")
            else:
                st.info("No date information available in skills data.")
        else:
            st.info("Skills data not available.")
            
 # --- TAB 3: SKILL & EXPERIENCE ---
    with tab3:
        st.subheader("🛠️ Skill & Experience Analysis")
        st.markdown("Objective: Align the \"Level\" of training with market reality to ensure graduate ROI.")
        
        # Seniority Pay-Scale, Experience Gate, Credential Depth
        c3a, c3b, c3c = st.columns(3)
        
        with c3a:
            st.markdown("#### Seniority Pay-Scale")
            st.caption("Average salary by experience level (weighted by vacancies)")
            pay_scale = df.groupby('exp_segment').apply(
                lambda g: (g['average_salary'] * g['num_vacancies']).sum() / g['num_vacancies'].sum()
            ).reset_index(name='avg_salary')
            pay_scale = pay_scale.sort_values('exp_segment')
            fig_pay = px.bar(pay_scale, x='exp_segment', y='avg_salary',
                             labels={'exp_segment': 'Experience Level', 'avg_salary': 'Avg Salary (SGD)'},
                             color='avg_salary', color_continuous_scale='Blues',
                             title="Salary by Seniority")
            fig_pay.update_layout(xaxis_tickangle=-45, showlegend=False)
            st.plotly_chart(fig_pay, use_container_width=True, key="seniority_payscale_chart")
        
        with c3b:
            st.markdown("#### The \"Experience Gate\"")
            st.caption("Vacancies accessible at each experience tier")
            gate_df = df.groupby('exp_segment')['num_vacancies'].sum().reset_index()
            gate_df = gate_df.sort_values('exp_segment')
            fig_gate = px.bar(gate_df, x='exp_segment', y='num_vacancies',
                              labels={'exp_segment': 'Experience Level', 'num_vacancies': 'Vacancies'},
                              color='num_vacancies', color_continuous_scale='Viridis',
                              title="Market Access by Experience")
            fig_gate.update_layout(xaxis_tickangle=-45, showlegend=False)
            st.plotly_chart(fig_gate, use_container_width=True, key="experience_gate_chart")
        
        with c3c:
            st.markdown("#### Credential Depth")
            st.caption("Distribution of position levels (seniority titles)")
            cred_df = df.copy()
            cred_df['positionLevels'] = cred_df['positionLevels'].fillna('Not specified')
            pos_levels = cred_df.groupby('positionLevels')['num_vacancies'].sum().reset_index()
            pos_levels = pos_levels.sort_values('num_vacancies', ascending=False)
            fig_cred = px.bar(pos_levels, x='positionLevels', y='num_vacancies',
                              labels={'positionLevels': 'Position Level', 'num_vacancies': 'Vacancies'},
                              color='num_vacancies', color_continuous_scale='Teal',
                              title="Credential / Level Distribution")
            fig_cred.update_layout(xaxis_tickangle=-45, showlegend=False)
            st.plotly_chart(fig_cred, use_container_width=True, key="credential_depth_chart")

        # Experience vs Compensation
        st.markdown("#### Experience vs Compensation")
        st.caption("Average salary by years of experience required. Shows the pay-off for each experience tier.")
        
        # Add sector filter
        exp_comp_sectors = ['All'] + sorted(df['category'].unique().tolist())
        col_exp_filter, col_exp_dropdown = st.columns([1, 3])
        with col_exp_filter:
            st.markdown("**Filter by Sector**")
        with col_exp_dropdown:
            selected_exp_sector = st.selectbox("", exp_comp_sectors, key="exp_comp_sector_filter", label_visibility="collapsed")
        
        # Filter data based on sector selection
        if selected_exp_sector == 'All':
            exp_comp_df = df
        else:
            exp_comp_df = df[df['category'] == selected_exp_sector]
        
        exp_comp = exp_comp_df.groupby('min_exp').apply(
            lambda g: (g['average_salary'] * g['num_vacancies']).sum() / g['num_vacancies'].sum()
        ).reset_index(name='avg_salary')
        exp_comp = exp_comp.sort_values('min_exp')
        chart_title = "Compensation by Experience Requirement" if selected_exp_sector == 'All' else f"Compensation by Experience Requirement - {selected_exp_sector}"
        fig_exp_comp = px.line(
            exp_comp, x='min_exp', y='avg_salary', markers=True,
            labels={'min_exp': 'Years of Experience Required', 'avg_salary': 'Avg Salary (SGD)'},
            title=chart_title
        )
        fig_exp_comp.update_traces(line=dict(color='#2E86C1', width=2), marker=dict(size=10))
        st.plotly_chart(fig_exp_comp, use_container_width=True, key="exp_comp_chart")
            
        st.markdown("#### 🔍 Category vs Experience Heatmap")
        st.caption("Where is the demand concentrated?")
        
        # Heatmap: X=Experience Level, Y=Top 10 Categories, Z=Vacancies
        top_10_cats = df['category'].value_counts().head(10).index
        heat_data = df[df['category'].isin(top_10_cats)]
        
        heat_pivot = heat_data.pivot_table(index='category', columns='exp_segment', 
                                         values='num_vacancies', aggfunc='sum').fillna(0)
        
        fig_heat = px.imshow(heat_pivot, aspect='auto', color_continuous_scale='Viridis',
                             labels=dict(x="Experience Level", y="Sector", color="Vacancies"))
        st.plotly_chart(fig_heat, use_container_width=True, key="skill_experience_heatmap")

# --- TAB 4: EDUCATION GAP & OPPORTUNITY ---
    with tab4:
        st.subheader("🎓 Educational Gap & Opportunity")
        st.markdown("Objective: Identify \"Blue Ocean\" opportunities where job matching rates are highest due to low competition.")
        
        # Prepare metrics for Opportunity Score and Competition Index
        p2_metrics = df.groupby('category').agg({
            'num_vacancies': 'sum',
            'num_applications': 'sum',
            'min_exp': 'mean',
            'job_id': 'count'
        }).reset_index()
        p2_metrics['opp_score'] = p2_metrics['num_vacancies'] / (p2_metrics['min_exp'] + 1)
        p2_metrics['comp_index'] = p2_metrics.apply(
            lambda x: x['num_applications'] / x['num_vacancies'] if x['num_vacancies'] > 0 else 0,
            axis=1
        )

        st.markdown("#### Competition Index")
        st.caption("Formula: Total Applications ÷ Total Vacancies")
        top_comp = p2_metrics.nlargest(10, 'comp_index')
        fig_comp = px.bar(top_comp, x='comp_index', y='category', orientation='h',
                          color='comp_index', color_continuous_scale='OrRd',
                          title="Most Competitive Sectors")
        fig_comp.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_comp, use_container_width=True, key="competition_index_chart")

        # Supply vs Demand Treemap
        st.markdown("#### Supply vs Demand")
        st.caption("Treemap: Rectangle size = Vacancies (demand), Color intensity = Applications (supply).")
        supply_demand = p2_metrics[p2_metrics['category'] != 'Others'].copy()
        supply_demand = supply_demand.sort_values('num_vacancies', ascending=False).head(20)
        # Create treemap: size by vacancies, color by applications
        fig_supply_demand = px.treemap(
            supply_demand, path=[px.Constant("All Sectors"), 'category'],
            values='num_vacancies', color='num_applications',
            color_continuous_scale='RdYlGn_r',
            labels={'num_vacancies': 'Vacancies (Size)', 'num_applications': 'Applications (Color)'},
            title='Supply vs Demand Treemap',
            hover_data=['num_vacancies', 'num_applications']
        )
        st.plotly_chart(fig_supply_demand, use_container_width=True, key="supply_demand_treemap")

        # The "Hidden Demand" Quadrant Analysis
        st.markdown("#### The \"Hidden Demand\"")
        st.caption("Quadrant analysis: Vacancies vs Applications. Identifies hidden opportunities (high vacancies, low applications).")
        hidden_demand = p2_metrics[p2_metrics['category'] != 'Others'].copy()
        if len(hidden_demand) > 0:
            # Calculate median thresholds for quadrant division
            median_vac = hidden_demand['num_vacancies'].median()
            median_app = hidden_demand['num_applications'].median()
            # Categorize quadrants
            def assign_quadrant(row):
                if row['num_vacancies'] >= median_vac and row['num_applications'] < median_app:
                    return 'Hidden Opportunity'
                elif row['num_vacancies'] >= median_vac and row['num_applications'] >= median_app:
                    return 'Competitive Market'
                elif row['num_vacancies'] < median_vac and row['num_applications'] < median_app:
                    return 'Niche Market'
                else:
                    return 'Oversupplied'
            hidden_demand['quadrant'] = hidden_demand.apply(assign_quadrant, axis=1)
            # Create text column: show category name except for Niche Market
            hidden_demand['category_text'] = hidden_demand.apply(
                lambda row: '' if row['quadrant'] == 'Niche Market' else row['category'], axis=1
            )
            # Create scatter plot with category labels
            fig_hidden = px.scatter(
                hidden_demand, x='num_vacancies', y='num_applications',
                size='num_vacancies', color='quadrant',
                hover_name='category', text='category_text',
                labels={'num_vacancies': 'Vacancies', 'num_applications': 'Applications'},
                title='Hidden Demand Quadrant Analysis',
                color_discrete_map={
                    'Hidden Opportunity': '#28B463',
                    'Competitive Market': '#E67E22',
                    'Niche Market': '#95A5A6',
                    'Oversupplied': '#E74C3C'
                }
            )
            fig_hidden.update_traces(textposition='top center', textfont_size=9)
            # Add quadrant divider lines
            fig_hidden.add_hline(y=median_app, line_dash="dash", line_color="gray", 
                               annotation_text=f"Median Apps: {median_app:.0f}")
            fig_hidden.add_vline(x=median_vac, line_dash="dash", line_color="gray",
                               annotation_text=f"Median Vacancies: {median_vac:.0f}")
            fig_hidden.update_layout(height=600)
            st.plotly_chart(fig_hidden, use_container_width=True, key="hidden_demand_chart")
        

if __name__ == "__main__":
    main()