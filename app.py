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
</style>
""", unsafe_allow_html=True)

class Config:
    DATA_FILE = 'data/SGJobData_opt.csv.xz'
    SKILL_FILE = 'data/skillset.csv'
    CACHE_TTL = 3600

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

        df = pd.read_csv(Config.DATA_FILE, parse_dates=['posting_date'])

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
            skills_df = pd.read_csv(Config.SKILL_FILE)
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

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Executive Summary", 
        "🏭 Sectoral Demand & Momentum", 
        "🛠️ Skill & Experience", 
        "🎓 Education Gap & Opportunity",
        "💰 Salary & Value", 
        "📋 Curriculum Deep-Dive"
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

        # Prepare Chart Data
        if agg_func == 'sum':
            top_sectors_chart = df.groupby('category')[metric_col].sum().sort_values(ascending=False).head(10).reset_index()
            top_sectors_chart.columns = ['category', 'Value']
        else:
            top_sectors_chart = df['category'].value_counts().head(10).reset_index()
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
            
            title_counts = filtered_df['title'].value_counts().reset_index()
            title_counts.columns = ['Job Title', 'Count']
            # Normalize to 100
            max_count = title_counts['Count'].max()
            title_counts['prevalence'] = (title_counts['Count'] / max_count) * 100
            
            fig_role = px.bar(title_counts.head(10), x='prevalence', y='Job Title', orientation='h',
                              color='prevalence', color_continuous_scale='Blues',
                              labels={'prevalence': 'Prevalence Index'})
            fig_role.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_role, use_container_width=True, key="role_prevalence_chart")

        # 3. Demand Velocity (Time Series)
        st.markdown("#### 📈 Demand Velocity (Vacancy Growth)")
        st.caption("Trend of vacancies over time. Steep lines = High Momentum.")
        
        # Aggregate by Month
        velocity_df = df.groupby('month_year')['num_vacancies'].sum().reset_index()
        
        if len(velocity_df) > 1:
            fig_vel = px.line(velocity_df, x='month_year', y='num_vacancies', markers=True,
                              line_shape='spline', title="Total Market Vacancy Trend")
            
            # Add trendline annotation
            start_val = velocity_df.iloc[0]['num_vacancies']
            end_val = velocity_df.iloc[-1]['num_vacancies']
            growth = ((end_val - start_val) / start_val) * 100 if start_val > 0 else 0
            
            fig_vel.add_annotation(x=velocity_df.iloc[-1]['month_year'], y=end_val,
                                   text=f"Growth: {growth:.1f}%", showarrow=True, arrowhead=1)
            st.plotly_chart(fig_vel, use_container_width=True, key="demand_velocity_chart")
        else:
            st.warning("Not enough data points for time-series velocity.")
            
 # --- TAB 3: SKILL & EXPERIENCE ---
    with tab3:
        st.subheader("🛠️ Skill & Experience Analysis")
        st.markdown("Objective: Align the \"Level\" of training with market reality to ensure graduate ROI.")
        
        st.markdown("#### 1. Experience Distribution")
        st.caption("Distribution of required experience levels across all sectors")
        
        exp_dist = df['min_exp'].value_counts().sort_index().reset_index()
        exp_dist.columns = ['Experience (Years)', 'Count']
        
        fig_exp = px.bar(exp_dist, x='Experience (Years)', y='Count', 
                       title="Required Experience Distribution")
        st.plotly_chart(fig_exp, use_container_width=True, key="experience_distribution_chart")

        # Seniority Pay-Scale, Experience Gate, Credential Depth
        c3a, c3b, c3c = st.columns(3)
        
        with c3a:
            st.markdown("#### 2. Seniority Pay-Scale")
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
            st.markdown("#### 3. The \"Experience Gate\"")
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
            st.markdown("#### 4. Credential Depth")
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
            
        st.markdown("#### 🔍 Deep Dive: Category vs Experience Heatmap")
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

        c4_1, c4_2 = st.columns(2)
        
        with c4_1:
            st.markdown("#### Opportunity Score")
            st.caption("Formula: Total Vacancies ÷ (Avg Min Exp + 1)")
            top_opp = p2_metrics.nlargest(10, 'opp_score')
            fig_opp = px.bar(top_opp, x='opp_score', y='category', orientation='h',
                             color='opp_score', color_continuous_scale='Teal',
                             title="Highest Opportunity Sectors")
            fig_opp.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_opp, use_container_width=True, key="opportunity_score_chart")
        
        with c4_2:
            st.markdown("#### Competition Index")
            st.caption("Formula: Total Applications ÷ Total Vacancies")
            top_comp = p2_metrics.nlargest(10, 'comp_index')
            fig_comp = px.bar(top_comp, x='comp_index', y='category', orientation='h',
                              color='comp_index', color_continuous_scale='OrRd',
                              title="Most Competitive Sectors")
            fig_comp.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_comp, use_container_width=True, key="competition_index_chart")

        st.markdown("#### 🔍 Deep Dive: Category vs Experience Heatmap")
        st.caption("Where is the demand concentrated?")
        
        # Heatmap: X=Experience Level, Y=Top 10 Categories, Z=Vacancies
        top_10_cats = df['category'].value_counts().head(10).index
        heat_data = df[df['category'].isin(top_10_cats)]
        
        heat_pivot = heat_data.pivot_table(index='category', columns='exp_segment', 
                                         values='num_vacancies', aggfunc='sum').fillna(0)
        
        fig_heat = px.imshow(heat_pivot, aspect='auto', color_continuous_scale='Viridis',
                             labels=dict(x="Experience Level", y="Sector", color="Vacancies"))
        st.plotly_chart(fig_heat, use_container_width=True, key="education_gap_heatmap")
        
    # --- TAB 5: SALARY ---
    with tab5:
        st.subheader("Compensation & ROI")
        
        st.markdown("#### 🫧 Experience vs. Compensation Matrix")
        bubble_data = df.groupby('category').agg(
            avg_exp=('min_exp', 'mean'),
            avg_sal=('average_salary', 'mean'),
            volume=('num_vacancies', 'sum')
        ).reset_index()
        
        fig_bub = px.scatter(bubble_data, x='avg_exp', y='avg_sal', size='volume', color='category',
                             hover_name='category', size_max=60,
                             labels={'avg_exp': 'Avg Min Experience (Years)', 'avg_sal': 'Avg Salary (SGD)'})
        st.plotly_chart(fig_bub, use_container_width=True, key="salary_bubble_chart")
        
        st.divider()
        st.markdown("#### 🔥 Salary Heatmap (Top Sectors vs Titles)")
        
        top_cats = df['category'].value_counts().head(8).index
        top_ts = df['title'].value_counts().head(8).index
        heatmap_df = df[df['category'].isin(top_cats) & df['title'].isin(top_ts)]

        if not heatmap_df.empty:
            pivot = heatmap_df.pivot_table(index='category', columns='title', values='average_salary', aggfunc='mean')
            fig_heat = px.imshow(pivot, labels=dict(x="Job Title", y="Sector", color="Salary"),
                                 color_continuous_scale='RdBu_r', aspect="auto")
            st.plotly_chart(fig_heat, use_container_width=True, key="salary_heatmap")

    # --- TAB 6: MARKET MOMENTUM ---
    with tab6:
        st.subheader("Market Momentum & Trends")
        st.markdown("Identifying emerging sectors, seasonal peaks, and growth velocity.")
        
        if 'month_year' in df.columns:
            trend_data = df.groupby(['month_year', 'category'])['num_vacancies'].sum().reset_index()
            top_sectors = df.groupby('category')['num_vacancies'].sum().nlargest(8).index.tolist()
            filtered_trend = trend_data[trend_data['category'].isin(top_sectors)]
            
            st.markdown("#### 📈 Historical Vacancy Volume (Top 8 Sectors)")
            fig_line = px.line(filtered_trend, x='month_year', y='num_vacancies', color='category',
                               title="Vacancy Trends over Time", markers=True,
                               labels={
                                   'month_year': 'Date',
                                   'num_vacancies': 'Vacancy Volume',
                                   'category': 'Industry'
                               })
            st.plotly_chart(fig_line, use_container_width=True, key="vacancy_trend_chart")
            
            st.divider()
            m1, m2 = st.columns(2)
            
            with m1:
                st.markdown("#### 🔥 Seasonal Hiring Heatmap")
                st.caption("Identify peak hiring months. Darker colors = Higher demand.")
                filtered_trend['Month'] = filtered_trend['month_year'].dt.strftime('%Y-%m')
                heatmap_data = filtered_trend.pivot_table(index='category', columns='Month', values='num_vacancies', aggfunc='sum').fillna(0)
                fig_hm = px.imshow(heatmap_data, labels=dict(x="Month", y="Industry", color="Vacancies"),
                                   color_continuous_scale='Magma_r', aspect="auto")
                st.plotly_chart(fig_hm, use_container_width=True, key="seasonal_heatmap")

            with m2:
                st.markdown("#### 🚀 Momentum Leaderboard (Recent Growth)")
                st.caption("Which sectors are accelerating? (Avg Month-on-Month Growth, Last Quarter)")
                filtered_trend = filtered_trend.sort_values(['category', 'month_year'])
                filtered_trend['mom_growth'] = filtered_trend.groupby('category')['num_vacancies'].pct_change() * 100
                recent_growth = filtered_trend.groupby('category').tail(3).groupby('category')['mom_growth'].mean().reset_index()
                recent_growth = recent_growth.sort_values('mom_growth', ascending=False)
                
                fig_mom = px.bar(recent_growth, x='mom_growth', y='category', orientation='h',
                                 title="Avg MoM Growth Rate (Last 3 Months)",
                                 labels={'mom_growth': 'Growth Rate (%)', 'category': 'Industry'},
                                 color='mom_growth', color_continuous_scale='Tealgrn')
                fig_mom.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig_mom, use_container_width=True, key="momentum_leaderboard")
            
        else:
            st.warning("No time-series data available. Check 'posting_date' parsing.")

    # --- TAB 5: CURRICULUM ---
    with tab6:
        st.subheader("📋 Curriculum Alignment Tool")
        
        sectors = sorted(df['category'].astype(str).unique())
        selected_sector = st.selectbox("Select Target Sector:", sectors)
        
        col_c1, col_c2 = st.columns(2)
        sec_df = df[df['category'] == selected_sector]

        with col_c1:
            st.info(f"📊 Live Market Stats: **{selected_sector}**")
            if not sec_df.empty:
                st.write(f"**Total Vacancies:** {sec_df['num_vacancies'].sum():,}")
                entry_sal = sec_df[sec_df['min_exp']<=2]['average_salary'].mean()
                st.write(f"**Avg Entry Salary:** ${entry_sal:,.0f}" if not pd.isna(entry_sal) else "N/A")
                st.markdown("**Top Job Title Keywords:**")
                st.bar_chart(sec_df['title'].value_counts().head(5))
            else:
                st.warning("No data found for this sector.")

        with col_c2:
            st.success(f"📚 Reference Curriculum: **{selected_sector}**")
            if not skills_df.empty:
                ref = skills_df[skills_df['category'] == selected_sector]
                if not ref.empty:
                    for idx, row in ref.iterrows():
                        with st.expander(f"Level: {row.get('positionLevels', 'General')}"):
                            st.write(f"**Qual:** {row.get('required_qualification', '-')}")
                            st.write(f"**Skills:** {row.get('skills', '-')}")
                else:
                    st.warning("No skills mapped in skillset.csv for this sector.")
            else:
                st.error("Skillset file missing.")
        
        if not sec_df.empty:
            st.markdown("### 🔍 Raw Data Explorer")
            search_text = st.text_input("Search Job Titles:", placeholder="e.g. 'Analyst' or 'Java'")
            
            display_df = sec_df.copy()
            if search_text:
                display_df = display_df[display_df['title'].str.contains(search_text, case=False)]

            # Remove duplicates based on title
            display_df = display_df.drop_duplicates(subset=['title'], keep='first').reset_index(drop=True)

            rename_map = {
                'title': 'Job Title',
                'positionLevels': 'Position Level',
                'min_exp': 'Min Exp (Years)',
                'average_salary': 'Avg Salary ($)'
            }
            cols_to_select = [c for c in ['title', 'positionLevels', 'min_exp', 'average_salary'] if c in display_df.columns]
            final_df = display_df[cols_to_select].rename(columns=rename_map)
            
            st.dataframe(
                final_df.head(100),
                use_container_width=True,
                column_config={
                    "Avg Salary ($)": st.column_config.NumberColumn(
                        "Avg Salary ($)",
                        format="$%d"
                    )
                },
                hide_index=True  # Hide the pandas index
            )

if __name__ == "__main__":
    main()