import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import re
import os
from datetime import datetime

# ==========================================
# 1. CONFIGURATION & STYLING
# ==========================================
st.set_page_config(page_title="Workforce Intelligence Portal", layout="wide", page_icon="📊")

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
    DATA_FILE = 'data/SGJobData.csv' 
    SKILL_FILE = 'data/skillset.csv'
    CACHE_TTL = 3600

# ==========================================
# 2. DATA PROCESSING ENGINE
# ==========================================
class DataProcessor:
    @staticmethod
    def clean_job_title(title):
        """
        Improved rule-based cleaning that preserves complete job titles
        """
        if not isinstance(title, str):
            return "Unknown Title"
        
        # Remove location information (after | or -)
        title = re.split(r'[-|]', title)[0].strip()
        
        # Remove salary information
        title = re.sub(r'\$[\d,\.]+[kK]?', '', title)
        title = re.sub(r'up to \$?[\d,\.]+[kK]?', '', title)
        title = re.sub(r'basic \$?[\d,\.]+[kK]?', '', title)
        
        # Remove parentheses content (skills/requirements)
        title = re.sub(r'\([^)]*\)', '', title)
        
        # Remove urgency markers
        title = re.sub(r'urgent hiring!!!?', '', title, flags=re.IGNORECASE)
        title = re.sub(r'immediate', '', title, flags=re.IGNORECASE)
        
        # Remove hashtags and extra symbols
        title = re.sub(r'#\w+', '', title)
        title = re.sub(r'[|/]', ' ', title)
        
        # Remove common noise words but keep important job title components
        noise_words = {
            'urgent', 'hiring', 'entry', 'level', 'immediate', 'available', 
            'position', 'role', 'job', 'career', 'opportunity', 'full', 'time', 
            'part', 'permanent', 'contract', 'temporary', 'up', 'to', '$',
            'basic', 'salary', 'plus', 'commission', 'bonus', 'benefits', 'package'
        }
        
        # Tokenize and clean, but be more conservative
        words = title.lower().split()
        meaningful_words = []
        
        for word in words:
            word = word.strip('.,!?()[]{}:;"\'')
            if (word.isalpha() and 
                len(word) > 1 and 
                word not in noise_words):
                meaningful_words.append(word)
        
        # Reconstruct and capitalize properly
        if meaningful_words:
            cleaned_title = ' '.join(meaningful_words)
            # Capitalize first letter of each word for job titles
            cleaned_title = ' '.join(word.capitalize() for word in cleaned_title.split())
            return cleaned_title
        else:
            return title.strip()

    @staticmethod
    def parse_categories(cat_str):
        try:
            if isinstance(cat_str, str):
                return json.loads(cat_str.replace("'", '"'))
            return []
        except:
            return []

    @staticmethod
    def extract_category_name(val):
        if isinstance(val, dict):
            return val.get('category', val.get('name', str(val)))
        return str(val)

    @staticmethod
    def remove_outliers(df, col):
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return np.where(df[col] > upper_bound, upper_bound, 
                        np.where(df[col] < lower_bound, lower_bound, df[col]))

    @staticmethod
    @st.cache_data(ttl=Config.CACHE_TTL)
    def load_and_clean_data():
        # 1. Load Data
        if os.path.exists(Config.DATA_FILE):
             df = pd.read_csv(Config.DATA_FILE)
        elif os.path.exists(Config.DATA_FILE + '.xz'):
             df = pd.read_csv(Config.DATA_FILE + '.xz')
        else:
            st.error(f"🚨 Data file not found at `{Config.DATA_FILE}`.")
            st.stop()
        
        # 2. Drop rows with missing categories
        df.dropna(subset=['categories'], inplace=True)
        
        # 3. Explode Categories
        df['parsed_categories'] = df['categories'].apply(DataProcessor.parse_categories)
        df_exploded = df.explode('parsed_categories')
        df_exploded['category_name'] = df_exploded['parsed_categories'].apply(DataProcessor.extract_category_name)
        
        # 4. Clean Titles
        df_exploded['clean_title'] = df_exploded['title'].apply(DataProcessor.clean_job_title)
        
        # 5. Type Conversion
        numeric_cols = ['minimumYearsExperience', 'numberOfVacancies', 'salary_minimum', 'salary_maximum']
        for col in numeric_cols:
            if col in df_exploded.columns:
                df_exploded[col] = pd.to_numeric(df_exploded[col], errors='coerce')
        
        df_exploded['numberOfVacancies'].fillna(1, inplace=True)
        df_exploded['minimumYearsExperience'].fillna(0, inplace=True)

        # 6. Salary Calculation
        if 'salary_minimum' in df_exploded.columns and 'salary_maximum' in df_exploded.columns:
            valid_salary = (df_exploded['salary_minimum'] > 0) & (df_exploded['salary_maximum'] > 0)
            df_exploded = df_exploded[valid_salary]
            df_exploded['avg_salary'] = (df_exploded['salary_minimum'] + df_exploded['salary_maximum']) / 2
        else:
            df_exploded['avg_salary'] = 0

        # 7. Outlier Handling
        df_exploded['minimumYearsExperience'] = np.clip(df_exploded['minimumYearsExperience'], 0, 15)
        if 'avg_salary' in df_exploded.columns and not df_exploded.empty:
            df_exploded['avg_salary'] = DataProcessor.remove_outliers(df_exploded, 'avg_salary')
            
        # 8. Date Parsing
        date_col = None
        for col in ['metadata_newPostingDate', 'metadata_createdAt', 'postingDate']:
            if col in df_exploded.columns:
                date_col = col
                break
        
        if date_col:
            df_exploded['posting_date'] = pd.to_datetime(df_exploded[date_col], errors='coerce')
            df_exploded['month_year'] = df_exploded['posting_date'].dt.to_period('M').dt.to_timestamp()
            df_exploded.dropna(subset=['posting_date'], inplace=True)

        # 9. Load Skillset
        if os.path.exists(Config.SKILL_FILE):
            skills_df = pd.read_csv(Config.SKILL_FILE)
        else:
            skills_df = pd.DataFrame()

        return df_exploded, skills_df

# ==========================================
# 3. INSIGHT ENGINE
# ==========================================
class InsightEngine:
    @staticmethod
    def get_market_insights(df, metric_col, group_col):
        if df.empty: return ["No Data", "No Data"], ["No Data", "No Data"]
        grouped = df.groupby(group_col)[metric_col].sum().sort_values(ascending=False)
        top = grouped.head(1)
        bottom = grouped.tail(1)
        
        finding1 = f"**Dominant Player:** The **{top.index[0]}** sector leads with **{top.values[0]:,.0f}** vacancies."
        finding2 = f"**Niche Market:** **{bottom.index[0]}** shows the lowest activity with **{bottom.values[0]:,.0f}** vacancies."
        
        insight1 = f"The concentration of demand in {top.index[0]} suggests a mature market reaching saturation."
        insight2 = f"Lower volume in {bottom.index[0]} may indicate a specialized niche or declining sector."
        return [finding1, finding2], [insight1, insight2]

    @staticmethod
    def get_salary_insights(df):
        if df.empty: return ["No Data", "No Data"], ["No Data", "No Data"]
        avg_sal = df.groupby('category_name')['avg_salary'].mean().sort_values(ascending=False)
        top_pay = avg_sal.head(1)
        
        finding1 = f"**Highest Payer:** **{top_pay.index[0]}** offers the best average monthly compensation at **${top_pay.values[0]:,.0f}**."
        finding2 = f"**Salary Spread:** **{(avg_sal.max() - avg_sal.min()):,.0f} SGD** gap between highest and lowest paying sectors."
        
        insight1 = "High starting salaries in tech/finance reflect acute talent shortages."
        insight2 = "Lower salary sectors might offer other non-monetary benefits to emphasize."
        return [finding1, finding2], [insight1, insight2]

    @staticmethod
    def get_heatmap_insights(pivot_df):
        if pivot_df.empty: return ["No Data", "No Data"], ["No Data", "No Data"]
        
        max_val = pivot_df.max().max()
        max_col = pivot_df.max().idxmax()
        max_row = pivot_df[max_col].idxmax()
        
        finding1 = f"**Top Premium Role:** The **{max_col}** role in **{max_row}** commands the highest specific average salary of **${max_val:,.0f}**."
        finding2 = "**Overlap Gaps:** Empty or light spots in the heatmap indicate roles that are not commonly defined in those specific sectors."
        
        insight1 = "Target curriculum development towards these high-value intersections (Specific Role + Specific Sector)."
        insight2 = "Encourage students to specialize; generalist roles often show lower color intensity (lower pay) across the board."
        return [finding1, finding2], [insight1, insight2]

    @staticmethod
    def get_momentum_insights(growth_df):
        if growth_df.empty:
            return ["No data available for trend analysis.", ""], ["-", "-"]
            
        top_grower = growth_df.iloc[0]
        bottom_grower = growth_df.iloc[-1]
        
        finding1 = f"**Fastest Mover:** **{top_grower['category_name']}** is accelerating with a **{top_grower['mom_growth']:.1f}%** recent growth rate."
        finding2 = f"**Cooling Down:** **{bottom_grower['category_name']}** has seen a dip/slowdown of **{bottom_grower['mom_growth']:.1f}%**."
        
        insight1 = f"Investigate {top_grower['category_name']} for emerging skills needs; rapid growth often creates curriculum gaps."
        insight2 = f"Declining momentum in {bottom_grower['category_name']} might signal market saturation or seasonality."
        
        return [finding1, finding2], [insight1, insight2]

def render_findings_box(findings, insights):
    def format_bold(text):
        return re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)

    f1 = format_bold(findings[0])
    f2 = format_bold(findings[1])
    i1 = format_bold(insights[0])
    i2 = format_bold(insights[1])

    st.markdown(f"""
    <div class="insight-box">
        <div class="finding-title">🔍 Key Findings</div>
        <div class="insight-text">
            <ul>
                <li>{f1}</li>
                <li>{f2}</li>
            </ul>
        </div>
        <div class="finding-title">💡 Strategic Insights</div>
        <div class="insight-text">
            <ul>
                <li>{i1}</li>
                <li>{i2}</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# 4. MAIN APP
# ==========================================
def main():
    # Load
    with st.spinner("Loading and Cleansing Data..."):
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

    st.title("🎓 Workforce Intelligence Portal")
    st.markdown("Aligning Curriculum with Real-Time Market Structure")
    st.write(f"**Data Period:** {period_str}")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Executive Summary", "🚀 Market Demand", "💰 Salary & Value", "📈 Market Momentum", "📋 Curriculum Deep-Dive"
    ])

    # --- TAB 1: EXECUTIVE ---
    with tab1:
        st.subheader("High-Level Market Snapshot")

        # -----------------------------------------------------------
        # 1. DATA PREPARATION (METRICS)
        # -----------------------------------------------------------
        
        # 1.1 Ensure View Column Exists & is Numeric
        view_col = 'metadata_totalNumberOfView'
        
        if view_col not in df.columns:
            df[view_col] = 0 # Create dummy if missing
            st.toast(f"⚠️ Column `{view_col}` not found. Views set to 0.", icon="ℹ️")
        else:
            # Force convert to numeric, turning errors/blanks into 0
            df[view_col] = pd.to_numeric(df[view_col], errors='coerce').fillna(0)

        # 1.2 HELPER: Calculate Stats for a specific metric
        def get_kpi_stats(df, count_method='sum', value_col='numberOfVacancies'):
            """
            count_method: 'sum' (for vacancies/views) or 'count' (for job posts)
            value_col: The column to sum (e.g., numberOfVacancies)
            """
            if count_method == 'sum':
                total_val = df[value_col].sum()
                # Top sector by sum of that value
                top_sector = df.groupby('category_name')[value_col].sum().idxmax()
                
                # Weighted Average Salary (More accurate for Vacancies/Views)
                if total_val > 0:
                    avg_sal = (df['avg_salary'] * df[value_col]).sum() / total_val
                else:
                    avg_sal = 0
            else: # 'count' (Job Posts)
                total_val = len(df)
                # Top sector by count of rows
                top_sector = df['category_name'].value_counts().idxmax()
                # Simple Average Salary
                avg_sal = df['avg_salary'].mean()

            return total_val, top_sector, avg_sal

        # CALC 1: Total Vacancies
        vac_total, vac_top, vac_sal = get_kpi_stats(df, 'sum', 'numberOfVacancies')
        
        # CALC 2: Total Job Posted (Rows)
        post_total, post_top, post_sal = get_kpi_stats(df, 'count', 'clean_title')
        
        # CALC 3: Total Job Views (Using metadata_totalNumberOfView)
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
                    <b>🏆 Top:</b> {vac_top}<br>
                    <b>💰 Avg Sal:</b> ${vac_sal:,.0f}
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
                    <b>🏆 Top:</b> {post_top}<br>
                    <b>💰 Avg Sal:</b> ${post_sal:,.0f}
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
                    <b>🏆 Top:</b> {view_top}<br>
                    <b>💰 Avg Sal:</b> ${view_sal:,.0f}
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
            metric_col = 'numberOfVacancies'
            agg_func = 'sum'
            bar_color = '#2E86C1' # Blue
            x_label = 'Total Vacancies'
        elif chart_metric == "Job Posts":
            metric_col = 'category_name' # Just for counting
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
            top_sectors_chart = df.groupby('category_name')[metric_col].sum().sort_values(ascending=False).head(10).reset_index()
            top_sectors_chart.columns = ['category_name', 'Value']
        else:
            top_sectors_chart = df['category_name'].value_counts().head(10).reset_index()
            top_sectors_chart.columns = ['category_name', 'Value']

        # Render Chart
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=top_sectors_chart['category_name'],
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
        st.plotly_chart(fig, use_container_width=True)

        # Insights Footer
        # f, i = InsightEngine.get_market_insights(df, 'numberOfVacancies', 'category_name')
        # render_findings_box(f, i)

    # --- TAB 2: DEMAND ---
    with tab2:
        st.subheader("Structural Market Demand")
        c1, c2 = st.columns(2)
        
        with c1:
            st.markdown("#### 📦 Bulk Hiring Map")
            st.caption("Size = Total Vacancies | Color = Vacancies per Post")
            
            tree_data = df.groupby('category_name').agg(
                total_vacancies=('numberOfVacancies', 'sum'),
                post_count=('category_name', 'count')
            ).reset_index()
            tree_data['bulk_factor'] = tree_data['total_vacancies'] / tree_data['post_count']
            
            fig_tree = px.treemap(tree_data, path=['category_name'], values='total_vacancies',
                                  color='bulk_factor', color_continuous_scale='Blues')
            st.plotly_chart(fig_tree, use_container_width=True)
            
            if not tree_data.empty:
                leader = tree_data.sort_values('bulk_factor', ascending=False).iloc[0]['category_name']
                render_findings_box(
                    [f"**Bulk Leader:** {leader} has highest hiring efficiency.", "Fragmented sectors rely on single-headcount replacements."],
                    ["High bulk factors suggest 'Class-to-Job' partnership opportunities.", "Low bulk factors imply need for broad networking."]
                )

        with c2:
            st.markdown("#### 🏷️ Top Job Titles")
            top_titles = df['clean_title'].value_counts().head(10).reset_index()
            top_titles.columns = ['Job Title', 'Count']
            
            fig_bar = px.bar(top_titles, x='Count', y='Job Title', orientation='h', color='Count')
            fig_bar.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_bar, use_container_width=True)
            
            if not top_titles.empty:
                render_findings_box(
                    [f"'{top_titles.iloc[0]['Job Title']}' is the most requested title.", "Standardized titles dominate."],
                    ["Map curriculum modules to these specific keywords.", "Generic titles imply need for soft skills."]
                )

    # --- TAB 3: SALARY ---
    with tab3:
        st.subheader("Compensation & ROI")
        
        st.markdown("#### 🫧 Experience vs. Compensation Matrix")
        bubble_data = df.groupby('category_name').agg(
            avg_exp=('minimumYearsExperience', 'mean'),
            avg_sal=('avg_salary', 'mean'),
            volume=('numberOfVacancies', 'sum')
        ).reset_index()
        
        fig_bub = px.scatter(bubble_data, x='avg_exp', y='avg_sal', size='volume', color='category_name',
                             hover_name='category_name', size_max=60,
                             labels={'avg_exp': 'Avg Min Experience (Years)', 'avg_sal': 'Avg Salary (SGD)'})
        st.plotly_chart(fig_bub, use_container_width=True)
        
        fs, isi = InsightEngine.get_salary_insights(df)
        render_findings_box(fs, isi)
        
        st.divider()
        st.markdown("#### 🔥 Salary Heatmap (Top Sectors vs Titles)")
        
        top_cats = df['category_name'].value_counts().head(8).index
        top_ts = df['clean_title'].value_counts().head(8).index
        heatmap_df = df[df['category_name'].isin(top_cats) & df['clean_title'].isin(top_ts)]
        
        if not heatmap_df.empty:
            pivot = heatmap_df.pivot_table(index='category_name', columns='clean_title', values='avg_salary', aggfunc='mean')
            fig_heat = px.imshow(pivot, labels=dict(x="Job Title", y="Sector", color="Salary"),
                                 color_continuous_scale='RdBu_r', aspect="auto")
            st.plotly_chart(fig_heat, use_container_width=True)
            
            hm_f, hm_i = InsightEngine.get_heatmap_insights(pivot)
            render_findings_box(hm_f, hm_i)

    # --- TAB 4: MARKET MOMENTUM ---
    with tab4:
        st.subheader("Market Momentum & Trends")
        st.markdown("Identifying emerging sectors, seasonal peaks, and growth velocity.")
        
        if 'month_year' in df.columns:
            trend_data = df.groupby(['month_year', 'category_name'])['numberOfVacancies'].sum().reset_index()
            top_sectors = df.groupby('category_name')['numberOfVacancies'].sum().nlargest(8).index.tolist()
            filtered_trend = trend_data[trend_data['category_name'].isin(top_sectors)]
            
            st.markdown("#### 📈 Historical Vacancy Volume (Top 8 Sectors)")
            fig_line = px.line(filtered_trend, x='month_year', y='numberOfVacancies', color='category_name',
                               title="Vacancy Trends over Time", markers=True,
                               labels={
                                   'month_year': 'Date',
                                   'numberOfVacancies': 'Vacancy Volume',
                                   'category_name': 'Industry'
                               })
            st.plotly_chart(fig_line, use_container_width=True)
            
            st.divider()
            m1, m2 = st.columns(2)
            
            with m1:
                st.markdown("#### 🔥 Seasonal Hiring Heatmap")
                st.caption("Identify peak hiring months. Darker colors = Higher demand.")
                filtered_trend['Month'] = filtered_trend['month_year'].dt.strftime('%Y-%m')
                heatmap_data = filtered_trend.pivot_table(index='category_name', columns='Month', values='numberOfVacancies', aggfunc='sum').fillna(0)
                fig_hm = px.imshow(heatmap_data, labels=dict(x="Month", y="Industry", color="Vacancies"),
                                   color_continuous_scale='Magma_r', aspect="auto")
                st.plotly_chart(fig_hm, use_container_width=True)

            with m2:
                st.markdown("#### 🚀 Momentum Leaderboard (Recent Growth)")
                st.caption("Which sectors are accelerating? (Avg Month-on-Month Growth, Last Quarter)")
                filtered_trend = filtered_trend.sort_values(['category_name', 'month_year'])
                filtered_trend['mom_growth'] = filtered_trend.groupby('category_name')['numberOfVacancies'].pct_change() * 100
                recent_growth = filtered_trend.groupby('category_name').tail(3).groupby('category_name')['mom_growth'].mean().reset_index()
                recent_growth = recent_growth.sort_values('mom_growth', ascending=False)
                
                fig_mom = px.bar(recent_growth, x='mom_growth', y='category_name', orientation='h',
                                 title="Avg MoM Growth Rate (Last 3 Months)",
                                 labels={'mom_growth': 'Growth Rate (%)', 'category_name': 'Industry'},
                                 color='mom_growth', color_continuous_scale='Tealgrn')
                fig_mom.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig_mom, use_container_width=True)

            fm, im = InsightEngine.get_momentum_insights(recent_growth)
            render_findings_box(fm, im)
            
        else:
            st.warning("No time-series data available. Check 'posting_date' parsing.")

    # --- TAB 5: CURRICULUM ---
    with tab5:
        st.subheader("📋 Curriculum Alignment Tool")
        
        sectors = sorted(df['category_name'].astype(str).unique())
        selected_sector = st.selectbox("Select Target Sector:", sectors)
        
        col_c1, col_c2 = st.columns(2)
        sec_df = df[df['category_name'] == selected_sector]

        with col_c1:
            st.info(f"📊 Live Market Stats: **{selected_sector}**")
            if not sec_df.empty:
                st.write(f"**Total Vacancies:** {sec_df['numberOfVacancies'].sum():,}")
                entry_sal = sec_df[sec_df['minimumYearsExperience']<=2]['avg_salary'].mean()
                st.write(f"**Avg Entry Salary:** ${entry_sal:,.0f}" if not pd.isna(entry_sal) else "N/A")
                st.markdown("**Top Job Title Keywords:**")
                st.bar_chart(sec_df['clean_title'].value_counts().head(5))
            else:
                st.warning("No data found for this sector.")

        with col_c2:
            st.success(f"📚 Reference Curriculum: **{selected_sector}**")
            if not skills_df.empty:
                ref = skills_df[skills_df['category_name'] == selected_sector]
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
                display_df = display_df[display_df['clean_title'].str.contains(search_text, case=False)]
            
            # Remove duplicates based on clean_title
            display_df = display_df.drop_duplicates(subset=['clean_title'], keep='first').reset_index(drop=True)

            rename_map = {
                'clean_title': 'Job Title', 
                'positionLevels': 'Position Level',
                'minimumYearsExperience': 'Min Exp (Years)',
                'avg_salary': 'Avg Salary ($)'
            }
            cols_to_select = [c for c in ['clean_title', 'positionLevels', 'minimumYearsExperience', 'avg_salary'] if c in display_df.columns]
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