import streamlit as st
import plotly.express as px

from chart_style import render_plotly_chart


def calculate_executive_metrics(df):
    return {
        "total_vacancies": df["num_vacancies"].sum(),
        "total_posts": len(df),
        "total_views": df["num_views"].sum(),
        "top_sector_vacancy": df.groupby("category")["num_vacancies"].sum().idxmax(),
        "top_sector_posts": df["category"].value_counts().idxmax(),
        "top_sector_views": df.groupby("category")["num_views"].sum().idxmax(),
    }


def get_top_sectors_data(df, metric="num_vacancies", limit=10):
    df_filtered = df[df["category"] != "Others"]
    if metric == "count":
        data = df_filtered["category"].value_counts().head(limit)
    else:
        data = df_filtered.groupby("category")[metric].sum().sort_values(ascending=False).head(limit)
    return data.reset_index(name="Value")


def render(df):
    st.subheader("High-Level Market Snapshot")
    metrics = calculate_executive_metrics(df)

    kpi1, kpi2, kpi3 = st.columns(3)
    with kpi1:
        st.metric(label="👥 Total Vacancies", value=f"{metrics['total_vacancies']:,.0f}",
                  help=f"Top sector: {metrics['top_sector_vacancy']}")
        st.caption(f"🏆 **Top:** {metrics['top_sector_vacancy']}")
    with kpi2:
        st.metric(label="📝 Total Job Posts", value=f"{metrics['total_posts']:,.0f}",
                  help=f"Top sector: {metrics['top_sector_posts']}")
        st.caption(f"🏆 **Top:** {metrics['top_sector_posts']}")
    with kpi3:
        st.metric(label="👁️ Total Job Views", value=f"{metrics['total_views']:,.0f}",
                  help=f"Top sector: {metrics['top_sector_views']}")
        st.caption(f"🏆 **Top:** {metrics['top_sector_views']}")

    st.divider()

    c_head, c_opt = st.columns([3, 1])
    with c_head:
        st.markdown("#### 📊 Top 10 Sectors Breakdown")
    with c_opt:
        chart_metric = st.selectbox("View By:", ["Vacancies", "Job Posts", "Job Views"], index=0)

    if chart_metric == "Vacancies":
        chart_data = get_top_sectors_data(df, "num_vacancies", 10)
        x_label, bar_color = "Total Vacancies", "#2E86C1"
    elif chart_metric == "Job Posts":
        chart_data = get_top_sectors_data(df, "count", 10)
        x_label, bar_color = "Number of Posts", "#28B463"
    else:
        chart_data = get_top_sectors_data(df, "num_views", 10)
        x_label, bar_color = "Total Views", "#E67E22"

    chart_data = chart_data.sort_values("Value", ascending=True)
    fig = px.bar(
        chart_data,
        x="Value",
        y="category",
        orientation="h",
        title=f"Top 10 Sectors by {chart_metric}",
        labels={"Value": x_label, "category": "Sector"},
        color_discrete_sequence=[bar_color],
    )
    fig.update_traces(
        hovertemplate=f"Sector: %{{y}}<br>{x_label}: %{{x:,.0f}}<extra></extra>"
    )
    render_plotly_chart(fig, key="executive_top_sectors_chart", height=520)
