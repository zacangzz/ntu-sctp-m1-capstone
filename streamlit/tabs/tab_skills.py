import streamlit as st
import plotly.express as px

from chart_style import render_plotly_chart
from data_loader import load_skills_analysis_data


def get_skill_popularity(df):
    """Unique jobs per skill, sorted descending."""
    return df.groupby("skill")["job_id"].nunique().sort_values(ascending=False)


def get_skill_growth(df):
    """Emerging/declining skills: recent 3mo vs previous 3mo growth rate."""
    skill_timeline = df.groupby(["year_month", "skill"])["job_id"].nunique().reset_index()
    skill_timeline.columns = ["year_month", "skill", "unique_jobs"]

    recent_period = skill_timeline["year_month"].max()
    prev_period_end = recent_period - 3
    prev_period_start = recent_period - 5

    recent_3m = skill_timeline[skill_timeline["year_month"] > prev_period_end]
    prev_3m = skill_timeline[
        (skill_timeline["year_month"] > prev_period_start)
        & (skill_timeline["year_month"] <= prev_period_end)
    ]

    recent_counts = recent_3m.groupby("skill")["unique_jobs"].sum()
    prev_counts = prev_3m.groupby("skill")["unique_jobs"].sum()

    common_skills = recent_counts.index.intersection(prev_counts.index)
    growth_rate = (
        (recent_counts[common_skills] - prev_counts[common_skills])
        / prev_counts[common_skills]
        * 100
    )
    growth_rate = growth_rate.sort_values(ascending=False)
    # Filter for skills with at least 20 jobs in previous period
    return growth_rate[prev_counts[growth_rate.index] >= 20]


def get_skill_salary(df):
    """Average salary, job count, and average experience per skill."""
    agg = df.groupby("skill").agg(
        avg_salary=("average_salary_cleaned", "mean"),
        job_count=("job_id", "nunique"),
        avg_exp=("min_exp", "mean"),
    ).reset_index()
    agg = agg.dropna(subset=["avg_salary"])
    # Filter for decent sample size
    return agg[agg["job_count"] >= 100].copy()


def get_universal_skills(df):
    """Cross-category coverage per skill."""
    agg = df.groupby("skill").agg(
        num_categories=("category_name", "nunique"),
        total_jobs=("job_id", "nunique"),
    ).reset_index()
    max_categories = df["category_name"].nunique()
    agg["transferability_score"] = (
        agg["num_categories"] / max_categories
    ) * agg["total_jobs"]
    return agg


def get_category_skills(df):
    """Top skills per category."""
    return (
        df.groupby(["category_name", "skill"])["job_id"]
        .nunique()
        .reset_index()
        .rename(columns={"job_id": "job_count"})
    )


def render():
    st.subheader("🔬 Skills Analysis")
    st.markdown("Objective: Deep-dive into skill-level demand patterns to inform granular curriculum decisions.")

    try:
        sa_df = load_skills_analysis_data()
    except Exception as e:
        st.error(f"Failed to load skills analysis data: {e}")
        st.stop()

    # --- (a) Skill Popularity Overview ---
    st.markdown("#### 📊 Skill Popularity Overview")

    skill_counts = get_skill_popularity(sa_df)

    # 80% demand callout
    sorted_skills = skill_counts.sort_values(ascending=False)
    cumulative_pct = sorted_skills.cumsum() / sorted_skills.sum() * 100
    skills_for_80pct = (cumulative_pct <= 80).sum()
    st.info(
        f"**{skills_for_80pct}** skills (out of {len(skill_counts):,}) "
        f"account for **80%** of total demand."
    )

    col_pop1, col_pop2 = st.columns(2)
    with col_pop1:
        top20 = skill_counts.head(20).reset_index()
        top20.columns = ["skill", "unique_jobs"]
        fig_top = px.bar(
            top20, x="unique_jobs", y="skill", orientation="h",
            title="Top 20 Most In-Demand Skills",
            labels={"unique_jobs": "Unique Job Postings", "skill": "Skill"},
            color_discrete_sequence=["#2E86C1"],
        )
        fig_top.update_layout(yaxis=dict(autorange="reversed"), height=600)
        render_plotly_chart(fig_top, key="top20_skills")

    with col_pop2:
        bottom20 = skill_counts.tail(20).sort_values(ascending=True).reset_index()
        bottom20.columns = ["skill", "unique_jobs"]
        fig_bot = px.bar(
            bottom20, x="unique_jobs", y="skill", orientation="h",
            title="Bottom 20 Least In-Demand Skills",
            labels={"unique_jobs": "Unique Job Postings", "skill": "Skill"},
            color_discrete_sequence=["#E74C3C"],
        )
        fig_bot.update_layout(yaxis=dict(autorange="reversed"), height=600)
        render_plotly_chart(fig_bot, key="bottom20_skills")

    st.divider()

    # --- (b) Emerging & Declining Skills ---
    st.markdown("#### 📈 Emerging & Declining Skills")
    st.caption("Growth rate: recent 3 months vs previous 3 months. Minimum 20 jobs in prior period.")

    growth_rate = get_skill_growth(sa_df)

    col_em, col_de = st.columns(2)
    with col_em:
        emerging = growth_rate.head(15).reset_index()
        emerging.columns = ["skill", "growth_pct"]
        fig_emerging = px.bar(
            emerging, x="growth_pct", y="skill", orientation="h",
            title="Top 15 Emerging Skills",
            labels={"growth_pct": "Growth Rate (%)", "skill": "Skill"},
            color_discrete_sequence=["#28B463"],
        )
        fig_emerging.update_layout(yaxis=dict(autorange="reversed"), height=500)
        render_plotly_chart(fig_emerging, key="emerging_skills")

    with col_de:
        declining = growth_rate.tail(15).sort_values(ascending=True).reset_index()
        declining.columns = ["skill", "growth_pct"]
        fig_declining = px.bar(
            declining, x="growth_pct", y="skill", orientation="h",
            title="Top 15 Declining Skills",
            labels={"growth_pct": "Growth Rate (%)", "skill": "Skill"},
            color_discrete_sequence=["#E74C3C"],
        )
        fig_declining.update_layout(yaxis=dict(autorange="reversed"), height=500)
        render_plotly_chart(fig_declining, key="declining_skills")

    st.divider()

    # --- (c) Premium Skills (Salary Analysis) ---
    st.markdown("#### 💰 Premium Skills (Salary Analysis)")
    st.caption("Skills with at least 100 unique job postings.")

    skill_salary = get_skill_salary(sa_df)

    # Top 25 highest paying
    top_premium = skill_salary.nlargest(25, "avg_salary")
    fig_premium = px.bar(
        top_premium, x="avg_salary", y="skill", orientation="h",
        title="Top 25 Highest Paying Skills",
        labels={"avg_salary": "Average Salary (SGD)", "skill": "Skill"},
        color_discrete_sequence=["#F39C12"],
    )
    fig_premium.update_layout(yaxis=dict(autorange="reversed"), height=700)
    render_plotly_chart(fig_premium, key="premium_skills")

    # Salary vs Popularity bubble chart
    st.markdown("##### Salary vs Popularity")
    fig_bubble = px.scatter(
        skill_salary, x="job_count", y="avg_salary", size="avg_exp",
        hover_name="skill",
        title="Salary vs Popularity (Bubble Size = Avg Experience Required)",
        labels={
            "job_count": "Unique Job Postings",
            "avg_salary": "Average Salary (SGD)",
            "avg_exp": "Avg Experience (Years)",
        },
        color_discrete_sequence=["#3498DB"],
    )
    fig_bubble.update_layout(height=600)
    render_plotly_chart(fig_bubble, key="salary_vs_popularity")

    # High-value skills: high salary + low experience
    median_salary = skill_salary["avg_salary"].median()
    median_exp = skill_salary["avg_exp"].median()
    high_value = skill_salary[
        (skill_salary["avg_salary"] > median_salary)
        & (skill_salary["avg_exp"] < median_exp)
    ].sort_values("avg_salary", ascending=False).head(20)

    fig_hv = px.bar(
        high_value, x="avg_salary", y="skill", orientation="h",
        title="Top 20 High-Value Skills (Above-Median Salary, Below-Median Experience)",
        labels={"avg_salary": "Average Salary (SGD)", "skill": "Skill"},
        color_discrete_sequence=["#8E44AD"],
    )
    fig_hv.update_layout(yaxis=dict(autorange="reversed"), height=600)
    render_plotly_chart(fig_hv, key="high_value_skills")

    st.divider()

    # --- (d) Universal / Transferable Skills ---
    st.markdown("#### 🌐 Universal / Transferable Skills")

    universal = get_universal_skills(sa_df)

    col_uni1, col_uni2 = st.columns(2)
    with col_uni1:
        top_universal = universal.nlargest(25, "num_categories")
        fig_uni = px.bar(
            top_universal, x="num_categories", y="skill", orientation="h",
            title="Top 25 Most Universal Skills (by Category Count)",
            labels={"num_categories": "Number of Categories", "skill": "Skill"},
            color_discrete_sequence=["#1ABC9C"],
        )
        fig_uni.update_layout(yaxis=dict(autorange="reversed"), height=700)
        render_plotly_chart(fig_uni, key="universal_skills")

    with col_uni2:
        top_transferable = universal.nlargest(25, "transferability_score")
        fig_trans = px.bar(
            top_transferable, x="transferability_score", y="skill", orientation="h",
            title="Top 25 Most Transferable Skills (Coverage × Demand)",
            labels={"transferability_score": "Transferability Score", "skill": "Skill"},
            color_discrete_sequence=["#2980B9"],
        )
        fig_trans.update_layout(yaxis=dict(autorange="reversed"), height=700)
        render_plotly_chart(fig_trans, key="transferable_skills")

    st.divider()

    # --- (e) Skills by Category ---
    st.markdown("#### 🏭 Skills by Category")

    category_skills_df = get_category_skills(sa_df)
    all_categories = sorted(
        category_skills_df["category_name"].dropna().unique().tolist()
    )
    selected_category = st.selectbox(
        "Select Sector:", all_categories, key="skills_category_selector"
    )

    cat_data = (
        category_skills_df[category_skills_df["category_name"] == selected_category]
        .nlargest(10, "job_count")
    )
    fig_cat = px.bar(
        cat_data, x="job_count", y="skill", orientation="h",
        title=f"Top 10 Skills in {selected_category}",
        labels={"job_count": "Unique Job Postings", "skill": "Skill"},
        color_discrete_sequence=["#E67E22"],
    )
    fig_cat.update_layout(yaxis=dict(autorange="reversed"), height=450)
    render_plotly_chart(fig_cat, key="category_skills_chart")
