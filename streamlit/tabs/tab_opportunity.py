import streamlit as st
import pandas as pd
import plotly.express as px

from chart_style import render_plotly_chart
from data_loader import load_withskills_data


def get_education_metrics(df):
    metrics = df.groupby("category").agg(
        num_vacancies=("num_vacancies", "sum"),
        num_applications=("num_applications", "sum"),
        min_exp=("min_exp", "mean"),
        job_id=("job_id", "count"),
    ).reset_index()
    metrics["opp_score"] = metrics["num_vacancies"] / (metrics["min_exp"] + 1)
    metrics["comp_index"] = metrics.apply(
        lambda x: x["num_applications"] / x["num_vacancies"] if x["num_vacancies"] > 0 else 0,
        axis=1,
    )
    return metrics


def render(df):
    st.subheader("🎓 Opportunity")
    st.markdown('Objective: Identify "Blue Ocean" opportunities where job matching rates are highest.')

    p2_metrics = get_education_metrics(df)

    st.markdown("#### Supply vs Demand")
    st.caption("Treemap: Rectangle size = Vacancies (demand), Color = Applications (supply).")
    supply_demand = p2_metrics[p2_metrics["category"] != "Others"].copy()
    supply_demand = supply_demand.sort_values("num_vacancies", ascending=False).head(20)

    fig_supply_demand = px.treemap(
        supply_demand, path=[px.Constant("All Sectors"), "category"],
        values="num_vacancies", color="num_applications",
        color_continuous_scale="RdYlGn_r",
        labels={"num_vacancies": "Vacancies (Size)", "num_applications": "Applications (Color)"},
        title="Supply vs Demand Treemap",
        hover_data=["num_vacancies", "num_applications"],
    )
    render_plotly_chart(fig_supply_demand, key="supply_demand_treemap", height=560)

    # Hidden Demand Quadrant Analysis
    st.markdown('#### The "Hidden Demand"')
    st.caption("Quadrant analysis: High vacancies + Low applications = Hidden opportunities.")

    analysis_type = st.selectbox(
        "Analyze By:", ["Industry", "Job Title", "Skills"],
        key="hidden_demand_analysis_type",
        help="Choose the dimension for quadrant analysis",
    )

    if analysis_type == "Industry":
        hd_metrics = df.groupby("category").agg(
            num_vacancies=("num_vacancies", "sum"),
            num_applications=("num_applications", "sum"),
            min_exp=("min_exp", "mean"),
            job_id=("job_id", "count"),
        ).reset_index().rename(columns={"category": "name"})
        chart_title = "Hidden Demand Quadrant Analysis by Industry"
        hover_label = "Industry"

    elif analysis_type == "Job Title":
        ws = load_withskills_data()
        title_map = ws[["job_id", "jobtitle_cleaned"]].drop_duplicates()
        df_titled = df.merge(title_map, on="job_id", how="inner")
        hd_metrics = df_titled.groupby("jobtitle_cleaned").agg(
            num_vacancies=("num_vacancies", "sum"),
            num_applications=("num_applications", "sum"),
            min_exp=("min_exp", "mean"),
            job_id=("job_id", "count"),
        ).reset_index().rename(columns={"jobtitle_cleaned": "name"})
        chart_title = "Hidden Demand Quadrant Analysis by Job Title"
        hover_label = "Job Title"

    else:  # Skills
        try:
            ws = load_withskills_data()
            skills_with_data = ws.merge(
                df[["job_id", "num_vacancies", "num_applications", "min_exp"]].drop_duplicates("job_id"),
                on="job_id", how="left",
            )
            hd_metrics = skills_with_data.groupby("skill").agg(
                num_vacancies=("num_vacancies", "sum"),
                num_applications=("num_applications", "sum"),
                min_exp=("min_exp", "mean"),
                job_id=("job_id", "count"),
            ).reset_index().rename(columns={"skill": "name"})
            chart_title = "Hidden Demand Quadrant Analysis by Skills"
            hover_label = "Skill"
        except Exception as e:
            st.error(f"Failed to load skills data: {e}")
            hd_metrics = pd.DataFrame()

    if not hd_metrics.empty:
        hd_metrics["opp_score"] = hd_metrics["num_vacancies"] / (hd_metrics["min_exp"] + 1)
        hd_metrics["comp_index"] = hd_metrics.apply(
            lambda x: x["num_applications"] / x["num_vacancies"] if x["num_vacancies"] > 0 else 0,
            axis=1,
        )

    hidden_demand = hd_metrics.copy() if not hd_metrics.empty else pd.DataFrame()

    if len(hidden_demand) > 0:
        sample_size = 50 if analysis_type in ["Job Title", "Skills"] else len(hidden_demand)
        if len(hidden_demand) > sample_size:
            hidden_demand = hidden_demand.nlargest(sample_size, "num_vacancies")

        median_vac = hidden_demand["num_vacancies"].median()
        median_app = hidden_demand["num_applications"].median()

        def assign_quadrant(row):
            if row["num_vacancies"] >= median_vac and row["num_applications"] < median_app:
                return "Hidden Opportunity"
            elif row["num_vacancies"] >= median_vac and row["num_applications"] >= median_app:
                return "Competitive Market"
            elif row["num_vacancies"] < median_vac and row["num_applications"] < median_app:
                return "Niche Market"
            else:
                return "Oversupplied"

        hidden_demand["quadrant"] = hidden_demand.apply(assign_quadrant, axis=1)
        hidden_demand["display_text"] = hidden_demand.apply(
            lambda row: "" if row["quadrant"] == "Niche Market" else row["name"], axis=1
        )

        fig_hidden = px.scatter(
            hidden_demand, x="num_vacancies", y="num_applications",
            size="num_vacancies", color="quadrant",
            hover_name="name", text="display_text",
            labels={"num_vacancies": "Vacancies", "num_applications": "Applications",
                     "name": hover_label},
            title=chart_title,
            color_discrete_map={
                "Hidden Opportunity": "#28B463",
                "Competitive Market": "#E67E22",
                "Niche Market": "#95A5A6",
                "Oversupplied": "#E74C3C",
            },
        )
        fig_hidden.update_traces(textposition="top center", textfont_size=8)
        fig_hidden.add_hline(y=median_app, line_dash="dash", line_color="gray",
                             annotation_text=f"Median Apps: {median_app:.0f}")
        fig_hidden.add_vline(x=median_vac, line_dash="dash", line_color="gray",
                             annotation_text=f"Median Vacancies: {median_vac:.0f}")
        fig_hidden.update_layout(height=600)
        render_plotly_chart(fig_hidden, key="hidden_demand_chart")
