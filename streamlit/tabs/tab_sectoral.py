import streamlit as st
import plotly.express as px

def get_demand_velocity(df):
    velocity_df = df[df["category"] != "Others"]
    top_10 = velocity_df.groupby("category")["num_vacancies"].sum().nlargest(10).index
    velocity_df = velocity_df[velocity_df["category"].isin(top_10)]
    agg_df = velocity_df.groupby(["month_year", "category"]).agg(
        num_applications=("num_applications", "sum"),
        num_vacancies=("num_vacancies", "sum"),
    ).reset_index()
    agg_df["bulk_factor"] = agg_df.apply(
        lambda x: x["num_applications"] / x["num_vacancies"] if x["num_vacancies"] > 0 else 0,
        axis=1,
    )
    return agg_df


def get_bulk_hiring_data(df):
    bulk_df = df[df["category"] != "Others"]
    top_sectors = bulk_df.groupby("category")["num_vacancies"].sum().nlargest(12).index
    bulk_filtered = bulk_df[bulk_df["category"].isin(top_sectors)]
    apps = bulk_filtered.pivot_table(index="category", columns="month_year",
                                     values="num_applications", aggfunc="sum").fillna(0)
    vacs = bulk_filtered.pivot_table(index="category", columns="month_year",
                                     values="num_vacancies", aggfunc="sum").fillna(0)
    return (apps / vacs.replace(0, 1)).fillna(0)


def render(df):
    st.subheader("🏭 Sectoral Demand & Momentum")
    st.markdown("Objective: Identify \"What\" to teach by tracking the velocity of industry needs.")

    st.markdown("#### 📈 Demand Velocity (Bulk Factor)")
    st.caption("Bulk Factor = Applications ÷ Vacancies. Higher values indicate stronger competition.")

    velocity_df = get_demand_velocity(df)
    if len(velocity_df) > 1:
        fig_vel = px.line(velocity_df, x="month_year", y="bulk_factor", color="category",
                          markers=True, line_shape="spline",
                          title="Top 10 Sectors: Bulk Factor Trend Over Time",
                          labels={"month_year": "Posting Date",
                                  "bulk_factor": "Bulk Factor (Apps/Vacancies)",
                                  "category": "Sector"})
        st.plotly_chart(fig_vel, use_container_width=True, key="demand_velocity_chart")
    else:
        st.warning("Not enough data points for time-series velocity.")

    st.markdown("#### 🗺️ Bulk Hiring Map")
    st.caption("Competition intensity by sector and time. Darker = higher bulk factor.")

    bulk_pivot = get_bulk_hiring_data(df)
    fig_bulk = px.imshow(bulk_pivot, aspect="auto", color_continuous_scale="YlOrRd",
                         labels=dict(x="Month", y="Sector", color="Bulk Factor"))
    st.plotly_chart(fig_bulk, use_container_width=True, key="bulk_hiring_map")
