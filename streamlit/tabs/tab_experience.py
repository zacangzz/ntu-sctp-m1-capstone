import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

EXP_BINS = [0, 2, 5, 10, float("inf")]
EXP_LABELS = ["0-2 yrs (Entry/Junior)", ">2-5 yrs (Mid-Level)",
              ">5-10 yrs (Senior)", "10+ yrs (Expert)"]


def _prepare_experience_df(df, selected_sector):
    """Use unique postings for all-sector view; preserve category rows for sector view."""
    if selected_sector == "All":
        df_exp = df.drop_duplicates(subset=["job_id"]).copy()
    else:
        df_exp = df[df["category"] == selected_sector].copy()

    df_exp["exp_group"] = pd.cut(
        df_exp["min_exp"],
        bins=EXP_BINS,
        labels=EXP_LABELS,
        right=True,
        include_lowest=True,
    )
    return df_exp


def render(df):
    st.subheader("🛠️ Experience Analysis")
    st.markdown('Objective: Align the "Level" of training with market reality to ensure graduate ROI.')

    exp_comp_sectors = ["All"] + sorted(df["category"].dropna().unique().tolist())
    selected_exp_sector = st.selectbox("Filter by Sector:", exp_comp_sectors, key="tab3_sector_filter")

    df_exp = _prepare_experience_df(df, selected_exp_sector)
    salary_col = "average_salary_cleaned" if "average_salary_cleaned" in df_exp.columns else "average_salary"

    c3_new1, c3_new2 = st.columns(2)

    with c3_new1:
        st.markdown("#### Experience Level Distribution")
        st.caption("Distribution of total vacancies by experience level")
        if selected_exp_sector == "All":
            st.caption("All-sector view uses unique `job_id` to avoid duplication from exploded categories.")

        exp_dist = df_exp.groupby("exp_group", observed=False)["num_vacancies"].sum().reset_index()
        exp_dist = exp_dist.sort_values("num_vacancies", ascending=False)
        exp_dist = exp_dist[exp_dist["num_vacancies"] > 0]

        if exp_dist.empty:
            st.info("No vacancy data available for the selected filter.")
            return

        max_idx = exp_dist["num_vacancies"].idxmax()
        explode = [0.1 if idx == max_idx else 0 for idx in exp_dist.index]

        distinct_colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A", "#98D8C8", "#F7DC6F", "#BB8FCE"]
        fig_pie = go.Figure(data=[go.Pie(
            labels=exp_dist["exp_group"],
            values=exp_dist["num_vacancies"],
            pull=explode, hole=0.3,
            marker=dict(colors=distinct_colors),
            textinfo="label+percent",
            textposition="auto",
            insidetextorientation="horizontal",
        )])
        fig_pie.update_layout(title="Vacancy Distribution by Experience Level",
                              height=400, showlegend=False)
        st.plotly_chart(fig_pie, use_container_width=True, key="exp_distribution_pie")

    with c3_new2:
        st.markdown("#### Average Salary Distribution by Experience")
        st.caption("Weighted salary ranges across experience levels (using cleaned salary column).")

        show_outliers = st.toggle("Show outliers", value=False, key="tab3_show_outliers")
        colors = ["#FFB6C1", "#87CEEB", "#90EE90", "#FFD700"]

        df_plot = df_exp.copy()
        df_plot["weight_cap"] = df_plot["num_vacancies"].fillna(0).clip(upper=5).astype(int)
        df_plot = df_plot[df_plot["weight_cap"] > 0]
        df_expanded = df_plot.loc[df_plot.index.repeat(df_plot["weight_cap"])].reset_index(drop=True)
        plot_df = df_expanded.dropna(subset=["exp_group", salary_col]).copy()

        if plot_df.empty:
            st.info("No salary data available for the selected filter.")
            return

        vacancy_totals = df_plot.groupby("exp_group", observed=False)["num_vacancies"].sum()

        fig, ax = plt.subplots(figsize=(12, 6))
        sns.boxplot(
            data=plot_df,
            x="exp_group",
            y=salary_col,
            order=EXP_LABELS,
            palette=colors,
            linewidth=1,
            showfliers=show_outliers,
            ax=ax,
        )
        ax.set_xticks(range(len(EXP_LABELS)))
        xticklabels = [f"{grp}\n(Total vacancies: {int(vacancy_totals.get(grp, 0))})" for grp in EXP_LABELS]
        ax.set_xticklabels(xticklabels)
        ax.set_xlabel("Experience (Years)", fontsize=12)
        ax.set_ylabel("Average Salary (SGD)", fontsize=12)
        ax.set_title("Average Salary Distribution by Experience Group (weighted by num_vacancies, cap=5)",
                     fontsize=14, fontweight="bold")
        plt.grid(True, axis="y", which="major", linestyle="--", alpha=0.6)
        plt.minorticks_on()
        plt.grid(True, which="minor", axis="y", linestyle=":", alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.caption(f"Salary source: `{salary_col}`.")
