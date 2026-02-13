import streamlit as st


PLOTLY_CHART_CONFIG = {
    "displaylogo": False,
    "responsive": True,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
    "toImageButtonOptions": {"format": "png", "filename": "chart_export", "scale": 2},
}


def apply_plotly_style(fig, height=None):
    fig.update_layout(
        template="plotly_white",
        font=dict(family="Inter, Segoe UI, Roboto, sans-serif", size=13, color="#0f172a"),
        title=dict(x=0.01, xanchor="left", font=dict(size=18, color="#0f172a")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
        margin=dict(l=16, r=16, t=56, b=24),
        hoverlabel=dict(bgcolor="#ffffff", font=dict(color="#0f172a")),
    )
    if height is not None:
        fig.update_layout(height=height)

    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(148, 163, 184, 0.25)",
        zeroline=False,
        linecolor="rgba(148, 163, 184, 0.5)",
        ticks="outside",
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(148, 163, 184, 0.25)",
        zeroline=False,
        linecolor="rgba(148, 163, 184, 0.5)",
        ticks="outside",
    )
    return fig


def render_plotly_chart(fig, key, height=None):
    apply_plotly_style(fig, height=height)
    st.plotly_chart(
        fig,
        use_container_width=True,
        key=key,
        config=PLOTLY_CHART_CONFIG,
    )
