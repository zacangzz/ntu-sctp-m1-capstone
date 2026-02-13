# Streamlit Dashboard

Interactive dashboard analyzing ~1M Singapore job postings to inform curriculum design decisions.

## How to Run

```bash
streamlit run streamlit/app.py
```

Run from the **project root** (the directory containing `data/`).

## Tabs

| Tab | Purpose |
|-----|---------|
| **Executive Summary** | KPI metrics (vacancies, posts, views) and top-10 sector bar chart |
| **Sectoral Demand & Momentum** | Bulk-factor velocity line chart, hiring heatmap, and category competition snapshot |
| **Experience Level** | Vacancy distribution pie chart and salary boxplot by experience segment |
| **Opportunity** | Supply-vs-demand treemap and hidden-demand quadrant scatter (by industry, job title, or skill) |
| **Skills Analysis** | Popularity, emerging/declining trends, premium salary skills, transferability, and per-category breakdowns |

## Data Dependencies

All paths are relative to the project root:

| File | Used By |
|------|---------|
| `data/cleaned-sgjobdata-exploded.parquet` | Tabs 1-4 (main dataset) |
| `data/skills_optimized.parquet` | Tab 2 skill timeline |
| `data/cleaned-sgjobdata-withskills.parquet` | Tabs 4-5 (skills-enriched data) |

## Visual Standardization

- Default chart library is interactive Plotly across all tabs.
- Shared chart styling and Plotly config are centralized in `streamlit/chart_style.py`.
- Use `render_plotly_chart(fig, key=...)` (instead of calling `st.plotly_chart` directly) to keep typography, spacing, grid, hover labels, and modebar behavior consistent.

## File Structure

```
streamlit/
  app.py              # Entry point: config, styling, tab shell, main()
  chart_style.py      # Shared Plotly styling + chart render helper
  data_loader.py      # All @st.cache_data loading functions
  tabs/
    __init__.py
    tab_executive.py   # Tab 1
    tab_sectoral.py    # Tab 2
    tab_experience.py  # Tab 3
    tab_opportunity.py # Tab 4
    tab_skills.py      # Tab 5
```

Each tab module exposes a `render(df)` function (Tab 5's `render()` loads its own data).
