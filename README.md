# Workforce Intelligence Portal

A Streamlit app for analyzing Singapore job market data and aligning curriculum with real-time market demand. Built for NTU SCTP M1 Capstone.

## Features

- **Executive Summary** — Total vacancies, job posts, and views with top sectors breakdown
- **Market Demand** — Bulk hiring map, top job titles by sector
- **Salary & Value** — Experience vs compensation matrix, salary heatmaps by sector and role
- **Market Momentum** — Historical vacancy trends, seasonal hiring patterns, growth leaderboard
- **Curriculum Deep-Dive** — Sector-specific job exploration with skillset reference

## Project Structure

```
├── app.py                 # Streamlit dashboard (main application)
├── preprocess_data.py     # Data cleansing pipeline → SGJobData_opt.csv.xz
├── title_cleaner.py       # Standalone job title cleaning utilities
├── requirements.txt
├── Dockerfile
├── data/
│   ├── SGJobData.csv.xz       # Raw job data (compressed)
│   ├── SGJobData_opt.csv.xz   # Pre-cleaned data (generated)
│   └── skillset.csv           # Curriculum/skills reference
└── uat/                   # Notebooks for EDA and testing
```

## Data Pipeline

1. **Raw data** — `SGJobData.csv.xz` (compressed for Git)
2. **Preprocess** — `python preprocess_data.py` applies cleansing (titles, salary, dates, types) and keeps only columns used by the app
3. **Output** — `SGJobData_opt.csv.xz` (smaller, ready for fast load)
4. **App** — Loads opt file, explodes categories at runtime, renders dashboards

## Prerequisites

- Python 3.9+
- pip

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
# 1. Generate pre-cleaned data (first time or after raw data update)
python preprocess_data.py

# 2. Run the app
streamlit run app.py
```

## Deployment

Dockerfile included for containerized deployment (e.g. Google Cloud Run).

**Live demo:** https://ntu-m1-capstone.theluwak.com/
