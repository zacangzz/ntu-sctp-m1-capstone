# Capstone Project

This README compiles information about this project and my learnings w.r.t. the process of getting here.

## About using AI coding tools
1. running claude code in CLI is pretty handy
2. initialising claude project with ```claude /init``` is useful after you're already had some work done and set up some foundation. this creates a claude.md file that helps guide claude on behaviour.
3. running the plan mode without properly writing out instructions sucks up tokens really really quickly and it gets expensive. plan mode at this stage uses opus 4.6 which is the most expensive - perhaps i should check if plan mode can use other models?
4. run ```/compact``` frequently can help to maintain some context without over consuming tokens!
5. always initialise git *AND* commit regularly before and after each claude action so that you can track changes and roll back if necessary!
6. have a structured plan and instructions in MD files, and then when providing instructions to claude or codex, reference it directly using ```@``` to point it to files and remind it to instructions.

## Files

* ```data/SGJobData.csv.xz```, this file is the CSV file data source for the project, except compressed so that GitHub can accept it.
    * processed data are saved as ```'*.parquet'``` files, and prefixed with 'cleaned', 3 files:
        * pre-exploded categories
        * post-exploded categories
        * pre-explode categories with skills
* ```notebooks/*.ipynb```, jupyter notebook for testing out graphs and commands etc.
    * ```-eda.ipynb```, file that generated the cleaned parquet dfs
    * ```-ed-ml.ipynb```, can ignore, first trial for generating skillslist
    * ```analysis.ipynb```, comprehensive skills analysis dashboard with visualizations
* ```app.py```, main source code for streamlit application hosting
    * run this in streamlit

## Skills Analysis Dashboard (notebooks/analysis.ipynb)

Comprehensive Jupyter notebook analyzing 2,053 unique skills across ~1M Singapore job postings (Oct 2022 - May 2024).

### Analysis Sections

1. **Skill Popularity Overview**
   - Top 20 most in-demand skills and 20 least popular skills
   - Distribution analysis showing skill demand concentration
   - Cumulative demand curve (80% of demand concentrated in top ~300 skills)

2. **Emerging & Declining Skills**
   - Growth rate analysis comparing recent 3 months vs previous 3 months
   - Top 20 fastest-growing and declining skills
   - Timeline trends for top 10 skills over 19-month period

3. **Skills vs Experience Level**
   - Heatmap showing skill-experience requirement patterns for top 30 skills
   - Box plots of experience distribution across top 15 skills
   - Analysis of skills requiring highest/lowest experience barriers

4. **High Premium Skills (Salary Analysis)**
   - Top 25 highest-paying skills with average salaries
   - Bubble chart: Salary vs Popularity vs Experience requirements
   - High-value skills identification (high salary + low experience barrier)

5. **Skills by Category**
   - Top 10 skills for each of the top 6 job categories
   - Stacked bar chart showing distribution of top skills across categories

6. **Universal Skills (Cross-Category Analysis)**
   - Skills appearing across most categories (transferable skills)
   - Transferability score metric (category coverage × demand)
   - Scatter plot: universality vs total demand

7. **Skill Co-occurrence Network**
   - Network graph showing which skills commonly appear together in job postings
   - Identifies skill clusters and complementary skill sets
   - Top 40 skills with 150 most common co-occurrence pairs

8. **Comprehensive Dashboard Summary**
   - Single-page overview combining all key insights
   - Statistical summary with key findings and trends

### Key Insights Generated

- Market concentration metrics (top N skills accounting for X% of demand)
- Fastest growing/declining skills with growth rates
- Highest-paying skills with salary benchmarks
- Most universal cross-industry transferable skills
- High-value skills (high ROI for learning)
- Skill combination patterns and clusters

### Data Source

Uses `data/cleaned-sgjobdata-withskills.parquet` as single source of truth (6.2M records, exploded by skills from ~1M unique jobs). Applies proper date parsing and category extraction without additional data cleaning.

## Data: data/cleaned-sgjobdata-exploded.parquet
this is a list dataframe of job postings.
 0   job_id                  string     : unique identifier
 1   title                   string     : job title (dirty and messy)
 2   company                 string     : company name
 3   min_exp                 Int64      : minimum experience required for this job
 4   positionlevels          string     : position level of the job
 5   num_applications        Int64      : number of applications for this job
 6   num_views               Int64      : number of views for this job
 7   num_vacancies           Int64      : number of vacancies for this 1 job posting
 8   average_salary          Float64    : salary expected for this job
 9   average_salary_cleaned  Float64    : salary expected for this job, cleaned using windsorisation
 10  category                object     : sector category, in JSON format containing id, and category (name)
 11  jobtitle_cleaned        string     : cleaned job title 
 12  skill                   string     : skills needed for this particular job_id

## How to set up environment
uv is a Python environment manager. To set up the environment, run the following commands after you have installed uv:

1. ```uv venv .venv``` : this will setup the virtual environment
2. ```uv venv .venv --python 3.12``` : this sets up the venv with python 3.12
3. ```source .venv/bin/activate``` : this activates the venv
4. ```uv sync``` : this needs the uv.lock file, it installs everything needed for this environment

## How to Lock env and dependencies? [for maintainer only, not for end users]
1. create a pyproject.toml file
2. run: ```uv lock```

## How to Run Streamlit app:
1. navigate to project folder
2. run command ```streamlit run app.py```

## Data Cleaning Notes // Issues
1. NULLs dropped via Job ID
2. Drop unused columns and rename for clarity
3. Capping average_salary column using Log-IQR concept, this caps upper bound at 19783.0, lower bound at 1110.0
    * problem with this is that data is already flawed, small numbers indicate user behaviour where they choose not to fill in salary, large numbers can be for the same reason or mistaking the field for 'annual' salary.
4. matching of job title posted to job titles from SkillsFuture list of jobs and skills
    * not a fool proof 100% match, for e.g. "Driver" is matched to "Engine Driver" which is wrong, the correct match should be "Transport Operator", the matching process uses sentence transformer to get closest possible match however cultural semantics are lost in this 'translation'.
    * first try was abandoned, where an LLM is used to perform one-shot inference, the data is minimally cleaned and then combined with company name to provide even more context in the hopes of the llm being able to generate a better list of 'top skills', however, the generated list of skills follow no taxonomy so there is no standard and no consistetncy for proper analysis. using a governement approved taxonomy is better sense.

## Collaboration via Git
* version control across many people is difficult, haven't learn how to collab properly

## Visualisation
* issues with getting the correct formulas to get the visuals that we want to see
* need to be more acquainted with the raw data, and do more test visualisations to see if things are good

## Links
* [Skills Framework Dataset](https://jobsandskills.skillsfuture.gov.sg/frameworks/skills-frameworks)

## Git & GitHub
* ```eval "$(ssh-agent -s)" ```
* ```ssh-add ~/.ssh/id_ed25519```
* ```ssh-add -l  ```
* ```ssh -T git@github.com```