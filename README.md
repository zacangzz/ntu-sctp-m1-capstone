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
    * other files are source files and file from skillsfuture
* ```notebooks/*.ipynb```, jupyter notebook for testing out graphs and commands etc. with 1 contribution from team members
    * ```-eda.ipynb```, file that generated the cleaned parquet dfs
    * ```-ed-ml.ipynb```, can ignore, first trial for generating skillslist
    * ```analysis.ipynb```, comprehensive skills analysis dashboard with visualizations
* ```scripts/ ```, test scripts, contribution from team members
* ```streamlit```, main source code for streamlit application hosting

### Data Source
Uses `data/cleaned-sgjobdata-withskills.parquet` as single source of truth (6.2M records, exploded by skills from ~1M unique jobs). Applies proper date parsing and category extraction without additional data cleaning.

## Data: data/cleaned-sgjobdata-withskills.parquet
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

* note: the file ```data/cleaned-sgjobdata.parquet``` is basically the same thing without skills, while the file ```data/cleaned-sgjobdata-exploded.parquet``` is the non skills version bit with category exploded

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
2. run command ```streamlit run streamlit/app.py```

## Data Cleaning Notes // Issues
1. NULLs dropped via Job ID
2. Drop unused columns and rename for clarity
3. Capping average_salary column using Log-IQR concept, this caps upper bound at 19783.0, lower bound at 1110.0
    * problem with this is that data is already flawed, small numbers indicate user behaviour where they choose not to fill in salary, large numbers can be for the same reason or mistaking the field for 'annual' salary.
4. matching of job title posted to job titles from SkillsFuture list of jobs and skills
   * the current data point is not enough
    * not a fool proof 100% match, for e.g. "Driver" is matched to "Engine Driver" which is wrong, the correct match should be "Transport Operator", the matching process uses sentence transformer to get closest possible match however cultural semantics are lost in this 'translation'.
    * first try was abandoned, where an LLM is used to perform one-shot inference, the data is minimally cleaned and then combined with company name to provide even more context in the hopes of the llm being able to generate a better list of 'top skills', however, the generated list of skills follow no taxonomy so there is no standard and no consistetncy for proper analysis. using a governement approved taxonomy is better sense.

## Collaboration via Git
* version control across many people is difficult, we haven't learn how to collab properly

## Visualisation
* issues with getting the correct formulas to get the visuals that we want to see
* need to be more acquainted with the raw data, and do more test visualisations to see if things are good

## Links
* [Skills Framework Dataset](https://jobsandskills.skillsfuture.gov.sg/frameworks/skills-frameworks)

## Git & GitHub identity debugging
* ```eval "$(ssh-agent -s)" ```
* ```ssh-add ~/.ssh/id_ed25519```
* ```ssh-add -l  ```
* ```ssh -T git@github.com```