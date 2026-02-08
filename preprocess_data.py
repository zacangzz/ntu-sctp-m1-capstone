#!/usr/bin/env python3
"""
Preprocess SGJobData.csv.xz: apply cleansing (title, salary, dates, types)
without exploding categories. Save as SGJobData_opt.csv.xz.
App.py handles category explosion at load time.
"""
import pandas as pd
import numpy as np
import re
import os


def clean_job_title(title):
    if not isinstance(title, str):
        return "Unknown Title"
    title = re.split(r'[-|]', title)[0].strip()
    title = re.sub(r'\$[\d,\.]+[kK]?', '', title)
    title = re.sub(r'up to \$?[\d,\.]+[kK]?', '', title)
    title = re.sub(r'basic \$?[\d,\.]+[kK]?', '', title)
    title = re.sub(r'\([^)]*\)', '', title)
    title = re.sub(r'urgent hiring!!!?', '', title, flags=re.IGNORECASE)
    title = re.sub(r'immediate', '', title, flags=re.IGNORECASE)
    title = re.sub(r'#\w+', '', title)
    title = re.sub(r'[|/]', ' ', title)
    noise_words = {
        'urgent', 'hiring', 'entry', 'level', 'immediate', 'available',
        'position', 'role', 'job', 'career', 'opportunity', 'full', 'time',
        'part', 'permanent', 'contract', 'temporary', 'up', 'to', '$',
        'basic', 'salary', 'plus', 'commission', 'bonus', 'benefits', 'package'
    }
    words = title.lower().split()
    meaningful_words = []
    for word in words:
        word = word.strip('.,!?()[]{}:;"\'')
        if word.isalpha() and len(word) > 1 and word not in noise_words:
            meaningful_words.append(word)
    if meaningful_words:
        cleaned_title = ' '.join(meaningful_words)
        cleaned_title = ' '.join(word.capitalize() for word in cleaned_title.split())
        return cleaned_title
    return title.strip()


def remove_outliers(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return np.where(df[col] > upper_bound, upper_bound,
                    np.where(df[col] < lower_bound, lower_bound, df[col]))


def main():
    input_file = "data/SGJobData.csv.xz"
    output_file = "data/SGJobData_opt.csv.xz"

    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        return 1

    print(f"Loading {input_file}...")
    df = pd.read_csv(input_file)

    print("Applying cleansing pipeline (no category explosion)...")
    # 1. Drop rows with missing categories
    df.dropna(subset=['categories'], inplace=True)

    # 2. Clean Titles
    df['clean_title'] = df['title'].apply(clean_job_title)

    # 3. Type Conversion
    numeric_cols = ['minimumYearsExperience', 'numberOfVacancies', 'salary_minimum', 'salary_maximum']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.copy()
    df['numberOfVacancies'] = df['numberOfVacancies'].fillna(1)
    df['minimumYearsExperience'] = df['minimumYearsExperience'].fillna(0)

    # 4. Salary Calculation
    if 'salary_minimum' in df.columns and 'salary_maximum' in df.columns:
        valid_salary = (df['salary_minimum'] > 0) & (df['salary_maximum'] > 0)
        df = df[valid_salary]
        df['avg_salary'] = (df['salary_minimum'] + df['salary_maximum']) / 2
    else:
        df['avg_salary'] = 0

    # 5. Outlier Handling
    df['minimumYearsExperience'] = np.clip(df['minimumYearsExperience'], 0, 15)
    if 'avg_salary' in df.columns and not df.empty:
        df['avg_salary'] = remove_outliers(df, 'avg_salary')

    # 6. Date Parsing
    date_col = None
    for col in ['metadata_newPostingDate', 'metadata_createdAt', 'postingDate']:
        if col in df.columns:
            date_col = col
            break

    if date_col:
        df['posting_date'] = pd.to_datetime(df[date_col], errors='coerce')
        df['month_year'] = df['posting_date'].dt.to_period('M').dt.to_timestamp()
        df.dropna(subset=['posting_date'], inplace=True)

    # Keep only columns used by app.py
    keep_cols = [
        'categories',
        'clean_title',
        'avg_salary',
        'posting_date',
        'month_year',
        'numberOfVacancies',
        'minimumYearsExperience',
        'positionLevels',
        'metadata_totalNumberOfView',
    ]
    df = df[[c for c in keep_cols if c in df.columns]]
    if 'metadata_totalNumberOfView' not in df.columns:
        df['metadata_totalNumberOfView'] = 0

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    print(f"Saving to {output_file}...")
    df.to_csv(output_file, index=False, compression='xz')
    print(f"Done. Output: {output_file} ({len(df):,} rows)")
    return 0


if __name__ == "__main__":
    exit(main())
