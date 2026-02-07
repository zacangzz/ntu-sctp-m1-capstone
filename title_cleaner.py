import pandas as pd
import re

def get_user_choice():
    """
    Get user input for number of records to process
    """
    print("=== Job Title Cleaning ===")
    print("\nUsing rule-based cleaning (optimized for job titles)")
    
    print("\nChoose number of records to process:")
    print("   0 - All records")
    print("   Or enter a specific number (e.g., 1000, 10000)")
    
    while True:
        try:
            count_input = input("\nEnter number of records: ").strip()
            if count_input == "0":
                sample_size = None
                count_desc = "all"
                break
            else:
                sample_size = int(count_input)
                if sample_size > 0:
                    count_desc = str(sample_size)
                    break
                else:
                    print("Please enter a positive number or 0")
        except ValueError:
            print("Please enter a valid number")
    
    return sample_size, count_desc

def rule_based_clean_title(title):
    """
    Improved rule-based cleaning that preserves complete job titles
    """
    # Remove location information (after | or -)
    title = re.split(r'[-|]', title)[0].strip()
    
    # Remove salary information
    title = re.sub(r'\$[\d,\.]+[kK]?', '', title)
    title = re.sub(r'up to \$?[\d,\.]+[kK]?', '', title)
    title = re.sub(r'basic \$?[\d,\.]+[kK]?', '', title)
    
    # Remove parentheses content (skills/requirements)
    title = re.sub(r'\([^)]*\)', '', title)
    
    # Remove urgency markers
    title = re.sub(r'urgent hiring!!!?', '', title, flags=re.IGNORECASE)
    title = re.sub(r'immediate', '', title, flags=re.IGNORECASE)
    
    # Remove hashtags and extra symbols
    title = re.sub(r'#\w+', '', title)
    title = re.sub(r'[|/]', ' ', title)
    
    # Remove common noise words but keep important job title components
    noise_words = {
        'urgent', 'hiring', 'entry', 'level', 'immediate', 'available', 
        'position', 'role', 'job', 'career', 'opportunity', 'full', 'time', 
        'part', 'permanent', 'contract', 'temporary', 'up', 'to', '$',
        'basic', 'salary', 'plus', 'commission', 'bonus', 'benefits', 'package'
    }
    
    # Tokenize and clean, but be more conservative
    words = title.lower().split()
    meaningful_words = []
    
    for word in words:
        word = word.strip('.,!?()[]{}:;"\'')
        if (word.isalpha() and 
            len(word) > 1 and 
            word not in noise_words):
            meaningful_words.append(word)
    
    # Reconstruct and capitalize properly
    if meaningful_words:
        cleaned_title = ' '.join(meaningful_words)
        # Capitalize first letter of each word for job titles
        cleaned_title = ' '.join(word.capitalize() for word in cleaned_title.split())
        return cleaned_title
    else:
        return title.strip()

def clean_job_title(title):
    """
    Clean job title using optimized rule-based approach
    """
    if pd.isna(title) or not isinstance(title, str):
        return title
    
    return rule_based_clean_title(title)

def clean_job_titles_batch(df, title_column='title', sample_size=None):
    """
    Clean job titles in batch and add cleaned column
    """
    df_copy = df.copy()
    
    # Optionally sample data for faster testing
    if sample_size and sample_size < len(df_copy):
        df_copy = df_copy.sample(n=sample_size, random_state=42)
        print(f"Using sample of {sample_size} titles")
    
    print(f"Cleaning {len(df_copy)} job titles...")
    df_copy['cleaned_title'] = df_copy[title_column].apply(clean_job_title)
    
    return df_copy

def analyze_title_cleaning(df, original_col='title', cleaned_col='cleaned_title'):
    """
    Analyze the effectiveness of title cleaning
    """
    print("Title Cleaning Analysis:")
    print(f"Total titles: {len(df)}")
    
    # Show examples of cleaned titles
    sample_size = min(10, len(df))
    print(f"\nSample cleaned titles (first {sample_size}):")
    for i in range(sample_size):
        original = df[original_col].iloc[i]
        cleaned = df[cleaned_col].iloc[i]
        print(f"Original: {original}")
        print(f"Cleaned:  {cleaned}")
        print("-" * 50)
    
    # Statistics
    changed_titles = df[df[original_col] != df[cleaned_col]]
    print(f"\nTitles changed: {len(changed_titles)} ({len(changed_titles)/len(df)*100:.1f}%)")
    
    # Most common words in cleaned titles
    all_cleaned_words = ' '.join(df[cleaned_col].dropna()).lower().split()
    word_freq = pd.Series(all_cleaned_words).value_counts().head(20)
    print(f"\nTop 20 most common words in cleaned titles:")
    print(word_freq)

def load_data(file_path):
    """
    Load data from a csv file into a pandas DataFrame
    """
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None






def get_user_choice():
    """
    Get user input for number of records to process
    """
    print("=== Job Title Cleaning ===")
    print("\nUsing rule-based cleaning (optimized for job titles)")
    
    print("\nChoose number of records to process:")
    print("   0 - All records")
    print("   Or enter a specific number (e.g., 1000, 10000)")
    
    while True:
        try:
            count_input = input("\nEnter number of records: ").strip()
            if count_input == "0":
                sample_size = None
                count_desc = "all"
                break
            else:
                sample_size = int(count_input)
                if sample_size > 0:
                    count_desc = str(sample_size)
                    break
                else:
                    print("Please enter a positive number or 0")
        except ValueError:
            print("Please enter a valid number")
    
    return sample_size, count_desc

def save_cleaned_data(df, count_desc):
    """
    Save cleaned data to CSV with appropriate filename - only title and cleaned_title columns
    """
    filename = f"clean_title-{count_desc}.csv"
    filepath = f"data/{filename}"
    
    # Create data directory if it doesn't exist
    import os
    os.makedirs('data', exist_ok=True)
    
    # Save only the title and cleaned_title columns
    output_df = df[['title', 'cleaned_title']].copy()
    output_df.to_csv(filepath, index=False)
    print(f"\n✅ Cleaned data saved to: {filepath}")
    print(f"📊 Total records saved: {len(output_df)}")
    print(f"📝 Columns: title, cleaned_title")
    
    return filepath

def load_data(file_path):
    """
    Load data from a csv file into a pandas DataFrame
    """
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None

def main():
    """
    Main entry point for the script with user interaction
    """
    # Get user choices
    sample_size, count_desc = get_user_choice()
    
    # Load data
    file_path = "data/SGJobData.csv.xz"
    df = load_data(file_path)
    if df is None:
        print("Error: Could not load data")
        return
    
    print(f"\n📁 Original data shape: {df.shape}")
    
    # Clean job titles
    print(f"\n🧹 Cleaning job titles using optimized rule-based method...")
    df_cleaned = clean_job_titles_batch(df, sample_size=sample_size)
    
    # Analyze the cleaning results
    analyze_title_cleaning(df_cleaned)
    
    # Save cleaned data
    save_cleaned_data(df_cleaned, count_desc)
    
    print("\n🎉 Title cleaning completed successfully!")

if __name__ == "__main__":
    main()
