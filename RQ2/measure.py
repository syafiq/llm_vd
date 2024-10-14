import argparse
import tempfile
import subprocess
import json
import os
import pandas as pd
from transformers import AutoTokenizer
from multiprocessing import cpu_count
from tqdm import tqdm

def get_token_length(text):
    return len(tokenizer.encode(text, truncation=False))

def run_multimetric_parallel(df, language, batch_size=10, jobs=4):
    """
    Run multimetric analysis on multiple code snippets in parallel using --jobs option.
    
    Args:
        df: DataFrame containing code snippets.
        language: Programming language (e.g., 'javascript').
        batch_size: Number of code snippets to process in one batch.
        jobs: Number of parallel jobs to run with --jobs option.
    
    Returns:
        DataFrame with calculated metrics.
    """
    try:
        # Define the file extension based on the language
        language_extensions = {
            'js': '.js',
            'python': '.py',
            'java': '.java',
            'php': '.php',
            'go': '.go',
            'c_cpp': '.cpp'
        }
        extension = language_extensions.get(language.lower())
        if not extension:
            raise ValueError(f"Unsupported language: {language}")

        # List to store results
        all_results = []

        # Process the code snippets in batches
        for i in tqdm(range(0, len(df), batch_size), desc="Processing Batches"):
            batch = df.iloc[i:i + batch_size]
            temp_files = []  # To store temporary file paths
            
            # Create temporary files for the current batch
            for idx, row in batch.iterrows():
                temp_file = tempfile.NamedTemporaryFile(suffix=extension, mode='w+', delete=False)
                temp_file.write(row['text'])
                temp_file.flush()
                temp_files.append(temp_file.name)  # Store temp file path
                temp_file.close()  # We don't need to keep the file open
            
            # Run multimetric on the batch of files with --jobs
            command = ['multimetric', '--jobs', str(jobs)] + temp_files
            result = subprocess.run(command, capture_output=True, text=True)
            
            print(result)

            if result.returncode != 0:
                print(f"Error running multimetric: {result.stderr}")
                return None

            # Parse the multimetric output and match to DataFrame rows
            metrics = json.loads(result.stdout)
            for idx, temp_file in zip(batch.index, temp_files):
                file_metrics = metrics['files'][temp_file]
                # Add the calculated metrics to the corresponding DataFrame row
                df.at[idx, 'h_volume'] = file_metrics.get('halstead_volume', None)
                df.at[idx, 'h_difficulty'] = file_metrics.get('halstead_difficulty', None)
                df.at[idx, 'h_effort'] = file_metrics.get('halstead_effort', None)
                df.at[idx, 'cyclomatic_complexity'] = file_metrics.get('cyclomatic_complexity', None)
                df.at[idx, 'nloc'] = file_metrics.get('loc', None)

            # Clean up the temporary files after processing
            for temp_file in temp_files:
                os.remove(temp_file)

        return df

    except Exception as e:
        print(f"Error: {e}")
        return None

parser = argparse.ArgumentParser(description='Process language input for tokenizer.')
parser.add_argument('lang', type=str, help='The language code to be used')
args = parser.parse_args()
lang = args.lang

max_length = 4000
tokenizer = AutoTokenizer.from_pretrained(f"../model/{lang}")
df_test = pd.read_json(f"../{lang}/{lang}_date_test.json")
df_train = pd.read_json(f"../{lang}/{lang}_date_train.json")
df = pd.concat([df_train, df_test], ignore_index=True)
df['token_length'] = df['text'].apply(get_token_length)
df = df[df['token_length'] <= max_length].reset_index(drop=True)

df['h_volume'] = None
df['h_difficulty'] = None
df['h_effort'] = None
df['cyclomatic_complexity'] = None
df['nloc'] = None

df = run_multimetric_parallel(df, language=f"{lang}", batch_size=40, jobs=cpu_count())
df.to_json(f"multimetric/{lang}.json", orient='records', lines=True)
