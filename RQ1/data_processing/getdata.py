import argparse
import pandas as pd
import sqlite3 as lite
from sqlite3 import Error
from pathlib import Path
from datetime import date
from sklearn.model_selection import train_test_split

def create_connection(db_file):
    conn = None
    try:
        conn = lite.connect(db_file, timeout=10) 
    except Error as e:
        print(e)
    return conn

parser = argparse.ArgumentParser(description='Get data for a specific programming language.')
parser.add_argument('lang', type=str, help='Programming language to filter on')
args = parser.parse_args()
lang = args.lang

#cvefixes_loc = <somewhere>
cvefixes_loc = "../../../kevin/CVEfixes.db"
conn = create_connection(cvefixes_loc)

query = """
       SELECT m.code, m.before_change, c.committer_date
       FROM file_change f, method_change m, commits c
       WHERE m.file_change_id = f.file_change_id
       AND c.hash = f.hash
       AND f.programming_language = ?
       """

df = pd.read_sql_query(query, conn, params=[lang])

df = df.drop_duplicates(subset=['code'], ignore_index=True)
df = df.rename(columns={'before_change': 'label', 'code': 'text'})
df.loc[df.label == 'False', 'label'] = 0
df.loc[df.label == 'True', 'label'] = 1

df['committer_date'] = pd.to_datetime(df['committer_date'])
df = df.sort_values(by='committer_date')
dfx = df.drop(df.columns[[2]], axis=1)
split_index = int(len(df) * 0.8)
train = dfx.iloc[:split_index]
test = dfx.iloc[split_index:]
test, validation = train_test_split(test, test_size=0.5)
train.to_json(f"{lang}_date_train.json", orient='records')
validation.to_json(f"{lang}_date_valid.json", orient='records')
test.to_json(f"{lang}_date_test.json", orient='records')
