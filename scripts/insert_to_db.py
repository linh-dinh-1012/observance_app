import pandas as pd
import sqlite3
import os

CSV_PATH = 'data/files_table.csv'
DB_PATH = 'database/observance.db'

def load_dataframe(csv_path):
    df = pd.read_csv(csv_path)
    return df

def insert_into_database(df, db_path):
    conn = sqlite3.connect(db_path)
    df.to_sql('files', conn, if_exists='append', index=False)
    conn.close()
    print(f"Added {len(df)} lignes to database")

def main():
    if not os.path.exists(CSV_PATH):
        print(f"File {CSV_PATH} not found")
        return
    
    df = load_dataframe(CSV_PATH)
    insert_into_database(df, DB_PATH)

if __name__ == '__main__':
    main()