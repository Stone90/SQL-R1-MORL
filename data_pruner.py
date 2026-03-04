import pandas as pd
import os

# Updated to use the symlink path inside the project root
DB_ROOT = "data/NL2SQL/SynSQL-2.5M/databases"
FILES = ["data/train.parquet", "data/test.parquet"]

print(f"Validating against: {os.path.abspath(DB_ROOT)}")

for file_path in FILES:
    if not os.path.exists(file_path):
        print(f"Skipping {file_path} (not found)")
        continue

    df = pd.read_parquet(file_path)
    original_len = len(df)

    # Check existence via the symlink
    df = df[df['db_id'].apply(lambda x: os.path.exists(os.path.join(DB_ROOT, x, f"{x}.sqlite")))]

    print(f"{file_path}: Pruned {original_len - len(df)} rows. Remaining: {len(df)}")
    df.to_parquet(file_path)
