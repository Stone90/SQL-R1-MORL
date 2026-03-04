import pandas as pd
import os

# Audit script for the restored example data
for filename in ['example_data/train.parquet', 'example_data/test.parquet']:
    if os.path.exists(filename):
        df = pd.read_parquet(filename)
        print(f"\n{'='*20} AUDITING: {filename} {'='*20}")
        print(f"Total Rows: {len(df)}")

        # Check the first row's structure
        row = df.iloc[0]
        print(f"Columns: {df.columns.tolist()}")
        print(f"Sample db_id: {row.get('db_id', 'N/A')}")

        # See exactly what the model is being told
        if 'prompt' in row:
            p = row['prompt']
            content = p[1]['content'] if isinstance(p, list) and len(p) > 1 else str(p)
            print(f"\n--- Model Prompt (Row 0) ---")
            print(content[:500] + "...")
    else:
        print(f"{filename} still missing!")
