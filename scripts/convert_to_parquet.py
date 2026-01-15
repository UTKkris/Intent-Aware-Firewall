
import pandas as pd
import numpy as np
import glob
import os
import sys
import warnings
import pyarrow as pa
import pyarrow.parquet as pq


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)

print("Starting MEMORY-SAFE CHUNK-BASED conversion (v8)...")
print("This will read, clean, and harmonize all CSVs.")


csv_path = r'../data/Processed Traffic Data for ML Algorithms/*.csv'
parquet_path = r'../data/cic_ids_2018_full_harmonized_CLEAN.parquet' 
chunk_size = 500000 


all_files = glob.glob(csv_path)

if not all_files:
    print(f"Error: No CSV files found at {csv_path}")
    sys.exit(1)

print(f"Found {len(all_files)} CSV files. Finding common columns...")


common_cols = None
for f in all_files:
    df_header = pd.read_csv(f, nrows=0, low_memory=False)
    current_cols = set(df_header.columns.str.strip())
    if common_cols is None:
        common_cols = current_cols
    else:
        common_cols = common_cols.intersection(current_cols)

common_cols = list(common_cols)
if 'Label' not in common_cols:
    print("CRITICAL ERROR: 'Label' column not found. Stopping.")
    sys.exit(1)
print(f"\nFound {len(common_cols)} common columns.")


print("Creating a unified data type map (read all as string)...")
dtype_map = {col: 'str' for col in common_cols}
print("Data type map created.")


print("Converting and cleaning files one-by-one, in chunks...")

if os.path.exists(parquet_path):
    os.remove(parquet_path)
    print(f"Removed old file: {parquet_path}")

total_rows = 0
writer = None 

try:
    for f in all_files:
        print(f"\n--- Processing file: {os.path.basename(f)} ---")
        
        chunk_iterator = pd.read_csv(
            f, 
            low_memory=False, 
            usecols=common_cols, 
            chunksize=chunk_size,
            dtype=dtype_map
        )
        
        for i, chunk_df in enumerate(chunk_iterator):
            chunk_df.columns = chunk_df.columns.str.strip() 
            
            
            numeric_cols = [col for col in chunk_df.columns if col not in ['Label', 'Timestamp']]
            
            
            chunk_df[numeric_cols] = chunk_df[numeric_cols].apply(pd.to_numeric, errors='coerce')
            
            
            chunk_df[numeric_cols] = chunk_df[numeric_cols].replace([np.inf, -np.inf], np.nan)
            
            
            chunk_df[numeric_cols] = chunk_df[numeric_cols].astype(np.float64)
            
            
            chunk_df[numeric_cols] = chunk_df[numeric_cols].fillna(0.0)
            
            
            iat_cols = [col for col in numeric_cols if 'IAT' in col or 'Duration' in col]
            for col in iat_cols:
                chunk_df[col] = chunk_df[col].clip(lower=0)
            
           
            chunk_df['Label'] = chunk_df['Label'].apply(lambda x: 'Benign' if str(x).upper() == 'BENIGN' else 'Attack')
           
            
            table = pa.Table.from_pandas(chunk_df, preserve_index=False) 
            
            if writer is None:
               
                writer = pq.ParquetWriter(
                    parquet_path, 
                    table.schema, 
                    compression='snappy'
                )
            
           
            writer.write_table(table.cast(writer.schema))
            
            total_rows += len(chunk_df)
            print(f"  ...appended chunk {i+1} ({len(chunk_df)} rows). Total rows: {total_rows}", end='\r')
        
        print(f"\nFinished file {os.path.basename(f)}. Total rows: {total_rows}   ")

finally:
    if writer is not None:
        writer.close()
        print("\nParquet writer closed.")

print("\n--- HARMONIZED CHUNK-BASED CONVERSION COMPLETE! ---")
print(f"File saved to: {parquet_path}")