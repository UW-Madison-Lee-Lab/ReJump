import pandas as pd
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Read and analyze a parquet file')
    parser.add_argument('--file_path', 
                       type=str,
                       default="/var/lib/condor/execute/slot1/dir_1963007/liftr/results/moons/Qwen2.5-1.5B-Instruct_1_shot_iter0_correct_train.parquet",
                       help='Path to the parquet file')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Check if file exists
    if not os.path.exists(args.file_path):
        print(f"File does not exist: {args.file_path}")
        exit(1)

    try:
        # Read parquet file
        df = pd.read_parquet(args.file_path)
        
        # Set display options to show all columns and rows
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)
        
        # Display basic information
        print("\n=== Basic Information ===")
        print(f"Data shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Display data types
        print("\n=== Data Types ===")
        print(df.dtypes)
        
        # Display first few rows
        print("\n=== First 5 Rows ===")
        print(df.head())
        
        # Display basic statistics
        print("\n=== Basic Statistics ===")
        print(df.describe())

        print(type(df[["prompt"]]))
        # print(df[["prompt"]][0])
        # print(df[["prompt"]][1])
        print(df["prompt"].tolist())

    except Exception as e:
        print(f"Error reading file: {str(e)}")
    
    

if __name__ == "__main__":
    main() 