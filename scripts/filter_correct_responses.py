import pandas as pd
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()
    
    # Read the generated responses and evaluation results
    df = pd.read_parquet(args.input_path)
    
    # Filter responses where is_correct is True
    correct_df = df[df['is_correct'] == True].copy()
    
    # Save filtered dataset
    correct_df.to_parquet(args.output_path)

if __name__ == "__main__":
    main() 