import pandas as pd
import argparse
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_path", type=str, required=True)
    args = parser.parse_args()
    
    # Read evaluation results
    df = pd.read_parquet(args.eval_path)
    
    # Calculate accuracy
    accuracy = df['is_correct'].mean()
    
    # Return 0 (success) if accuracy is 1.0, 1 (failure) otherwise
    if accuracy == 1.0:
        return 0
    return 1

if __name__ == "__main__":
    sys.exit(main()) 