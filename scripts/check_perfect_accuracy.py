import pandas as pd
import argparse
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_path", type=str, required=True)
    args = parser.parse_args()
    
    df = pd.read_parquet(args.eval_path)

    return 1

if __name__ == "__main__":
    sys.exit(main()) 