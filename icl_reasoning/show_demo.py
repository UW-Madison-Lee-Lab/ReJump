#!/usr/bin/env python3
"""
Convert parquet file to human-readable format
Default input file: /staging/szhang967/icl_datasets/blobs_50shot_n1.0_f0.0_test10_icl3_seed42.parquet
"""

import pandas as pd
import json
import os
import argparse
from pathlib import Path


def convert_to_readable_format(parquet_file, output_format="txt"):
    """
    Convert parquet file to a human-readable format
    
    Args:
        parquet_file: Path to the parquet file
        output_format: Output format, supports txt and json
    
    Returns:
        Path to the output file
    """
    # Read the parquet file
    print(f"Reading parquet file: {parquet_file}")
    df = pd.read_parquet(parquet_file)
    
    # Determine the output file path
    output_file = os.path.splitext(parquet_file)[0] + f".{output_format}"
    print(f"Output file: {output_file}")
    
    if output_format == "json":
        # Convert DataFrame to JSON format
        with open(output_file, 'w', encoding='utf-8') as f:
            # Convert DataFrame to a list of dictionaries
            records = []
            for _, row in df.iterrows():
                record = {}
                for column in df.columns:
                    # Try to parse JSON strings
                    if isinstance(row[column], str) and (row[column].startswith('{') or row[column].startswith('[')):
                        try:
                            record[column] = json.loads(row[column])
                        except json.JSONDecodeError:
                            record[column] = row[column]
                    else:
                        record[column] = row[column]
                records.append(record)
            
            # Write to JSON file
            json.dump(records, f, indent=2, ensure_ascii=False)
    else:
        # Convert DataFrame to text format
        with open(output_file, 'w', encoding='utf-8') as f:
            for idx, row in df.iterrows():
                f.write(f"=== Sample {idx+1} ===\n\n")
                
                # Write prompt
                f.write("--- Prompt ---\n")
                f.write(row['prompt'])
                f.write("\n\n")
                
                # Write features and label
                f.write("--- Features and Label ---\n")
                f.write(f"Features: {row['features']}\n")
                f.write(f"Label: {row['label']}\n\n")
                
                # Write ICL example metadata if available
                if 'icl_example_meta_info' in row:
                    f.write("--- ICL Example Metadata ---\n")
                    # Handle if it's a string that needs to be parsed
                    meta_info = row['icl_example_meta_info']
                    
                    # Handle different types of meta_info
                    if isinstance(meta_info, str):
                        try:
                            meta_info = json.loads(meta_info)
                        except json.JSONDecodeError:
                            f.write(f"{meta_info}\n")
                    elif hasattr(meta_info, 'tolist'):  # Handle numpy arrays or pandas series
                        meta_info = meta_info.tolist()
                    
                    # Format and write the metadata
                    if isinstance(meta_info, list):
                        for i, config in enumerate(meta_info):
                            f.write(f"Config {i+1}:\n")
                            if isinstance(config, dict):
                                for key, value in config.items():
                                    f.write(f"  {key}: {value}\n")
                            else:
                                f.write(f"  {config}\n")
                    elif isinstance(meta_info, dict):
                        for key, value in meta_info.items():
                            f.write(f"{key}: {value}\n")
                    else:
                        # If not a list or dict, just write it directly
                        f.write(f"{meta_info}\n")
                    f.write("\n")
                
                # Write test data metadata
                f.write("--- Test Data Info ---\n")
                for key, value in row['test_data'].items():
                    f.write(f"{key}: {value}\n")
                f.write("\n")
                
                # Write separator
                f.write("="*50 + "\n\n")
    
    print(f"Saved to: {output_file}")
    return output_file


def main():
    parser = argparse.ArgumentParser(description='Convert parquet file to human-readable format')
    parser.add_argument('--input', type=str, 
                        default='/staging/szhang967/icl_datasets/blobs_50shot_n1.0_f0.0_test10_icl3_seed42.parquet',
                        help='Path to the input parquet file')
    parser.add_argument('--format', type=str, choices=['txt', 'json'], default='txt',
                        help='Output format (txt or json)')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: File '{args.input}' not found")
        return
    
    # Convert file format
    output_file = convert_to_readable_format(args.input, args.format)
    print(f"Conversion complete! Saved to: {output_file}")


if __name__ == "__main__":
    main()
