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
                
                # Write data source if available
                if 'data_source' in row:
                    f.write(f"--- Data Source ---\n")
                    f.write(f"{row['data_source']}\n\n")
                
                # Write prompt
                f.write("--- Prompt ---\n")
                if isinstance(row['prompt'], list):
                    # For the new format with role-based prompts
                    for prompt_item in row['prompt']:
                        if isinstance(prompt_item, dict):
                            f.write(f"Role: {prompt_item.get('role', 'unknown')}\n")
                            f.write(f"Content: {prompt_item.get('content', '')}\n")
                        else:
                            f.write(f"{prompt_item}\n")
                elif hasattr(row['prompt'], 'tolist'):  # Handle numpy arrays
                    prompt_list = row['prompt'].tolist()
                    for item in prompt_list:
                        if isinstance(item, dict):
                            f.write(f"Role: {item.get('role', 'unknown')}\n")
                            f.write(f"Content: {item.get('content', '')}\n")
                        else:
                            f.write(f"{str(item)}\n")
                else:
                    # For the old format with direct string prompts
                    f.write(str(row['prompt']))
                f.write("\n\n")
                
                # Write the label from reward model or directly
                f.write("--- Label Information ---\n")
                if 'reward_model' in row and isinstance(row['reward_model'], dict):
                    f.write(f"Style: {row['reward_model'].get('style', 'unknown')}\n")
                    ground_truth = row['reward_model'].get('ground_truth', 'N/A')
                    
                    # Handle different ground_truth formats
                    if isinstance(ground_truth, dict):
                        f.write("Ground Truth:\n")
                        if 'label' in ground_truth:
                            f.write(f"  Label: {ground_truth['label']}\n")
                        if 'features' in ground_truth:
                            # Format features nicely
                            if isinstance(ground_truth['features'], list):
                                features_str = ", ".join([f"{x:.3f}" for x in ground_truth['features']])
                                f.write(f"  Features: [{features_str}]\n")
                            else:
                                f.write(f"  Features: {ground_truth['features']}\n")
                    else:
                        f.write(f"Ground Truth: {ground_truth}\n")
                elif 'label' in row:
                    # For backwards compatibility with the old format
                    f.write(f"Label: {row['label']}\n")
                f.write("\n")
                
                # Write ability if available
                if 'ability' in row:
                    f.write(f"--- Ability ---\n")
                    f.write(f"{row['ability']}\n\n")
                
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
                
                # Write extra info if available
                if 'extra_info' in row and isinstance(row['extra_info'], dict):
                    f.write("--- Extra Info ---\n")
                    for key, value in row['extra_info'].items():
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
