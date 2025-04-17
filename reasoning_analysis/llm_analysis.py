#!/usr/bin/env python3
"""
Process LLM analysis data from model responses
This module handles the extraction and evaluation of model functions from LLM analysis JSON data.

The module also computes:
- model_family_best_mse for regression tasks: Calculates the MSE achieved by using 
  scikit-learn's implementation of the corresponding model family.
- model_family_best_accuracy for classification tasks: Calculates the accuracy achieved by using
  scikit-learn's implementation of the corresponding model family.

These metrics help compare LLM-generated models with their scikit-learn counterparts.
"""

import json
import os
import argparse
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional

# Import required functions directly from data_processing
from data_processing import (
    extract_and_execute_model_functions,
    create_model_evaluation_table,
    get_ground_truth
)

def process_llm_analysis(llm_json, ground_truth: Dict[str, Any], data_type: str):
    """
    Process LLM analysis to create model evaluation table
    
    Args:
        llm_json: JSON string or object containing model functions
        ground_truth: Ground truth data containing in_context_samples
        data_type: 'classification' or 'regression'
        
    Returns:
        Model evaluation table or None if processing fails
    """
    if not llm_json or not ground_truth:
        return None
        
    try:
        # Extract model functions and evaluate
        model_results = extract_and_execute_model_functions(llm_json, ground_truth, data_type)
        
        # Parse the JSON to get the original models data
        models_data = []
        if isinstance(llm_json, str):
            try:
                models_data = json.loads(llm_json)
            except json.JSONDecodeError:
                models_data = []
        else:
            models_data = llm_json if isinstance(llm_json, list) else []
        
        # Create model evaluation table
        evaluation_table = create_model_evaluation_table(model_results, models_data, data_type)
        
        return evaluation_table
    except Exception as e:
        print(f"Error processing LLM analysis: {e}")
        return None

def process_parquet_file(input_file: str, output_dir: Optional[str] = None, 
                         max_samples: int = 0, data_type: str = 'regression'):
    """
    Process LLM analysis from parquet file and save results to JSON
    
    Args:
        input_file: Path to the input parquet file
        output_dir: Directory to save the output file (default: same as input file)
        max_samples: Maximum number of samples to include (default: 0 = all samples)
        data_type: Data type ('regression' or 'classification')
        
    Returns:
        Path to the output JSON file
    """
    # Read the parquet file
    print(f"Reading parquet file: {input_file}")
    df = pd.read_parquet(input_file)
    print(f"DataFrame loaded with {len(df)} rows and columns: {df.columns.tolist()}")
    
    # Check if required columns exist
    if 'llm_analysis_extracted_json' not in df.columns:
        print("Warning: 'llm_analysis_extracted_json' column not found in DataFrame")
    
    # Initialize output data structure
    llm_analysis_data = {
        "metadata": {
            "input_file": str(input_file),
            "timestamp": pd.Timestamp.now().isoformat(),
            "total_samples": len(df),
            "processed_samples": 0,
            "data_type": data_type,
        },
        "samples": []
    }
    
    # Sample the dataframe if needed
    display_df = df
    if max_samples > 0 and max_samples < len(df):
        display_df = df.sample(n=max_samples, random_state=42)
        print(f"Sampled {len(display_df)} out of {len(df)} samples for processing")
    
    # Process each row
    processed_samples = 0
    for idx, row in display_df.iterrows():
        # Check if this row has LLM analysis data
        if 'llm_analysis_extracted_json' not in row or row['llm_analysis_extracted_json'] is None:
            continue
            
        # Get ground truth
        ground_truth = get_ground_truth(row)
        if ground_truth is None or 'in_context_samples' not in ground_truth:
            continue
            
        # Get data source if available
        data_source = row.get('data_source', 'unknown')
        
        # Process LLM analysis
        llm_json = row['llm_analysis_extracted_json']
        evaluation_table = process_llm_analysis(llm_json, ground_truth, data_type)
        
        if evaluation_table:
            # Create sample data structure
            sample_data = {
                "index": int(idx),
                "data_source": data_source,
                "prompt": row.get('prompt', ''),
                "model_responses": row.get('responses', []),
                "model_evaluation_table": evaluation_table
            }
            
            # Add to samples list
            llm_analysis_data["samples"].append(sample_data)
            processed_samples += 1
    
    # Update metadata
    llm_analysis_data["metadata"]["processed_samples"] = processed_samples
    
    # Determine output file path
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = Path(input_file).parent
    
    json_output_file = output_dir / f"{Path(input_file).stem}_llm_analysis.json"
    print(f"Output file will be: {json_output_file}")
    
    # Save to JSON
    with open(json_output_file, 'w', encoding='utf-8') as f:
        json.dump(llm_analysis_data, f, indent=2, default=str)
    
    print(f"Processed {processed_samples} samples with LLM analysis data")
    print(f"LLM analysis data saved to: {json_output_file}")
    
    return str(json_output_file)

def main():
    parser = argparse.ArgumentParser(description='Process LLM analysis from parquet files and save to JSON')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to the input parquet file with model responses')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save the output file (default: same as input file)')
    parser.add_argument('--max-samples', type=int, default=0,
                        help='Maximum number of samples to include (default: 0 = all samples)')
    parser.add_argument('--data_type', type=str,
                        choices=['regression', 'classification'],
                        help='Data type to extract')
    
    args = parser.parse_args()
    
    # Check if input file exists
    assert os.path.exists(args.input), f"File '{args.input}' not found"
    
    # Process the file
    output_file = process_parquet_file(
        args.input,
        args.output_dir,
        args.max_samples,
        args.data_type
    )
    print(f"LLM analysis processing complete! Saved to: {output_file}")

if __name__ == "__main__":
    main() 