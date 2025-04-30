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
from collections import defaultdict
import logging # Add logging import

# Import required functions directly from data_processing
from data_processing import (
    extract_and_execute_model_functions,
    create_model_evaluation_table,
    get_ground_truth
)
# Import visualization function
# from visualize_cot import analyze_hypotheses, create_visualization # Import create_visualization
from visualize_cpg import analyze_cpg, create_visualization # Import create_visualization

def process_llm_analysis_logical_graph(llm_json: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Analyzes the logical graph (CoT) structure within the LLM's analysis JSON.

    Args:
        llm_json: The parsed JSON data from the LLM's analysis, expected to contain a CoT graph.
        ground_truth: Ground truth data (currently unused in this specific analysis).

    Returns:
        A dictionary containing metrics about the CoT graph, or None if analysis fails.
    """
    return {"summary": {"test": "test_cpg"}}
    # import pdb; pdb.set_trace()
    ##convert llm_json to dict
    if not isinstance(llm_json, dict) or 'nodes' not in llm_json:
        # Use logging instead of print for warnings/errors
        logging.warning("Input 'llm_json' is not a valid dictionary or is missing 'nodes' key.")
        return None
    
    try:
        # Call the analyze_hypotheses function from visualize_cot
        cot_metrics = analyze_hypotheses(llm_json)
        return cot_metrics
    except Exception as e:
        # Use logging for errors
        logging.error(f"Error analyzing CoT graph: {e}", exc_info=True) # Add traceback info
        return None
    
import numpy as np
def get_robust_avg_prob(logprobs_list: List[float], remove_outliers: bool = True) -> float:
    #remove -9999.0
    if remove_outliers:
        logprobs_list = [x for x in logprobs_list if x != -9999.0]
    avg_logprob = np.mean(logprobs_list).item()
    avg_prob = np.exp(avg_logprob)
    return avg_prob

def postprocess_llm_json_dict(llm_json_dict: Dict[str, Any], llm_analysis_splitted_reasoning_dict: Dict[str, Any]) -> Dict[str, Any]:
    llm_analysis_splitted_reasoning_dict  = json.loads(llm_analysis_splitted_reasoning_dict)
    # import pdb; pdb.set_trace()
    reasoning_dict_ids = list(llm_analysis_splitted_reasoning_dict.keys())
    llm_json_dict_ids = [str(node['id']) for node in llm_json_dict["nodes"]]
    try:
        #llm_json_dict_ids should be a subset of reasoning_dict_ids, every id in llm_json_dict_ids should be in reasoning_dict_ids
        assert set(llm_json_dict_ids) <= set(reasoning_dict_ids), "llm_json_dict_ids is not a subset of reasoning_dict_ids"
        
    except Exception as e:
        return None
        # import pdb; pdb.set_trace()
    
    for node in llm_json_dict["nodes"]:
        node_info_dict = llm_analysis_splitted_reasoning_dict.get(str(node['id']))
        
        
        node['text'] = node_info_dict['sentence']
        node['logprobs'] = node_info_dict['logprobs_list']
        # node['avg_logprob'] = node_info_dict['avg_logprob']
        node['n_tokens'] = node_info_dict['n_tokens']
        node['sentence_tokens'] = node_info_dict['sentence_tokens']

        # node['avg_prob'] = node_info_dict['avg_prob']
        node['avg_prob'] = get_robust_avg_prob(node_info_dict['logprobs_list'], remove_outliers=True)
    
    return llm_json_dict



def process_parquet_file(input_file: str, output_dir: Optional[str] = None, 
                         max_samples: int = 0):
    """
    Process LLM analysis from parquet file and save results to JSON
    
    Args:
        input_file: Path to the input parquet file
        output_dir: Directory to save the output file (default: same as input file)
        max_samples: Maximum number of samples to include (default: 0 = all samples)
        
    Returns:
        Path to the output JSON file
    """
    print(f"Reading parquet file: {input_file}")
    input_path = Path(input_file)
    base_dir = input_path.parent
    viz_main_dir = base_dir / "visualize_graph"
    viz_main_dir.mkdir(parents=True, exist_ok=True) # Create the main visualization directory
    print(f"Saving visualizations to: {viz_main_dir}")

    try:
        df = pd.read_parquet(input_file)
        print(f"DataFrame loaded with {len(df)} rows and columns: {df.columns.tolist()}")
    except Exception as e:
        print(f"Error reading parquet file: {e}")
        return None

    llm_analysis_data = {
        "metadata": {
            "source_file": input_file,
            "processed_samples": 0,
            "max_samples_limit": max_samples,
            "average_summary_metrics": None
        },
        "samples": []
    }

    # Use defaultdicts for easier aggregation
    metric_sums = defaultdict(float) # Initialize top-level keys to 0.0
    metric_counts = defaultdict(int) # Initialize top-level keys to 0
    nested_metric_sums = defaultdict(lambda: defaultdict(float)) # For nested keys
    nested_metric_counts = defaultdict(lambda: defaultdict(int)) # For nested keys

    processed_samples = 0
    valid_samples_for_avg = 0

    # Limit rows if max_samples is set
    if max_samples > 0:
        df = df.head(max_samples)

    for index, row in df.iterrows():
        # Check if this row has LLM analysis data
        # if 'llm_analysis_extracted_json' not in row or pd.isna(row['llm_analysis_extracted_json']):
        #     logging.warning(f"Skipping row {index} due to missing or NaN 'llm_analysis_extracted_json'")
        #     continue
            
        # Get ground truth
        # ground_truth = get_ground_truth(row)
        # if ground_truth is None or 'in_context_samples' not in ground_truth:
        #     continue
            
        # Get data source if available
        data_source = row.get('data_source', 'unknown')
        prompt = row.get('prompt', 'unknown')
        # Renamed variable to avoid conflict with function name
        ground_truth_data = row.get('get_ground_truth', 'unknown') 
        
        # Process LLM analysis - Parse JSON first
        llm_json_str = row['llm_analysis_extracted_json']
        llm_json_dict = None
        try:
            llm_json_dict = json.loads(llm_json_str)
        except json.JSONDecodeError as e:
            logging.error(f"Skipping row {index} due to JSONDecodeError in 'llm_analysis_extracted_json': {e}")
            # Store error info for this sample if needed, or just continue
            sample_data = {
                "row_index": index,
                "llm_json_dict": {"error": "Failed to parse JSON", "original_string": llm_json_str},
                "cot_analysis_metrics": None,
                "prompt": prompt,
                "data_source": data_source,
                "get_ground_truth": ground_truth_data
            }
            llm_analysis_data["samples"].append(sample_data)
            continue # Skip analysis and visualization for this sample

        # Ensure llm_json_dict is a dictionary before proceeding
        if not isinstance(llm_json_dict, dict):
             logging.error(f"Skipping row {index} because parsed JSON is not a dictionary: {type(llm_json_dict)}")
             # Store error info
             sample_data = {
                "row_index": index,
                "llm_json_dict": {"error": "Parsed JSON is not a dictionary", "parsed_data": llm_json_dict},
                "cot_analysis_metrics": None,
                "prompt": prompt,
                "data_source": data_source,
                "get_ground_truth": ground_truth_data
             }
             llm_analysis_data["samples"].append(sample_data)
             continue

        # Analyze the CoT graph structure using the parsed dictionary
        
        llm_json_dict = postprocess_llm_json_dict(llm_json_dict, row["llm_analysis_splitted_reasoning_dict"])
        if llm_json_dict is None:
            continue
        cot_analysis_metrics = process_llm_analysis_logical_graph(llm_json_dict)

        # Generate and save visualization
        if llm_json_dict and 'nodes' in llm_json_dict: # Check if dict has nodes before visualizing
            sample_viz_dir = viz_main_dir / f"sample_{index}"
            sample_viz_dir.mkdir(exist_ok=True)
            try:
                # Call create_visualization with the dictionary and sample-specific dir
                # It will save 'cot_graph.png' inside sample_viz_dir
                create_visualization(
                    llm_json_dict,
                    output_dir=str(sample_viz_dir),
                    display=False, # Do not show the plot interactively
                )
                # create_visualization(
                #     llm_json_dict,
                #     output_dir=str(sample_viz_dir),
                #     display=False, # Do not show the plot interactively
                # )
                logging.info(f"Saved visualization for sample {index} to {sample_viz_dir}")
            except Exception as e:
                logging.error(f"Error generating visualization for sample {index}: {e}", exc_info=True)
        else:
             logging.warning(f"Skipping visualization for sample {index} due to missing 'nodes' in parsed JSON.")

        # Continue with metric aggregation only if analysis was successful
        if cot_analysis_metrics and "summary" in cot_analysis_metrics:
            # import pdb; pdb.set_trace()
            summary = cot_analysis_metrics["summary"]
            valid_samples_for_avg += 1
            
            # Accumulate sums and counts
            for key, value in summary.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, (int, float)): # Only average numeric types
                            nested_metric_sums[key][sub_key] += sub_value
                            nested_metric_counts[key][sub_key] += 1
                elif isinstance(value, (int, float)): # Handle top-level numeric keys
                     metric_sums[key] += value
                     metric_counts[key] += 1
                 
            # Store results for this sample (original metrics)
            sample_data = {
                "row_index": index,
                "llm_json_dict": llm_json_dict, # Store the dict directly
                "cot_analysis_metrics": cot_analysis_metrics,
                "prompt": prompt,
                "data_source": data_source,
                "get_ground_truth": ground_truth_data
                # Potentially add other row data or evaluation results here later
            }
            llm_analysis_data["samples"].append(sample_data)
            processed_samples += 1
        
    # Calculate averages
    average_metrics = {}
    # Average top-level keys
    for key, total_sum in metric_sums.items():
        count = metric_counts[key]
        average_metrics[key] = total_sum / count if count > 0 else None
        
    # Average nested keys
    for key, sub_dict in nested_metric_sums.items():
        if key not in average_metrics: # Create nested dict if not already from top-level
             average_metrics[key] = {}
        for sub_key, total_sum in sub_dict.items():
            count = nested_metric_counts[key][sub_key]
            average_metrics[key][sub_key] = total_sum / count if count > 0 else None
            
    # Update the total processed count and add averages to metadata
    llm_analysis_data["metadata"]["processed_samples"] = processed_samples
    llm_analysis_data["metadata"]["average_summary_metrics"] = average_metrics
    llm_analysis_data["metadata"]["samples_used_for_averages"] = valid_samples_for_avg

    # Determine output file path
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = Path(input_file).parent
    
    json_output_file = output_dir / f"{Path(input_file).stem}_logical_graph_llm_analysis.json"
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
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input):
         print(f"Error: File '{args.input}' not found") # Use print for startup errors
         return # Exit if file not found

    # Setup logging
    log_file = Path(args.output_dir if args.output_dir else Path(args.input).parent) / f"{Path(args.input).stem}_processing.log"
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(log_file), logging.StreamHandler()])
    logging.info(f"Starting processing for {args.input}")
    logging.info(f"Saving logs to: {log_file}")

    # Process the file
    output_file = process_parquet_file(
        args.input,
        args.output_dir,
        args.max_samples
    )
    print(f"LLM analysis processing complete! Saved to: {output_file}")

if __name__ == "__main__":
    main() 