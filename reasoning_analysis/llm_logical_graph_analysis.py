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

def analyze_node_dependencies(llm_json: Dict[str, Any]) -> Dict[str, Dict[str, Dict[str, int]]]:
    """
    Analyzes the dependencies between different node types in the logical graph.
    
    Args:
        llm_json: The parsed JSON data from the LLM's analysis with nodes and dependencies.
        
    Returns:
        A dictionary mapping each node type to the node types that follow it, with counts.
    """
    # Initialize dependency structure
    dependency_stats = {}
    # Create a map of node IDs to node types for quick lookup
    node_id_to_type = {node['id']: node['type'] for node in llm_json['nodes']}
    
    # Count occurrences of each dependency relationship
    for node in llm_json['nodes']:
        node_type = node['type']
        depends_on = node['depends_on']
        
        # Skip nodes without dependencies
        if depends_on is None or not depends_on:
            continue
            
        # Process dependencies - for each node the current node depends on
        for dep_id in depends_on:
            if dep_id in node_id_to_type:
                dep_type = node_id_to_type[dep_id]
                
                # This is the key change: dep_type is followed by node_type
                if dep_type not in dependency_stats:
                    dependency_stats[dep_type] = {}
                    
                if node_type not in dependency_stats[dep_type]:
                    dependency_stats[dep_type][node_type] = 0
                    
                dependency_stats[dep_type][node_type] += 1
    
    return dependency_stats

def analyze_confidence_transitions(llm_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Analyzes the confidence transitions between connected nodes in the logical graph.
    
    Args:
        llm_json: The parsed JSON data from the LLM's analysis with nodes and dependencies.
        
    Returns:
        A list of dictionaries containing source and target node types with their confidence values.
    """
    # Create maps for quick lookups
    node_id_to_type = {}
    node_id_to_prob = {}
    
    for node in llm_json['nodes']:
        node_id = node['id']
        node_id_to_type[node_id] = node['type']
        node_id_to_prob[node_id] = node['avg_prob']
    
    transitions = []
    
    # Process the dependencies to find connections
    for node in llm_json['nodes']:
        node_id = node['id']
        depends_on = node['depends_on']
        
        # Skip nodes without dependencies
        if depends_on is None or not depends_on:
            continue
            
        # Current node is the target
        target_type = node_id_to_type[node_id]
        target_prob = node_id_to_prob[node_id]
        
        # Process each dependency
        for source_id in depends_on:
            if source_id in node_id_to_type:
                source_type = node_id_to_type[source_id]
                source_prob = node_id_to_prob[source_id]
                
                # Add transition data
                transitions.append({
                    "source_node": source_type,
                    "source_avg_prob": source_prob,
                    "target_node": target_type,
                    "target_avg_prob": target_prob
                })
    
    return transitions

def process_llm_analysis_logical_graph(llm_json: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Analyzes the logical graph (CoT) structure within the LLM's analysis JSON.

    Args:
        llm_json: The parsed JSON data from the LLM's analysis, expected to contain a CoT graph.
        ground_truth: Ground truth data (currently unused in this specific analysis).

    Returns:
        A dictionary containing metrics about the CoT graph, or None if analysis fails.
    """
    # return {"summary": {"test": "test_cpg"}}
    # # import pdb; pdb.set_trace()
    # ##convert llm_json to dict
    # if not isinstance(llm_json, dict) or 'nodes' not in llm_json:
    #     # Use logging instead of print for warnings/errors
    #     logging.warning("Input 'llm_json' is not a valid dictionary or is missing 'nodes' key.")
    #     return None
    
    # try:
    #     # Call the analyze_hypotheses function from visualize_cot
    #     cot_metrics = analyze_hypotheses(llm_json)
    #     return cot_metrics
    # except Exception as e:
    #     # Use logging for errors
    #     logging.error(f"Error analyzing CoT graph: {e}", exc_info=True) # Add traceback info
    #     return None
    #统计llm_json中nodes每种类型的数量以及他们对应的avg_prob的平均值，请注意，avg_prob可能为None，为None的话就只统计数量，avg_prob的平均值设置为0
    node_type_counts = {}
    node_avg_prob_sum = {}
    node_avg_prob_counts = {}  # Track counts of non-None avg_prob values
    for node in llm_json['nodes']:
        node_type = node['type']
        avg_prob = node['avg_prob']
        if node_type not in node_type_counts:
            node_type_counts[node_type] = 0
            node_avg_prob_sum[node_type] = 0
            node_avg_prob_counts[node_type] = 0
        
        node_type_counts[node_type] += 1
        
        if avg_prob is not None:
            node_avg_prob_sum[node_type] += avg_prob
            node_avg_prob_counts[node_type] += 1
    
    for node_type in node_type_counts.keys():
        if node_avg_prob_counts[node_type] > 0:  # Only divide if there are non-None values
            node_avg_prob_sum[node_type] /= node_avg_prob_counts[node_type]
        else:
            node_avg_prob_sum[node_type] = None  # Set to None if all avg_prob values were None
    
    # Analyze node dependencies
    dependency_stats = analyze_node_dependencies(llm_json)
    
    # Analyze confidence transitions between nodes
    confidence_transitions = analyze_confidence_transitions(llm_json)
    
    return {
        "summary": {
            "node_type_counts": node_type_counts, 
            "node_avg_prob_sum": node_avg_prob_sum, 
            "dependency": dependency_stats,
            "node_confidence_transitions": confidence_transitions
        }
    }


import numpy as np
def get_robust_avg_prob(logprobs_list: List[float], remove_outliers: bool = True) -> float:
    #remove -9999.0
    if logprobs_list is None:
        return None
    if remove_outliers:
        logprobs_list = [x for x in logprobs_list if x != -9999.0]
    avg_logprob = np.mean(logprobs_list).item()
    avg_prob = np.exp(avg_logprob).item()
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
        # node['logprobs'] = node_info_dict['logprobs_list']
        # node['avg_logprob'] = node_info_dict['avg_logprob']
        node['n_tokens'] = node_info_dict['n_tokens']
        # node['sentence_tokens'] = node_info_dict['sentence_tokens']

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
    
    # 创建两个不同的嵌套结构字典
    # 标准二级嵌套
    regular_nested_metric_sums = defaultdict(lambda: defaultdict(float))
    regular_nested_metric_counts = defaultdict(lambda: defaultdict(int))
    # 三级嵌套，用于dependency
    dependency_metric_sums = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    dependency_metric_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    
    # 收集所有样本的节点置信度转换
    all_confidence_transitions = []

    processed_samples = 0
    valid_samples_for_avg = 0
    # Counter for visualizations
    visualization_count = 0
    # Maximum number of visualizations
    max_visualizations = 10

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
        if llm_json_str is None:
            continue
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

        # Generate and save visualization - limit to max_visualizations
        if llm_json_dict and 'nodes' in llm_json_dict and visualization_count < max_visualizations:

            sample_viz_dir = viz_main_dir / f"sample_{index}"
            sample_viz_dir.mkdir(exist_ok=True)
            try:
                # Call create_visualization with the dictionary and sample-specific dir
                # It will save 'cot_graph.png' inside sample_viz_dir
                create_visualization(
                    llm_json_dict,
                    output_dir=str(sample_viz_dir),
                    display=False, # Do not show the plot interactively
                    show_avg_prob=True
                )
                visualization_count += 1
                logging.info(f"Saved visualization {visualization_count}/{max_visualizations} for sample {index} to {sample_viz_dir}")
            except Exception as e:
                logging.error(f"Error generating visualization for sample {index}: {e}", exc_info=True)
        elif visualization_count >= max_visualizations and 'nodes' in llm_json_dict:
            pass
            # logging.info(f"Skipping visualization for sample {index}: reached the maximum limit of {max_visualizations} visualizations")
        elif 'nodes' not in llm_json_dict:
            logging.warning(f"Skipping visualization for sample {index} due to missing 'nodes' in parsed JSON.")

        # Continue with metric aggregation only if analysis was successful

        if cot_analysis_metrics and "summary" in cot_analysis_metrics:
            # import pdb; pdb.set_trace()

            summary = cot_analysis_metrics["summary"]
            valid_samples_for_avg += 1
            
            # Accumulate sums and counts
            for key, value in summary.items():
                if key == "dependency":
                    # Handle dependency dictionary which has a nested structure
                    if key not in dependency_metric_sums:
                        dependency_metric_sums[key] = defaultdict(lambda: defaultdict(float))
                        dependency_metric_counts[key] = defaultdict(lambda: defaultdict(int))
                    
                    # For each node type
                    for node_type, followers in value.items():
                        # For each follower type
                        for follower_type, count in followers.items():
                            if isinstance(count, (int, float)):
                                dependency_metric_sums[key][node_type][follower_type] += count
                                dependency_metric_counts[key][node_type][follower_type] += 1
                elif key == "node_confidence_transitions":
                    all_confidence_transitions.extend(value)
                elif isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, (int, float)): # Only average numeric types
                            regular_nested_metric_sums[key][sub_key] += sub_value
                            regular_nested_metric_counts[key][sub_key] += 1
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
    for key, sub_dict in regular_nested_metric_sums.items():
        if key not in average_metrics: # Create nested dict if not already from top-level
            average_metrics[key] = {}
        
        # Regular handling for doubly-nested structure
        for sub_key, total_sum in sub_dict.items():
            count = regular_nested_metric_counts[key][sub_key]
            average_metrics[key][sub_key] = total_sum / count if count > 0 else None
    
    # Handle dependency separately with its triple-nested structure
    if "dependency" in dependency_metric_sums:
        key = "dependency"
        if key not in average_metrics:
            average_metrics[key] = {}
            
        for node_type, followers in dependency_metric_sums[key].items():
            if node_type not in average_metrics[key]:
                average_metrics[key][node_type] = {}
            
            for follower_type, total_sum in followers.items():
                count = dependency_metric_counts[key][node_type][follower_type]
                average_metrics[key][node_type][follower_type] = total_sum / count if count > 0 else None

    # Update the total processed count and add averages to metadata
    llm_analysis_data["metadata"]["processed_samples"] = processed_samples
    llm_analysis_data["metadata"]["average_summary_metrics"] = average_metrics
    llm_analysis_data["metadata"]["samples_used_for_averages"] = valid_samples_for_avg
    llm_analysis_data["metadata"]["visualizations_created"] = visualization_count
    llm_analysis_data["metadata"]["all_confidence_transitions"] = all_confidence_transitions

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
    print(f"Created {visualization_count} visualizations (limited to {max_visualizations})")
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