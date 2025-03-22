import os
import re
import pandas as pd
import numpy as np
import glob
from collections import defaultdict
import json
import sys

def get_parquet_files(directory, pattern_1, pattern_2):
    """
    Get all parquet files in the directory that contain both pattern_1 and pattern_2
    """
    files = glob.glob(os.path.join(directory, "*.parquet"))
    filtered_files = [f for f in files if pattern_1 in f and pattern_2 in f]
    return filtered_files

def extract_iteration_number(filename):
    """
    Extract iteration number from filename
    For files like "iter0_test_single_Mar21_2_iter1_gen_test.parquet", 
    the iteration number is 1 (after "iter" and just before "gen_test")
    """
    # First, try to extract the iteration number from patterns like "iter1_gen_test"
    match = re.search(r'_iter(\d+)_gen_test', filename)
    if match:
        return int(match.group(1))
    
    # If that fails, try to extract from patterns like "iter0_test"
    match = re.search(r'iter(\d+)_test', filename)
    if match:
        return int(match.group(1))
    
    return -1

def blobs_reward_fn(response, ground_truth):
    """
    Evaluate if the model's response is correct based on the ground truth
    """
    response_extract = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
    if response_extract is not None and response_extract.group(1).strip().isdigit():
        response_class = int(response_extract.group(1).strip())
    else:
        return 0
    return response_class == ground_truth['label']

def analyze_file(file_path):
    """
    Analyze a parquet file and return the accuracy and correct prompt indices
    """
    df = pd.read_parquet(file_path)
    total = len(df)
    correct_count = 0
    correct_indices = []
    
    for idx, row in df.iterrows():
        prompt = row['prompt']
        response = row['responses'][0]  # Assuming first response is the one to evaluate
        ground_truth = row['reward_model']['ground_truth']
        
        is_correct = blobs_reward_fn(response, ground_truth)
        if is_correct:
            correct_count += 1
            # Using prompt as a unique identifier as required
            correct_indices.append(str(prompt))  # Convert to string for consistent handling
    
    accuracy = correct_count / total if total > 0 else 0
    return accuracy, correct_indices

def calculate_overlap(set1, set2):
    """Calculate overlap ratio between two sets"""
    if not set1 or not set2:
        return 0.0
    intersection = set1.intersection(set2)
    return len(intersection) / len(set1.union(set2))

def write_results(output_dir, iteration_results):
    """Write results to output directory"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Sort iterations
    iterations = sorted(iteration_results.keys())
    
    # Write accuracy per iteration
    with open(os.path.join(output_dir, "accuracies.json"), "w") as f:
        accuracies = {f"iter{iter_num}": data["accuracy"] for iter_num, data in iteration_results.items()}
        json.dump(accuracies, f, indent=2)
    
    # Write correct prompts per iteration
    for iter_num in iterations:
        correct_prompts = list(iteration_results[iter_num]["correct_prompts"])
        with open(os.path.join(output_dir, f"iter{iter_num}_correct_prompts.json"), "w") as f:
            json.dump(correct_prompts, f, indent=2)
    
    # Calculate overlap with previous iteration
    overlaps_with_prev = {}
    for i, iter_num in enumerate(iterations):
        if i > 0:
            prev_iter = iterations[i-1]
            curr_prompts = iteration_results[iter_num]["correct_prompts"]
            prev_prompts = iteration_results[prev_iter]["correct_prompts"]
            overlap = calculate_overlap(curr_prompts, prev_prompts)
            overlaps_with_prev[f"iter{iter_num}_with_iter{prev_iter}"] = overlap
    
    # Calculate overlap with first iteration
    overlaps_with_first = {}
    if iterations:
        first_iter = iterations[0]
        first_prompts = iteration_results[first_iter]["correct_prompts"]
        for iter_num in iterations[1:]:
            curr_prompts = iteration_results[iter_num]["correct_prompts"]
            overlap = calculate_overlap(curr_prompts, first_prompts)
            overlaps_with_first[f"iter{iter_num}_with_iter{first_iter}"] = overlap
    
    # Write overlaps to files
    with open(os.path.join(output_dir, "overlaps_with_prev.json"), "w") as f:
        json.dump(overlaps_with_prev, f, indent=2)
    
    with open(os.path.join(output_dir, "overlaps_with_first.json"), "w") as f:
        json.dump(overlaps_with_first, f, indent=2)
    
    # Print summary
    print("\nSummary:")
    print("=" * 50)
    print("Accuracies:")
    for iter_num in iterations:
        print(f"  Iteration {iter_num}: {iteration_results[iter_num]['accuracy']:.4f}")
    
    print("\nOverlap with Previous Iteration:")
    for key, value in overlaps_with_prev.items():
        print(f"  {key}: {value:.4f}")
    
    print("\nOverlap with First Iteration:")
    for key, value in overlaps_with_first.items():
        print(f"  {key}: {value:.4f}")

def main():
    if len(sys.argv) < 3:
        print("Usage: python analyze_iterative-sft.py <input_directory> <output_directory> [pattern1] [pattern2]")
        print("Default patterns: 'test_single_Mar21_2_iter' and 'gen_test'")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    pattern1 = sys.argv[3] if len(sys.argv) > 3 else "test_single_Mar21_2_iter"
    pattern2 = sys.argv[4] if len(sys.argv) > 4 else "gen_test"
    
    # Get matching parquet files
    files = get_parquet_files(input_dir, pattern1, pattern2)
    print(f"Found {len(files)} matching files:")
    for f in files:
        print(f"  - {os.path.basename(f)}")
    
    # Analyze each file
    iteration_results = {}
    for file_path in files:
        iter_num = extract_iteration_number(file_path)
        if iter_num >= 0:
            print(f"\nAnalyzing file: {os.path.basename(file_path)}")
            print(f"Extracted iteration number: {iter_num}")
            accuracy, correct_indices = analyze_file(file_path)
            print(f"Iteration {iter_num} accuracy: {accuracy:.4f}, correct examples: {len(correct_indices)}")
            
            # Convert list of prompt strings to set for comparison
            correct_prompts = set(correct_indices)
            
            iteration_results[iter_num] = {
                "accuracy": accuracy,
                "correct_prompts": correct_prompts,
                "file_path": file_path
            }
    
    if not iteration_results:
        print("No valid iteration data found. Check file patterns and naming.")
        sys.exit(1)
    
    # Write results
    write_results(output_dir, iteration_results)
    print(f"\nResults written to {output_dir}")

if __name__ == "__main__":
    main()
