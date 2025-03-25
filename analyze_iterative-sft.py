import os
import re
import pandas as pd
import numpy as np
import glob
from collections import defaultdict
import json
import sys
import random
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Add a custom JSON encoder to handle numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)

def get_parquet_files(directory, project_name):
    """
    Get all parquet files in the directory that contain project_name + "_iter" and "gen_test"
    """
    pattern1 = f"{project_name}_iter"
    pattern2 = "gen_test"
    files = glob.glob(os.path.join(directory, "*.parquet"))
    filtered_files = [f for f in files if pattern1 in f and pattern2 in f]
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

def format_example_for_human(prompt, response, is_correct, ground_truth):
    """Format example in a human-readable way"""
    formatted = "=" * 50 + "\n"
    
    # Handle prompt which could be a dictionary or a string
    if isinstance(prompt, dict):
        formatted += "PROMPT:\n" + json.dumps(prompt, indent=2, cls=NumpyEncoder) + "\n\n"
    else:
        formatted += "PROMPT:\n" + str(prompt) + "\n\n"
    
    formatted += "RESPONSE:\n" + response + "\n\n"
    
    # Extract predicted answer
    response_extract = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
    predicted = response_extract.group(1).strip() if response_extract else "No valid answer"
    
    formatted += f"PREDICTED: {predicted}\n"
    
    # Handle ground_truth which could be a dictionary
    if isinstance(ground_truth, dict):
        label = ground_truth.get('label', 'Unknown')
        formatted += f"CORRECT ANSWER: {label}\n"
    else:
        formatted += f"CORRECT ANSWER: {ground_truth}\n"
    
    formatted += f"IS CORRECT: {is_correct}\n"
    formatted += "=" * 50 + "\n\n"
    
    return formatted

def analyze_file(file_path):
    """
    Analyze a parquet file and return the accuracy, all examples with correctness info
    """
    df = pd.read_parquet(file_path)
    total = len(df)
    correct_count = 0
    all_examples = []
    
    for idx, row in df.iterrows():
        prompt = row['prompt']
        response = row['responses'][0]  # Assuming first response is the one to evaluate
        ground_truth = row['reward_model']['ground_truth']
        
        is_correct = blobs_reward_fn(response, ground_truth)
        if is_correct:
            correct_count += 1
        
        # Save all examples with correctness info
        all_examples.append({
            "prompt": prompt,
            "response": response,
            "ground_truth": ground_truth,
            "is_correct": bool(is_correct)
        })
    
    accuracy = correct_count / total if total > 0 else 0
    return accuracy, all_examples

def plot_accuracies(iterations, accuracies, output_path):
    """Create and save accuracy plot as PDF"""
    plt.figure(figsize=(8, 5))
    plt.plot(iterations, accuracies, 'o-', linewidth=2)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Accuracy across Iterations', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(0, 1.05)
    
    # Add iteration numbers on x-axis
    plt.xticks(iterations)
    
    # Format y-axis as percentage
    plt.gca().yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    
    # Make plot tight and clean for publication
    plt.tight_layout()
    
    # Save as PDF
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()

def sample_examples(examples, n_correct=3, n_incorrect=3, seed=0):
    """Sample n_correct correct examples and n_incorrect incorrect examples"""
    random.seed(seed)
    
    correct_examples = [ex for ex in examples if ex["is_correct"]]
    incorrect_examples = [ex for ex in examples if not ex["is_correct"]]
    
    # Sample correct examples (or take all if fewer than requested)
    sampled_correct = random.sample(correct_examples, min(n_correct, len(correct_examples)))
    
    # Sample incorrect examples (or take all if fewer than requested)
    sampled_incorrect = random.sample(incorrect_examples, min(n_incorrect, len(incorrect_examples)))
    
    # Prepare explanation if we couldn't get the requested number
    correct_explanation = ""
    if len(sampled_correct) < n_correct:
        correct_explanation = f"Note: Only {len(sampled_correct)} correct examples available (requested {n_correct})\n\n"
    
    incorrect_explanation = ""
    if len(sampled_incorrect) < n_incorrect:
        incorrect_explanation = f"Note: Only {len(sampled_incorrect)} incorrect examples available (requested {n_incorrect})\n\n"
    
    return sampled_correct, sampled_incorrect, correct_explanation, incorrect_explanation

def get_prompt_key(prompt):
    """Convert a prompt to a hashable key for set operations"""
    if isinstance(prompt, dict):
        # Sort the keys to ensure consistent serialization
        return json.dumps(prompt, sort_keys=True, cls=NumpyEncoder)
    elif isinstance(prompt, (np.ndarray, list, tuple)):
        # Convert arrays/lists to tuple of strings for hashing
        return str(prompt)
    return str(prompt)

def write_results(output_dir, project_name, iteration_results):
    """Write results to output directory with new format"""
    # Create the complete output directory path
    full_output_dir = os.path.join(output_dir, project_name)
    os.makedirs(full_output_dir, exist_ok=True)
    
    # Sort iterations
    iterations = sorted(iteration_results.keys())
    
    # Prepare data for accuracy plot
    iter_numbers = []
    accuracies = []
    
    # Process results for each iteration
    for iter_num in iterations:
        iter_data = iteration_results[iter_num]
        accuracy = iter_data["accuracy"]
        examples = iter_data["examples"]
        
        # Add to plot data
        iter_numbers.append(iter_num)
        accuracies.append(accuracy)
        
        # Create directory for this iteration
        iter_dir = os.path.join(full_output_dir, f"iter{iter_num}")
        os.makedirs(iter_dir, exist_ok=True)
        
        # Save all examples with correctness info (in a structured format)
        with open(os.path.join(iter_dir, "all_examples.json"), "w") as f:
            json.dump(examples, f, indent=2, cls=NumpyEncoder)
        
        # Sample some correct and incorrect examples
        sampled_correct, sampled_incorrect, correct_explanation, incorrect_explanation = sample_examples(
            examples, n_correct=3, n_incorrect=3
        )
        
        # Save sampled examples in human-readable format (text files)
        with open(os.path.join(iter_dir, "sample_correct_examples.txt"), "w") as f:
            f.write(correct_explanation)
            for ex in sampled_correct:
                f.write(format_example_for_human(
                    ex["prompt"], ex["response"], ex["is_correct"], ex["ground_truth"]
                ))
        
        with open(os.path.join(iter_dir, "sample_incorrect_examples.txt"), "w") as f:
            f.write(incorrect_explanation)
            for ex in sampled_incorrect:
                f.write(format_example_for_human(
                    ex["prompt"], ex["response"], ex["is_correct"], ex["ground_truth"]
                ))
        
        # Save sampled examples as JSON files
        with open(os.path.join(iter_dir, "sample_correct_examples.json"), "w") as f:
            json.dump(sampled_correct, f, indent=2, cls=NumpyEncoder)
        
        with open(os.path.join(iter_dir, "sample_incorrect_examples.json"), "w") as f:
            json.dump(sampled_incorrect, f, indent=2, cls=NumpyEncoder)
    
    # Generate and save accuracy plot
    plot_accuracies(iter_numbers, accuracies, os.path.join(full_output_dir, "accuracy_plot.pdf"))
    
    # Save accuracies as JSON
    with open(os.path.join(full_output_dir, "accuracies.json"), "w") as f:
        accuracies_dict = {f"iter{iter_num}": iteration_results[iter_num]["accuracy"] for iter_num in iterations}
        json.dump(accuracies_dict, f, indent=2)
    
    # Calculate overlap with previous iteration
    overlaps_with_prev = {}
    for i, iter_num in enumerate(iterations):
        if i > 0:
            prev_iter = iterations[i-1]
            curr_correct = {get_prompt_key(ex["prompt"]) for ex in iteration_results[iter_num]["examples"] if ex["is_correct"]}
            prev_correct = {get_prompt_key(ex["prompt"]) for ex in iteration_results[prev_iter]["examples"] if ex["is_correct"]}
            
            if not curr_correct or not prev_correct:
                overlap = 0.0
            else:
                intersection = curr_correct.intersection(prev_correct)
                overlap = len(intersection) / len(curr_correct.union(prev_correct))
            
            overlaps_with_prev[f"iter{iter_num}_with_iter{prev_iter}"] = overlap
    
    # Calculate overlap with first iteration
    overlaps_with_first = {}
    if iterations:
        first_iter = iterations[0]
        first_correct = {get_prompt_key(ex["prompt"]) for ex in iteration_results[first_iter]["examples"] if ex["is_correct"]}
        
        for iter_num in iterations[1:]:
            curr_correct = {get_prompt_key(ex["prompt"]) for ex in iteration_results[iter_num]["examples"] if ex["is_correct"]}
            
            if not curr_correct or not first_correct:
                overlap = 0.0
            else:
                intersection = curr_correct.intersection(first_correct)
                overlap = len(intersection) / len(curr_correct.union(first_correct))
            
            overlaps_with_first[f"iter{iter_num}_with_iter{first_iter}"] = overlap
    
    # Write overlaps to files
    with open(os.path.join(full_output_dir, "overlaps_with_prev.json"), "w") as f:
        json.dump(overlaps_with_prev, f, indent=2)
    
    with open(os.path.join(full_output_dir, "overlaps_with_first.json"), "w") as f:
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
    
    print(f"\nPlot saved to: {os.path.join(full_output_dir, 'accuracy_plot.pdf')}")
    print(f"Detailed results and examples saved to: {full_output_dir}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_iterative-sft.py <project_name>")
        print("Example: python analyze_iterative-sft.py NewSetting_50shot_Mar22")
        sys.exit(1)
    
    # Default directories
    input_dir = "/staging/szhang967/blobs"
    output_base_dir = "/staging/szhang967"
    
    # Project name is the only required parameter
    project_name = sys.argv[1]
    
    # Create the complete output directory path
    output_dir = os.path.join(output_base_dir, "iterative_sft_analysis")
    
    # Get matching parquet files
    files = get_parquet_files(input_dir, project_name)
    
    if not files:
        print(f"No matching files found for project '{project_name}' in {input_dir}")
        sys.exit(1)
        
    print(f"Found {len(files)} matching files:")
    for f in files:
        print(f"  - {os.path.basename(f)}")
    
    # Check for duplicate iteration numbers
    iter_to_files = defaultdict(list)
    for file_path in files:
        iter_num = extract_iteration_number(file_path)
        if iter_num >= 0:
            iter_to_files[iter_num].append(file_path)
    
    # Assert that each iteration has exactly one file
    has_error = False
    for iter_num, iter_files in iter_to_files.items():
        if len(iter_files) != 1:
            print(f"ERROR: Iteration {iter_num} has {len(iter_files)} matching files, expected exactly 1.")
            for f in iter_files:
                print(f"  - {os.path.basename(f)}")
            has_error = True
    
    if has_error:
        print("Validation failed: Some iterations have duplicate or missing files.")
        sys.exit(1)
    
    # Analyze each file
    iteration_results = {}
    for file_path in files:
        iter_num = extract_iteration_number(file_path)
        if iter_num >= 0:
            print(f"\nAnalyzing file: {os.path.basename(file_path)}")
            print(f"Extracted iteration number: {iter_num}")
            accuracy, examples = analyze_file(file_path)
            
            correct_examples = sum(1 for ex in examples if ex["is_correct"])
            print(f"Iteration {iter_num} accuracy: {accuracy:.4f}, correct examples: {correct_examples}")
            
            iteration_results[iter_num] = {
                "accuracy": accuracy,
                "examples": examples,
                "file_path": file_path
            }
    
    if not iteration_results:
        print("No valid iteration data found. Check file patterns and naming.")
        sys.exit(1)
    
    # Write results
    write_results(output_dir, project_name, iteration_results)
    print(f"\nResults written to {os.path.join(output_dir, project_name)}")

if __name__ == "__main__":
    main()
