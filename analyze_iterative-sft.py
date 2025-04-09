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
from transformers import AutoTokenizer

# Add import for select_reward_fn
from verl.trainer.ppo.helper import _select_rm_score_fn as select_reward_fn

# Load Qwen tokenizer
try:
    qwen_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct", trust_remote_code=True)
except Exception as e:
    print(f"Warning: Failed to load Qwen tokenizer: {e}")
    print("Will use approximate token count if tokenizer is unavailable.")
    qwen_tokenizer = None

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
    This function is kept for backward compatibility
    """
    response_extract = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
    if response_extract is not None and response_extract.group(1).strip().isdigit():
        response_class = int(response_extract.group(1).strip())
    else:
        return 0
    return response_class == ground_truth['label']

def format_example_for_human(prompt, response, is_correct, ground_truth, score=None, data_source=None):
    """Format example in a human-readable way"""
    formatted = "=" * 50 + "\n"
    
    # Handle prompt which could be a dictionary or a string
    if isinstance(prompt, dict):
        formatted += "PROMPT:\n" + json.dumps(prompt, indent=2, cls=NumpyEncoder) + "\n\n"
    else:
        formatted += "PROMPT:\n" + str(prompt) + "\n\n"
    
    # Add data source if available
    if data_source:
        formatted += f"DATA SOURCE: {data_source}\n\n"
    
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
    
    # Add score if available
    if score is not None:
        formatted += f"SCORE: {score}\n"
    
    formatted += "=" * 50 + "\n\n"
    
    return formatted

def get_token_count(text):
    """
    Count the number of tokens in the text using Qwen tokenizer
    If tokenizer is unavailable, use a simple approximation
    """
    if qwen_tokenizer is not None:
        return len(qwen_tokenizer.encode(text))
    else:
        # Approximate token count (rough estimation)
        return len(text.split())

def analyze_token_lengths(examples):
    """
    Analyze the token lengths of responses in the examples
    Returns statistics and token length distribution
    """
    token_lengths = [get_token_count(ex["response"]) for ex in examples]
    
    # Calculate statistics
    stats = {
        "median": np.median(token_lengths),
        "mean": np.mean(token_lengths),
        "min": np.min(token_lengths),
        "max": np.max(token_lengths),
        "std": np.std(token_lengths),
        "counts": token_lengths
    }
    
    return stats

def plot_token_distribution(token_stats, output_path):
    """Create and save token length distribution histogram as PDF"""
    plt.figure(figsize=(8, 5))
    plt.hist(token_stats["counts"], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(token_stats["median"], color='red', linestyle='dashed', linewidth=1, label=f'Median: {token_stats["median"]:.1f}')
    plt.axvline(token_stats["mean"], color='green', linestyle='dashed', linewidth=1, label=f'Mean: {token_stats["mean"]:.1f}')
    
    plt.xlabel('Token Count', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Response Token Length Distribution', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend()
    
    # Make plot tight and clean for publication
    plt.tight_layout()
    
    # Save as PDF
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()

def plot_token_stats_across_iterations(iterations, token_stats_by_iter, output_path):
    """Create and save token length statistics across iterations as PDF"""
    plt.figure(figsize=(10, 6))
    
    # Extract data for plotting
    medians = [token_stats_by_iter[iter_num]["median"] for iter_num in iterations]
    means = [token_stats_by_iter[iter_num]["mean"] for iter_num in iterations]
    mins = [token_stats_by_iter[iter_num]["min"] for iter_num in iterations]
    maxes = [token_stats_by_iter[iter_num]["max"] for iter_num in iterations]
    
    plt.plot(iterations, medians, 'o-', color='red', linewidth=2, label='Median')
    plt.plot(iterations, means, 's-', color='green', linewidth=2, label='Mean')
    plt.plot(iterations, mins, '^--', color='blue', linewidth=1.5, label='Min', alpha=0.7)
    plt.plot(iterations, maxes, 'v--', color='purple', linewidth=1.5, label='Max', alpha=0.7)
    
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Token Count', fontsize=12)
    plt.title('Response Token Length Statistics Across Iterations', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Add iteration numbers on x-axis
    plt.xticks(iterations)
    
    # Make plot tight and clean for publication
    plt.tight_layout()
    
    # Save as PDF
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()

def plot_token_boxplot(iterations, token_stats_by_iter, output_path):
    """Create and save box plot of token length distributions across iterations"""
    plt.figure(figsize=(10, 6))
    
    # Prepare data for boxplot
    data = [token_stats_by_iter[iter_num]["counts"] for iter_num in iterations]
    
    # Create boxplot
    bp = plt.boxplot(data, patch_artist=True, labels=[f"Iter {i}" for i in iterations])
    
    # Customize boxplot colors
    for box in bp['boxes']:
        box.set(facecolor='skyblue', alpha=0.7)
    
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Token Count', fontsize=12)
    plt.title('Response Token Length Distribution Comparison', fontsize=14)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Make plot tight and clean for publication
    plt.tight_layout()
    
    # Save as PDF
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()

def analyze_file(file_path):
    """
    Analyze a parquet file and return the accuracy, all examples with correctness info
    Using the scoring mechanism from main_eval.py
    """
    df = pd.read_parquet(file_path)
    total = len(df)
    correct_count = 0
    all_examples = []
    
    for idx, row in df.iterrows():
        prompt = row['prompt']
        responses = row['responses']  # Get all responses
        if 'data_source' in row:
            data_source = row['data_source']
        else:
            # Default to blobs if data_source not available
            data_source = "blobs"
        
        reward_model_data = row['reward_model']
        ground_truth = reward_model_data['ground_truth']
        
        try:
            # Use select_reward_fn to get the appropriate reward function
            reward_fn = select_reward_fn(data_source)
        except (ImportError, NotImplementedError):
            # Fallback to blobs_reward_fn if data_source is not supported
            reward_fn = blobs_reward_fn
        
        # Process all responses and find the best score
        response = responses[0]  # Default to the first response
        score_lst = []
        
        for r in responses:
            try:
                score = reward_fn(r, ground_truth)
                score_lst.append(score)
            except Exception as e:
                print(f"Warning: Error calculating score: {e}")
                score_lst.append(0)
        
        # Use max score to determine correctness
        if score_lst:
            max_score = np.max(score_lst)
            is_correct = max_score == 1
            # Use the response with the best score for analyzing
            if len(score_lst) > 1:
                response = responses[np.argmax(score_lst)]
        else:
            is_correct = False
            max_score = 0
        
        if is_correct:
            correct_count += 1
        
        # Calculate token length
        token_count = get_token_count(response)
        
        # Save all examples with correctness info and token count
        all_examples.append({
            "prompt": prompt,
            "response": response,
            "responses": responses,
            "ground_truth": ground_truth,
            "is_correct": bool(is_correct),
            "score": max_score,
            "token_count": token_count,
            "data_source": data_source
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

def sample_examples(examples, n_correct=100, n_incorrect=3, seed=0):
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

def save_token_stats_csv(iterations, token_stats_by_iter, accuracy_by_iter, output_path):
    """Save token length statistics to CSV file"""
    # Prepare data for CSV
    data = {
        "iteration": iterations,
        "accuracy": [accuracy_by_iter[iter_num] for iter_num in iterations],
        "median_tokens": [token_stats_by_iter[iter_num]["median"] for iter_num in iterations],
        "mean_tokens": [token_stats_by_iter[iter_num]["mean"] for iter_num in iterations],
        "min_tokens": [token_stats_by_iter[iter_num]["min"] for iter_num in iterations],
        "max_tokens": [token_stats_by_iter[iter_num]["max"] for iter_num in iterations],
        "std_tokens": [token_stats_by_iter[iter_num]["std"] for iter_num in iterations]
    }
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)

def track_same_prompts_across_iterations(iteration_results, num_samples=10, seed=42):
    """
    Track the same prompts across iterations by finding prompts in iter0 that were incorrectly answered
    and tracking their responses in all subsequent iterations.
    
    Returns a dictionary mapping prompt keys to a list of responses from each iteration.
    """
    iterations = sorted(iteration_results.keys())
    if not iterations or 0 not in iterations:
        print("Warning: Iteration 0 not found, cannot track prompts")
        return {}
    
    # Get incorrect examples from iter0
    iter0_incorrect = [ex for ex in iteration_results[0]["examples"] if not ex["is_correct"]]
    
    # If there are fewer incorrect examples than requested, take all of them
    if len(iter0_incorrect) <= num_samples:
        tracked_examples = iter0_incorrect
        print(f"Only found {len(iter0_incorrect)} incorrect examples in iteration 0, tracking all of them")
    else:
        # Randomly sample the specified number of incorrect examples
        random.seed(seed)
        tracked_examples = random.sample(iter0_incorrect, num_samples)
        print(f"Randomly sampled {num_samples} incorrect examples from iteration 0 to track")
    
    # Create a dictionary of prompt keys to track
    prompt_keys_to_track = {get_prompt_key(ex["prompt"]): ex["prompt"] for ex in tracked_examples}
    
    # Initialize the tracking dictionary
    tracked_prompts = {
        prompt_key: {
            "prompt": prompt,
            "iterations": {}
        } for prompt_key, prompt in prompt_keys_to_track.items()
    }
    
    # Track these prompts across all iterations
    for iter_num in iterations:
        for ex in iteration_results[iter_num]["examples"]:
            prompt_key = get_prompt_key(ex["prompt"])
            if prompt_key in prompt_keys_to_track:
                tracked_data = {
                    "response": ex["response"],
                    "is_correct": ex["is_correct"],
                    "ground_truth": ex["ground_truth"],
                    "token_count": ex.get("token_count", None)
                }
                
                # Add new fields if available
                if "score" in ex:
                    tracked_data["score"] = ex["score"]
                    
                if "data_source" in ex:
                    tracked_data["data_source"] = ex["data_source"]
                
                tracked_prompts[prompt_key]["iterations"][iter_num] = tracked_data
    
    return tracked_prompts

def format_tracked_examples_for_human(tracked_prompts):
    """Format tracked examples in a human-readable way"""
    result = ""
    
    for prompt_key, data in tracked_prompts.items():
        result += "=" * 50 + "\n"
        
        # Handle prompt which could be a dictionary or a string
        if isinstance(data["prompt"], dict):
            result += "PROMPT:\n" + json.dumps(data["prompt"], indent=2, cls=NumpyEncoder) + "\n\n"
        else:
            result += "PROMPT:\n" + str(data["prompt"]) + "\n\n"
        
        # Sort iterations
        iterations = sorted(data["iterations"].keys())
        
        for iter_num in iterations:
            iter_data = data["iterations"][iter_num]
            result += f"--- ITERATION {iter_num} ---\n"
            
            # Add data source if available
            if "data_source" in iter_data:
                result += f"DATA SOURCE: {iter_data['data_source']}\n"
                
            result += "RESPONSE:\n" + iter_data["response"] + "\n\n"
            
            # Extract predicted answer
            response_extract = re.search(r'<answer>(.*?)</answer>', iter_data["response"], re.DOTALL)
            predicted = response_extract.group(1).strip() if response_extract else "No valid answer"
            
            result += f"PREDICTED: {predicted}\n"
            
            # Handle ground_truth which could be a dictionary
            if isinstance(iter_data["ground_truth"], dict):
                label = iter_data["ground_truth"].get('label', 'Unknown')
                result += f"CORRECT ANSWER: {label}\n"
            else:
                result += f"CORRECT ANSWER: {iter_data['ground_truth']}\n"
            
            result += f"IS CORRECT: {iter_data['is_correct']}\n"
            
            # Add score if available
            if "score" in iter_data:
                result += f"SCORE: {iter_data['score']}\n"
            
            if iter_data["token_count"] is not None:
                result += f"TOKEN COUNT: {iter_data['token_count']}\n"
            
            result += "\n"
        
        result += "=" * 50 + "\n\n"
    
    return result

def format_tracked_examples_by_iteration(tracked_prompts):
    """Format tracked examples organized by iteration"""
    result = ""
    
    # Get all iterations from all prompts
    all_iterations = set()
    for prompt_data in tracked_prompts.values():
        all_iterations.update(prompt_data["iterations"].keys())
    
    # Sort iterations
    sorted_iterations = sorted(all_iterations)
    
    for iter_num in sorted_iterations:
        result += "=" * 50 + "\n"
        result += f"ITERATION {iter_num}\n"
        result += "=" * 50 + "\n\n"
        
        # Count for example numbering
        example_count = 1
        
        # Process each prompt for this iteration
        for prompt_key, data in tracked_prompts.items():
            if iter_num in data["iterations"]:
                iter_data = data["iterations"][iter_num]
                
                result += f"Example {example_count}:\n"
                result += "-" * 40 + "\n"
                
                # Handle prompt which could be a dictionary or a string
                if isinstance(data["prompt"], dict):
                    result += "PROMPT:\n" + json.dumps(data["prompt"], indent=2, cls=NumpyEncoder) + "\n\n"
                else:
                    result += "PROMPT:\n" + str(data["prompt"]) + "\n\n"
                
                # Add data source if available
                if "data_source" in iter_data:
                    result += f"DATA SOURCE: {iter_data['data_source']}\n\n"
                
                result += "RESPONSE:\n" + iter_data["response"] + "\n\n"
                
                # Extract predicted answer
                response_extract = re.search(r'<answer>(.*?)</answer>', iter_data["response"], re.DOTALL)
                predicted = response_extract.group(1).strip() if response_extract else "No valid answer"
                
                result += f"PREDICTED: {predicted}\n"
                
                # Handle ground_truth which could be a dictionary
                if isinstance(iter_data["ground_truth"], dict):
                    label = iter_data["ground_truth"].get('label', 'Unknown')
                    result += f"CORRECT ANSWER: {label}\n"
                else:
                    result += f"CORRECT ANSWER: {iter_data['ground_truth']}\n"
                
                result += f"IS CORRECT: {iter_data['is_correct']}\n"
                
                # Add score if available
                if "score" in iter_data:
                    result += f"SCORE: {iter_data['score']}\n"
                
                if iter_data["token_count"] is not None:
                    result += f"TOKEN COUNT: {iter_data['token_count']}\n"
                
                result += "\n"
                example_count += 1
        
        result += "\n\n"
    
    return result

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
    
    # Prepare data for token length analysis
    token_stats_by_iter = {}
    accuracy_by_iter = {}
    
    # Track the same prompts across iterations
    tracked_prompts = track_same_prompts_across_iterations(iteration_results, num_samples=10)
    
    # Create directory for tracking the same prompts
    same_prompt_dir = os.path.join(full_output_dir, "same_prompt")
    os.makedirs(same_prompt_dir, exist_ok=True)
    
    # Save tracked prompts as JSON
    with open(os.path.join(same_prompt_dir, "tracked_prompts.json"), "w") as f:
        json.dump(tracked_prompts, f, indent=2, cls=NumpyEncoder)
    
    # Save tracked prompts as human-readable text (organized by prompt)
    with open(os.path.join(same_prompt_dir, "tracked_prompts_by_prompt.txt"), "w") as f:
        f.write(format_tracked_examples_for_human(tracked_prompts))
    
    # Save tracked prompts as human-readable text (organized by iteration)
    with open(os.path.join(same_prompt_dir, "tracked_prompts_by_iteration.txt"), "w") as f:
        f.write(format_tracked_examples_by_iteration(tracked_prompts))
    
    # Process results for each iteration
    for iter_num in iterations:
        iter_data = iteration_results[iter_num]
        accuracy = iter_data["accuracy"]
        examples = iter_data["examples"]
        
        # Add to plot data
        iter_numbers.append(iter_num)
        accuracies.append(accuracy)
        accuracy_by_iter[iter_num] = accuracy
        
        # Create directory for this iteration
        iter_dir = os.path.join(full_output_dir, f"iter{iter_num}")
        os.makedirs(iter_dir, exist_ok=True)
        
        # Save all examples with correctness info (in a structured format)
        with open(os.path.join(iter_dir, "all_examples.json"), "w") as f:
            json.dump(examples, f, indent=2, cls=NumpyEncoder)
        
        # Sample some correct and incorrect examples
        sampled_correct, sampled_incorrect, correct_explanation, incorrect_explanation = sample_examples(
            examples, n_correct=100, n_incorrect=3
        )
        
        # Save sampled examples in human-readable format (text files)
        with open(os.path.join(iter_dir, "sample_correct_examples.txt"), "w") as f:
            f.write(correct_explanation)
            for ex in sampled_correct:
                score = ex.get("score", None)
                data_source = ex.get("data_source", None)
                f.write(format_example_for_human(
                    ex["prompt"], ex["response"], ex["is_correct"], ex["ground_truth"], 
                    score, data_source
                ))
        
        with open(os.path.join(iter_dir, "sample_incorrect_examples.txt"), "w") as f:
            f.write(incorrect_explanation)
            for ex in sampled_incorrect:
                score = ex.get("score", None)
                data_source = ex.get("data_source", None)
                f.write(format_example_for_human(
                    ex["prompt"], ex["response"], ex["is_correct"], ex["ground_truth"],
                    score, data_source
                ))
        
        # Save sampled examples as JSON files
        with open(os.path.join(iter_dir, "sample_correct_examples.json"), "w") as f:
            json.dump(sampled_correct, f, indent=2, cls=NumpyEncoder)
        
        with open(os.path.join(iter_dir, "sample_incorrect_examples.json"), "w") as f:
            json.dump(sampled_incorrect, f, indent=2, cls=NumpyEncoder)
            
        # Analyze token lengths for this iteration
        token_stats = analyze_token_lengths(examples)
        token_stats_by_iter[iter_num] = token_stats
        
        # Save token length statistics
        with open(os.path.join(iter_dir, "token_stats.json"), "w") as f:
            # Remove the 'counts' field which can be large
            stats_to_save = {k: v for k, v in token_stats.items() if k != 'counts'}
            json.dump(stats_to_save, f, indent=2, cls=NumpyEncoder)
            
        # Plot token distribution for this iteration
        plot_token_distribution(token_stats, os.path.join(iter_dir, "token_distribution.pdf"))
    
    # Generate and save accuracy plot
    plot_accuracies(iter_numbers, accuracies, os.path.join(full_output_dir, "accuracy_plot.pdf"))
    
    # Generate and save token stats across iterations plot
    plot_token_stats_across_iterations(iter_numbers, token_stats_by_iter, 
                                     os.path.join(full_output_dir, "token_stats_plot.pdf"))
    
    # Generate and save token boxplot comparison
    plot_token_boxplot(iter_numbers, token_stats_by_iter, 
                     os.path.join(full_output_dir, "token_boxplot.pdf"))
    
    # Save token stats to CSV for easier analysis
    save_token_stats_csv(iter_numbers, token_stats_by_iter, accuracy_by_iter,
                        os.path.join(full_output_dir, "token_stats_summary.csv"))
    
    # Save all token stats
    with open(os.path.join(full_output_dir, "all_token_stats.json"), "w") as f:
        # Remove the 'counts' field which can be large
        stats_to_save = {f"iter{iter_num}": {k: v for k, v in stats.items() if k != 'counts'} 
                         for iter_num, stats in token_stats_by_iter.items()}
        json.dump(stats_to_save, f, indent=2, cls=NumpyEncoder)
    
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
    
    print("\nToken Length Statistics:")
    for iter_num in iterations:
        stats = token_stats_by_iter[iter_num]
        print(f"  Iteration {iter_num}: median={stats['median']:.1f}, mean={stats['mean']:.1f}, min={stats['min']}, max={stats['max']}")
    
    print("\nOverlap with Previous Iteration:")
    for key, value in overlaps_with_prev.items():
        print(f"  {key}: {value:.4f}")
    
    print("\nOverlap with First Iteration:")
    for key, value in overlaps_with_first.items():
        print(f"  {key}: {value:.4f}")
    
    print(f"\nPlot saved to: {os.path.join(full_output_dir, 'accuracy_plot.pdf')}")
    print(f"Token stats plot saved to: {os.path.join(full_output_dir, 'token_stats_plot.pdf')}")
    print(f"Token distribution boxplot saved to: {os.path.join(full_output_dir, 'token_boxplot.pdf')}")
    print(f"Token stats summary CSV saved to: {os.path.join(full_output_dir, 'token_stats_summary.csv')}")
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
            
            try:
                accuracy, examples = analyze_file(file_path)
                
                correct_examples = sum(1 for ex in examples if ex["is_correct"])
                print(f"Iteration {iter_num} accuracy: {accuracy:.4f}, correct examples: {correct_examples}")
                
                iteration_results[iter_num] = {
                    "accuracy": accuracy,
                    "examples": examples,
                    "file_path": file_path
                }
            except Exception as e:
                print(f"ERROR analyzing file {os.path.basename(file_path)}: {e}")
                continue
    
    if not iteration_results:
        print("No valid iteration data found. Check file patterns and naming.")
        sys.exit(1)
    
    # Write results
    write_results(output_dir, project_name, iteration_results)
    print(f"\nResults written to {os.path.join(output_dir, project_name)}")

if __name__ == "__main__":
    main()
