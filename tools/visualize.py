#!/usr/bin/env python3
import pandas as pd
import json
import os
import argparse
import re
import html
from pathlib import Path
import random
import csv # Added for append_metrics_to_csv
from typing import Dict, Any, List, Optional, Tuple, Union # Added Union

# Assuming constants.py defines supported_datasets correctly
from constants import supported_datasets
# Assuming helper functions are correctly imported
from verl.trainer.ppo.helper import _select_rm_score_fn as select_reward_fn
from examples.data_preprocess.helper import _select_parse_fn as select_parse_fn # Renamed for clarity

try:
    from transformers import AutoTokenizer
    from sklearn.metrics import r2_score # Moved import here
except ImportError as e:
    print(f"Warning: Could not import required libraries (transformers/sklearn): {e}")
    AutoTokenizer = None
    r2_score = None
import numpy as np

# Default threshold (can be overridden by args)
DEFAULT_REGRESSION_CORRECT_THRESHOLD = -0.01

# --- Tokenizer Functions ---
def get_tokenizer(tokenizer_name="google/gemma-2-9b-it"): # Updated default tokenizer
    """Gets tokenizer, falling back to GPT-2 if needed."""
    if AutoTokenizer is None:
        print("Transformers library not available. Tokenizer cannot be loaded.")
        return None
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        return tokenizer
    except Exception as e:
        print(f"Warning: Failed to load tokenizer {tokenizer_name}: {e}")
        try:
            print("Falling back to GPT-2 tokenizer")
            return AutoTokenizer.from_pretrained("gpt2")
        except Exception as e2:
            print(f"Failed to load GPT-2 tokenizer: {e2}. Token length will not be calculated.")
            return None

def calculate_token_length(text: Optional[str], tokenizer) -> int:
    """Calculates token length of text using the provided tokenizer."""
    if tokenizer is None or text is None:
        return 0
    try:
        # Ensure text is string
        if not isinstance(text, str):
             if isinstance(text, bytes): text = text.decode('utf-8', errors='ignore')
             else: text = str(text)
        tokens = tokenizer.encode(text)
        return len(tokens)
    except Exception as e:
        print(f"Error calculating token length for text type {type(text)}: {e}")
        return 0

# --- Content Extraction Functions ---
def _ensure_string(text: Any) -> Optional[str]:
    """Helper to safely convert input to string."""
    if text is None:
        return None
    if isinstance(text, str):
        return text
    try:
        if isinstance(text, bytes):
            return text.decode('utf-8', errors='replace')
        else:
            return str(text)
    except Exception as e:
        print(f"Warning: Could not convert content to string: {e}")
        return str(text) # Last resort

def extract_think_content(text: Any) -> Optional[str]:
    """Extracts content between <think> and </think> tags."""
    text_str = _ensure_string(text)
    if text_str is None:
        return None
    match = re.search(r'<think>(.*?)</think>', text_str, re.DOTALL)
    return match.group(1).strip() if match else None

def extract_answer_content(text: Any) -> Optional[str]:
    """Extracts content between <answer> and </answer> tags."""
    text_str = _ensure_string(text)
    if text_str is None:
        return None
    match = re.search(r'<answer>(.*?)</answer>', text_str, re.DOTALL)
    return match.group(1).strip() if match else None

def clean_response_text(response_text: Any) -> str:
    """Cleans response text by ensuring it's a string and removing end tokens."""
    text_str = _ensure_string(response_text)
    if text_str is None:
        return ""
    # Simple cleaning, add more if needed
    return text_str.replace("<|endoftext|>", "").strip()

def get_experiment_name(exp_path: str) -> str:
    """
    Gets a formatted experiment name (model_name_dataset) from the file path.
    Expects path structure like: .../model_name/dataset_details_.../subdir/filename.parquet
    Example: results/google-gemini-2.0-flash/moons_50_shot_.../global_step_0/test_default.parquet
    Desired output: google-gemini-2.0-flash_moons

    Args:
        exp_path: The full path to the experiment file (e.g., parquet file).

    Returns:
        The formatted experiment name string, or a fallback name if parsing fails.
    """
    try:
        path_obj = Path(exp_path)

        # Check if the path has enough parts to extract the desired components.
        # We need at least file, subdir, dataset_details, model_name.
        # path_obj.parts includes the filename, path_obj.parent doesn't.
        # path_obj.parent.parts gives directory parts.
        # Example breakdown:
        # parts = ('results', 'google-gemini-2.0-flash', 'moons_50_shot_...', 'global_step_0', 'test_default.parquet')
        # parent.parts = ('results', 'google-gemini-2.0-flash', 'moons_50_shot_...', 'global_step_0')
        # We need at least 3 directory levels above the file.
        if len(path_obj.parent.parts) < 3:
             print(f"Warning: Path '{exp_path}' has fewer than 3 parent directories. Cannot extract model/dataset as expected.")
             return path_obj.stem # Fallback to filename without extension

        # Model Name is the 3rd directory from the end (parent of parent of parent)
        model_name = path_obj.parent.parent.parent.name

        # Dataset Details is the 2nd directory from the end (parent of parent)
        dataset_details = path_obj.parent.parent.name

        # Dataset name is the part before the first underscore in dataset_details
        dataset_name = dataset_details.split('_')[0]

        return f"{model_name}_{dataset_name}"

    except IndexError:
        # This might happen if split('_') fails on a directory name without underscores
        print(f"Warning: Could not parse dataset from directory name '{dataset_details}' in path '{exp_path}'.")
        return path_obj.stem # Fallback
    except Exception as e:
        print(f"Warning: An error occurred parsing path '{exp_path}': {e}")
        return path_obj.stem # Fallback


# --- CSV Metrics Logging ---
def append_metrics_to_csv(csv_path: str, experiment_path: str, metrics: Dict[str, Any]):
    """Appends experiment metrics to a CSV file."""
    experiment_name = get_experiment_name(experiment_path)
    file_exists = os.path.isfile(csv_path)

    # Flatten metrics if nested (optional, adjust as needed)
    row = {"experiment": experiment_name}
    for k, v in metrics.items():
        row[k] = f"{v:.4f}" if isinstance(v, float) else v # Format floats

    mode = 'a' if file_exists else 'w'
    try:
        with open(csv_path, mode, newline='', encoding='utf-8') as csvfile:
            # Define fieldnames based on current row keys, ensuring 'experiment' is first
            fieldnames = ['experiment'] + [k for k in row if k != 'experiment']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()

            # Filter row to only include existing fieldnames (handles adding new columns)
            filtered_row = {k: row.get(k, '') for k in fieldnames}
            writer.writerow(filtered_row)
    except IOError as e:
        print(f"Error writing to CSV {csv_path}: {e}")
    except Exception as e:
         print(f"An unexpected error occurred writing to CSV: {e}")

# --- NumPy Data Handling Helper ---
def safe_tolist(data: Any) -> Any:
     """Converts NumPy arrays/scalars within nested structures to Python lists/scalars."""
     if isinstance(data, np.ndarray):
         return data.tolist()
     elif isinstance(data, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8,
                         np.uint16, np.uint32, np.uint64)):
         return int(data)
     elif isinstance(data, (np.float_, np.float16, np.float32, np.float64)):
         return float(data)
     elif isinstance(data, np.bool_):
         return bool(data)
     elif isinstance(data, (list, tuple)):
         return [safe_tolist(item) for item in data]
     elif isinstance(data, dict):
         return {key: safe_tolist(value) for key, value in data.items()}
     return data # Return as is if not a NumPy type or container we handle

def load_data_and_tokenizer(input_file: str, tokenizer_name: str = "google/gemma-2-9b-it") -> Tuple[pd.DataFrame, Any]:
    """Loads data from parquet file and initializes tokenizer."""
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Error: File '{input_file}' not found")

    print(f"Reading parquet file: {input_file}")
    try:
        df = pd.read_parquet(input_file)
        print(f"DataFrame loaded successfully with {len(df)} rows")
        print(f"DataFrame columns: {df.columns.tolist()}")
        if 'responses' not in df.columns:
             raise ValueError("'responses' column not found in DataFrame")
    except Exception as e:
        raise ValueError(f"Error reading parquet file {input_file}: {e}") from e

    tokenizer = get_tokenizer(tokenizer_name)
    return df, tokenizer

def calculate_metrics(
    df: pd.DataFrame,
    regression_threshold: float
) -> Tuple[Dict[str, Any], List[bool], List[Any], List[Any]]:
    """Calculates overall metrics from the full dataframe."""
    print(f"Calculating metrics on all {len(df)} samples...")
    total_data_size = len(df)
    if total_data_size == 0:
        return {}, [], [], []

    # Per-row results
    metrics_per_row = []
    parsed_predictions_per_row = []
    ground_truth_labels_per_row = []
    correctness_flags_per_row = []

    # R2 calculation lists
    predictions_for_r2 = []
    ground_truths_for_r2 = []
    parseable_predictions_for_r2 = []
    parseable_ground_truths_for_r2 = []

    # Aggregate counts
    full_correct_count = 0
    parseable_count = 0
    parseable_correct_count = 0
    mse_sum = 0.0
    parseable_mse_sum = 0.0
    wrong_number_of_answers = 0
    task_type = "unknown" # Determine from first valid row

    for _, row in df.iterrows():
        # --- Extract data for the row ---
        data_source = row.get('data_source', None)
        if not data_source:
            print(f"Warning: Missing 'data_source' in row. Skipping metric calculation for this row.")
            metrics_per_row.append(None) # Placeholder
            parsed_predictions_per_row.append(None)
            ground_truth_labels_per_row.append(None)
            correctness_flags_per_row.append(False)
            continue

        if data_source not in supported_datasets:
             print(f"Warning: Unknown data_source '{data_source}'. Skipping.")
             continue

        current_task_type = supported_datasets[data_source]['type']
        if task_type == "unknown":
            task_type = current_task_type # Set task type based on first row

        ground_truth_dict = row.get('reward_model', {}).get('ground_truth', None)
        ground_truth_label = ground_truth_dict.get('label', None) if isinstance(ground_truth_dict, dict) else None

        responses = row.get('responses', [])
        # Ensure responses is a list and get the first response
        if not isinstance(responses, list): responses = [responses]
        response_text = clean_response_text(responses[0]) if responses else ""

        # --- Select functions ---
        try:
            reward_fn = select_reward_fn(data_source)
            parse_fn = select_parse_fn(data_source)
        except NotImplementedError:
            print(f"Warning: Reward or parse function not implemented for {data_source}. Skipping metrics.")
            continue
        except Exception as e:
             print(f"Error selecting functions for {data_source}: {e}. Skipping.")
             continue


        # --- Evaluate ---
        metric_value = 0.0
        is_correct = False
        parsed_prediction = None
        is_parseable = False

        if ground_truth_label is not None:
            try:
                # Calculate metric (reward)
                metric_value = reward_fn(response_text, ground_truth_dict)

                # Attempt to parse prediction
                parsed_prediction = parse_fn(response_text)

                # Check parseability and length match
                if isinstance(parsed_prediction, (list, np.ndarray)) and \
                   isinstance(ground_truth_label, (list, np.ndarray)) and \
                   len(parsed_prediction) == len(ground_truth_label) and \
                   parsed_prediction != []: # Check if parse_fn returned non-empty list for success
                    is_parseable = True
                elif not isinstance(parsed_prediction, (list, np.ndarray)) and \
                     not isinstance(ground_truth_label, (list, np.ndarray)) and \
                     parsed_prediction is not None and \
                   len(parsed_prediction) == len(ground_truth_label): # Handle scalar case
                    is_parseable = True
                elif isinstance(parsed_prediction, (list, np.ndarray)) and parsed_prediction == []:
                     # Explicitly treat empty list as unparseable if parse_fn indicates failure this way
                     is_parseable = False
                     wrong_number_of_answers += 1
                elif parsed_prediction is not None: # Parsed something, but length mismatch
                    is_parseable = False # Treat length mismatch as unparseable for stats
                    wrong_number_of_answers += 1
                else: # parse_fn returned None
                     is_parseable = False

                # Determine correctness based on task type and metric/threshold
                if task_type == "classification":
                    is_correct = (metric_value == 1.0)
                    mse_sum += (1.0 - metric_value) # Treat accuracy as 1 - error rate for consistency
                    if is_parseable: parseable_mse_sum += (1.0 - metric_value)
                elif task_type == "regression":
                    # metric_value is likely negative MSE from reward_fn
                    is_correct = (metric_value >= regression_threshold)
                    mse_sum += (-metric_value) # Accumulate positive MSE
                    if is_parseable: parseable_mse_sum += (-metric_value)
                else:
                    is_correct = bool(metric_value) # Generic fallback
                    # Cannot reliably calculate MSE/Accuracy for unknown tasks

            except Exception as e:
                print(f"Error processing row with data_source {data_source}: {e}")
                # Keep defaults (False, None, False)

            # --- Update counts and lists ---
            if is_parseable:
                parseable_count += 1
                if is_correct:
                    full_correct_count += 1
                    parseable_correct_count += 1
                # R2 lists (only if parseable and regression)
                if task_type == "regression":
                    parseable_predictions_for_r2.append(safe_tolist(parsed_prediction))
                    parseable_ground_truths_for_r2.append(safe_tolist(ground_truth_label))
                    predictions_for_r2.append(safe_tolist(parsed_prediction))
                    ground_truths_for_r2.append(safe_tolist(ground_truth_label))
            else:
                 if is_correct: # Still count correct if metric says so, even if unparseable
                      full_correct_count += 1
                 # Add random prediction for R2 calculation if unparseable regression
                 if task_type == "regression":
                      num_labels = len(safe_tolist(ground_truth_label))
                      random_pred = [random.uniform(0, 10) for _ in range(num_labels)]
                      if num_labels == 1: random_pred = random_pred[0] # Scalar if needed
                      predictions_for_r2.append(random_pred)
                      ground_truths_for_r2.append(safe_tolist(ground_truth_label))

        # Store per-row results
        metrics_per_row.append(metric_value)
        parsed_predictions_per_row.append(safe_tolist(parsed_prediction)) # Ensure basic types
        ground_truth_labels_per_row.append(safe_tolist(ground_truth_label))
        correctness_flags_per_row.append(is_correct)

    # --- Calculate final aggregate metrics ---
    accuracy = (full_correct_count / total_data_size) if total_data_size > 0 else 0
    parseable_accuracy = (parseable_correct_count / parseable_count) if parseable_count > 0 else 0
    parseable_proportion = (parseable_count / total_data_size) if total_data_size > 0 else 0
    unparseable_count = total_data_size - parseable_count

    metrics_dict = {
        "total_samples": total_data_size,
        "parseable_samples": parseable_count,
        "unparseable_samples": unparseable_count,
        "parseable_proportion": parseable_proportion * 100,
        "wrong_number_of_answers": wrong_number_of_answers,
    }

    if task_type == "classification":
        refined_accuracy = (mse_sum / total_data_size) # This is actually avg error rate
        parseable_refined_accuracy = (parseable_mse_sum / parseable_count) if parseable_count > 0 else 0
        metrics_dict.update({
            "overall_accuracy": accuracy * 100,
            "accuracy_per_point": (1.0 - refined_accuracy) * 100, # Convert error rate back to accuracy
            "parseable_accuracy": parseable_accuracy * 100,
            "parseable_accuracy_per_point": (1.0 - parseable_refined_accuracy) * 100,
        })
        print(f"Overall Accuracy: {metrics_dict['overall_accuracy']:.2f}%")
        print(f"Parseable Accuracy: {metrics_dict['parseable_accuracy']:.2f}% ({parseable_count}/{total_data_size})")

    elif task_type == "regression":
        overall_mse = (mse_sum / total_data_size) if total_data_size > 0 else 0
        parseable_mse = (parseable_mse_sum / parseable_count) if parseable_count > 0 else 0
        r2 = r2_score(ground_truths_for_r2, predictions_for_r2) if r2_score and ground_truths_for_r2 else 0.0
        parseable_r2 = r2_score(parseable_ground_truths_for_r2, parseable_predictions_for_r2) if r2_score and parseable_ground_truths_for_r2 else 0.0

        metrics_dict.update({
            "overall_accuracy_threshold": accuracy * 100, # Based on threshold
            "overall_mse": overall_mse,
            "parseable_accuracy_threshold": parseable_accuracy * 100, # Based on threshold
            "parseable_mse": parseable_mse,
            "r2_score": r2,
            "parseable_r2_score": parseable_r2,
        })
        print(f"Overall MSE: {metrics_dict['overall_mse']:.4f}")
        print(f"Parseable MSE: {metrics_dict['parseable_mse']:.4f} ({parseable_count}/{total_data_size})")
        print(f"Overall R2 Score: {metrics_dict['r2_score']:.4f}")
        print(f"Parseable R2 Score: {metrics_dict['parseable_r2_score']:.4f}")

    print(f"Unparseable Predictions: {unparseable_count} ({unparseable_count/total_data_size*100:.2f}%)")
    print(f"Wrong Number of Answers (Length Mismatch): {wrong_number_of_answers} ({wrong_number_of_answers/total_data_size*100:.2f}%)")

    return metrics_dict, correctness_flags_per_row, parsed_predictions_per_row, ground_truth_labels_per_row


def sample_data_for_visualization(
    df: pd.DataFrame,
    max_samples: int,
    correctness_flags: List[bool]
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Samples data for visualization, aiming for a balanced correct/incorrect split."""
    total_data_size = len(df)
    sampling_info = {"sampled": False, "target_correct": 0, "target_incorrect": 0}

    if max_samples <= 0 or max_samples >= total_data_size:
        return df, sampling_info # Return full dataframe if no sampling needed

    print("Sampling data for visualization...")
    correct_indices = [i for i, correct in enumerate(correctness_flags) if correct]
    incorrect_indices = [i for i, correct in enumerate(correctness_flags) if not correct]

    correct_df = df.iloc[correct_indices]
    incorrect_df = df.iloc[incorrect_indices]
    print(f"Found {len(correct_df)} correct samples and {len(incorrect_df)} incorrect samples")

    # Calculate how many samples to take from each group
    total_correct = len(correct_df)
    total_incorrect = len(incorrect_df)

    # Aim for 50/50 split if possible
    target_correct = min(total_correct, max_samples // 2)
    target_incorrect = min(total_incorrect, max_samples - target_correct)

    # Adjust if one group is too small
    if target_correct < max_samples // 2:
        target_incorrect = min(total_incorrect, max_samples - target_correct)
    if target_incorrect < max_samples // 2: # Check again after potential adjustment
        target_correct = min(total_correct, max_samples - target_incorrect)

    # Sample from each group
    sampled_correct = correct_df.sample(n=target_correct, random_state=42) if target_correct > 0 else pd.DataFrame(columns=df.columns)
    sampled_incorrect = incorrect_df.sample(n=target_incorrect, random_state=43) if target_incorrect > 0 else pd.DataFrame(columns=df.columns)

    # Combine and shuffle
    display_df = pd.concat([sampled_correct, sampled_incorrect])
    display_df = display_df.sample(frac=1, random_state=44).reset_index(drop=True) # Shuffle and reset index

    sampling_info = {"sampled": True, "target_correct": target_correct, "target_incorrect": target_incorrect}
    print(f"Sampled {target_correct} correct samples and {target_incorrect} incorrect samples for visualization.")
    print(f"Total samples for visualization: {len(display_df)}")

    return display_df, sampling_info


def format_features_for_display(features: Any) -> str:
     """Formats features (handling lists, arrays, scalars) for HTML/text display."""
     if isinstance(features, (list, tuple, np.ndarray)):
         # Check if it's a list/array of lists/arrays (e.g., multiple feature sets for n_query > 1)
         if features and isinstance(features[0], (list, tuple, np.ndarray)):
              formatted = []
              for i, feat_set in enumerate(features):
                   try:
                       f_str = ", ".join([f"{x:.3f}" for x in feat_set])
                       formatted.append(f"{i+1}. [{f_str}]")
                   except (TypeError, ValueError):
                       formatted.append(f"{i+1}. {str(feat_set)}")
              return "<br>".join(formatted) # Use <br> for HTML multiline
         else: # Single feature set
              try:
                   f_str = ", ".join([f"{x:.3f}" for x in features])
                   return f"[{f_str}]"
              except (TypeError, ValueError):
                  return str(features)
     else: # Scalar or other type
         return str(features)


def generate_html_report(
    output_file: str,
    display_df: pd.DataFrame,
    metrics_dict: Dict[str, Any],
    sampling_info: Dict[str, Any],
    tokenizer: Any,
    input_file_name: str,
    regression_threshold: float
):
    """Generates the HTML report content and writes it to a file."""
    print(f"Generating HTML report: {output_file}")
    html_content = [
        "<!DOCTYPE html><html><head>",
        "<meta charset=\"UTF-8\"><meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">",
    ]

    # --- Determine Title ---
    task_type = supported_datasets.get(display_df.iloc[0].get('data_source', ''), {}).get('type', 'unknown') if not display_df.empty else 'unknown'
    title = "ICL Reasoning Results"
    if task_type == "classification":
        acc = metrics_dict.get('overall_accuracy', 0)
        title += f" - Accuracy: {acc:.2f}%"
    elif task_type == "regression":
         mse = metrics_dict.get('overall_mse', 0)
         title += f" - MSE: {mse:.4f}"

    html_content.extend([
         f"<title>{title}</title>",
         "<style>",
         "body { font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }",
         ".sample { border: 1px solid #ddd; padding: 15px; margin-bottom: 20px; border-radius: 5px; }",
         ".section { margin-bottom: 15px; }",
         ".section-title { font-weight: bold; background-color: #f5f5f5; padding: 5px; margin-bottom: 5px;}",
         ".prompt { white-space: pre-wrap; font-family: monospace; max-height: 300px; overflow-y: auto; background-color: #f8f8f8; padding: 5px; border: 1px solid #eee;}",
         ".response { white-space: pre-wrap; font-family: monospace; }",
         ".think { background-color: #f9f9f9; padding: 10px; border-left: 3px solid #ccc; white-space: pre-wrap; font-family: monospace;}",
         ".answer { font-weight: bold; }",
         ".correct { color: green; }",
         ".incorrect { color: red; }",
         ".summary { background-color: #eef; padding: 15px; margin-bottom: 20px; border-radius: 5px; }",
         "table { border-collapse: collapse; width: 100%; margin-bottom: 10px;}",
         "th, td { border: 1px solid #ddd; padding: 8px; text-align: left; vertical-align: top; }",
         "th { background-color: #f2f2f2; }",
         ".metrics-table td:nth-child(2) { text-align: right; font-weight: bold; }",
         "details { margin-bottom: 10px; border: 1px solid #eee; padding: 5px; border-radius: 4px;}",
         "summary { cursor: pointer; font-weight: bold; }",
         ".token-count { color:#666; font-size:0.9em; margin-left: 10px; }",
         "</style></head><body>",
         f"<h1>ICL Reasoning Results: {Path(input_file_name).name}</h1>",
    ])

    # --- Summary Section ---
    html_content.append('<div class="summary"><h2>Results Summary</h2><table class="metrics-table">')
    html_content.append(f"<tr><td>Total Samples</td><td>{metrics_dict['total_samples']}</td></tr>")
    if sampling_info["sampled"]:
        html_content.append(f"<tr><td>Displayed Samples</td><td>{len(display_df)} (Balanced: {sampling_info['target_correct']} correct, {sampling_info['target_incorrect']} incorrect)</td></tr>")

    if task_type == "classification":
        html_content.append(f"<tr><td>Overall Accuracy</td><td>{metrics_dict.get('overall_accuracy', 0):.2f}%</td></tr>")
        html_content.append(f"<tr><td>Accuracy Per Point</td><td>{metrics_dict.get('accuracy_per_point', 0):.2f}%</td></tr>")
        html_content.append(f"<tr><td>Parseable Accuracy</td><td>{metrics_dict.get('parseable_accuracy', 0):.2f}%</td></tr>")
        html_content.append(f"<tr><td>Parseable Accuracy Per Point</td><td>{metrics_dict.get('parseable_accuracy_per_point', 0):.2f}%</td></tr>")
    elif task_type == "regression":
        html_content.append(f"<tr><td>Overall MSE</td><td>{metrics_dict.get('overall_mse', 0):.4f}</td></tr>")
        html_content.append(f"<tr><td>Parseable MSE</td><td>{metrics_dict.get('parseable_mse', 0):.4f}</td></tr>")
        html_content.append(f"<tr><td>Overall R2 Score</td><td>{metrics_dict.get('r2_score', 0):.4f}</td></tr>")
        html_content.append(f"<tr><td>Parseable R2 Score</td><td>{metrics_dict.get('parseable_r2_score', 0):.4f}</td></tr>")
        html_content.append(f"<tr><td>Correct Threshold (≥)</td><td>{regression_threshold}</td></tr>")
        html_content.append(f"<tr><td>Overall Accuracy (Threshold)</td><td>{metrics_dict.get('overall_accuracy_threshold', 0):.2f}%</td></tr>")
        html_content.append(f"<tr><td>Parseable Accuracy (Threshold)</td><td>{metrics_dict.get('parseable_accuracy_threshold', 0):.2f}%</td></tr>")

    html_content.append(f"<tr><td>Parseable Proportion</td><td>{metrics_dict.get('parseable_proportion', 0):.2f}% ({metrics_dict.get('parseable_samples', 0)}/{metrics_dict.get('total_samples', 0)})</td></tr>")
    html_content.append(f"<tr><td>Unparseable Samples</td><td>{metrics_dict.get('unparseable_samples', 0)}</td></tr>")
    html_content.append(f"<tr><td>Wrong # Answers</td><td>{metrics_dict.get('wrong_number_of_answers', 0)}</td></tr>")
    html_content.append("</table></div>")

    # --- Individual Samples ---
    for idx, row in display_df.iterrows():
        # --- Extract data ---
        # Use safe_tolist to handle potential numpy types from sampling/concat
        row_dict = safe_tolist(row.to_dict())

        data_source = row_dict.get('data_source', 'unknown')
        prompt_data = row_dict.get('prompt', [{}])[0] # Get first prompt dict
        datasample_text = prompt_data.get('datasample')
        question_text = prompt_data.get('content')
        responses = row_dict.get('responses', [])
        rules = row_dict.get('rules', [])

        if not isinstance(rules, list): rules = [rules]
        rules_text = clean_response_text(rules[0]) if rules else None
        if not isinstance(responses, list): responses = [responses]
        response_text = clean_response_text(responses[0]) if responses else "" # Use cleaned text

        reward_model_data = row_dict.get('reward_model', {})
        ground_truth_data = reward_model_data.get('ground_truth', {})
        gt_label = ground_truth_data.get('label', 'N/A')
        gt_features = ground_truth_data.get('features', 'N/A')

        extra_info = row_dict.get('extra_info', {})
        # --- Recalculate metric/prediction for this sample (needed for display) ---
        metric = 0.0
        is_correct = False
        prediction = None
        try:
             current_task_type = supported_datasets[data_source]['type']
             reward_fn = select_reward_fn(data_source)
             parse_fn = select_parse_fn(data_source)
             metric = reward_fn(response_text, ground_truth_data) # Use original cleaned response
             prediction = parse_fn(response_text)
             # Determine correctness again for display consistency
             if current_task_type == "classification":
                 is_correct = (metric == 1.0)
             elif current_task_type == "regression":
                 is_correct = (metric >= regression_threshold)

             # Handle cases where parse_fn returns None or []
             if prediction is None or (isinstance(prediction, list) and prediction == []):
                  prediction = "UNPARSEABLE"
                  is_correct = False # Mark unparseable as incorrect for display consistency
             elif isinstance(prediction, list) and isinstance(gt_label, list) and len(prediction) != len(gt_label):
                  prediction = f"PARSE ERROR (Length mismatch: {prediction})"
                  is_correct = False # Mark length mismatch as incorrect

        except Exception as e:
             print(f"Error recalculating metrics for display (idx {idx}): {e}")
             prediction = "ERROR"
             is_correct = False

        result_class = "correct" if is_correct else "incorrect"

        # --- Build HTML for sample ---
        html_content.append(f'<div class="sample">')
        html_content.append(f'<h2>Sample {extra_info.get("index", idx)+1} <span class="{result_class}" style="font-size: small;">({result_class.upper()})</span></h2>')

        # Data source and Extra Info
        html_content.append('<div class="section"><details><summary>Details & Config</summary>')
        html_content.append(f'<div><b>Data Source:</b> {data_source}</div>')
        html_content.append('<b>Extra Info:</b><table>')
        for k, v in extra_info.items():
            html_content.append(f'<tr><td>{html.escape(str(k))}</td><td>{html.escape(str(v))}</td></tr>')
        html_content.append('</table></details></div>')

        # Prompt (Datasample + Question)
        html_content.append('<div class="section"><div class="section-title">Input Prompt</div><details><summary>Show Prompt</summary><div class="prompt">')
        prompt_token_length = 0
        if datasample_text:
            ds_len = calculate_token_length(datasample_text, tokenizer)
            prompt_token_length += ds_len
            html_content.append(f'<b>[Datasample]</b><span class="token-count">[{ds_len} tokens]</span><br>{html.escape(datasample_text)}<hr>')
        if question_text:
            q_len = calculate_token_length(question_text, tokenizer)
            prompt_token_length += q_len
            html_content.append(f'<b>[Question]</b><span class="token-count">[{q_len} tokens]</span><br>{html.escape(question_text)}')
        html_content.append(f'</div><div><b>Total Prompt Tokens: {prompt_token_length}</b></div></details></div>')

        # Ground Truth
        html_content.append('<div class="section"><div class="section-title">Ground Truth</div>')
        html_content.append(f'<div><b>Label:</b> {html.escape(str(gt_label))}</div>')
        html_content.append(f'<div><b>Features:</b> {format_features_for_display(gt_features)}</div>') # Use helper
        html_content.append('</div>')

        # Prediction Result
        html_content.append('<div class="section"><div class="section-title">Prediction Result</div>')
        html_content.append(f'<div class="{result_class}"><b>Predicted:</b> {html.escape(str(prediction))}</div>')
        if task_type == "regression":
             html_content.append(f'<div class="{result_class}"><b>Metric (-MSE):</b> {metric:.4f}</div>')
        html_content.append('</div>')

                # Model Response (cleaned)
        if rules_text:
            html_content.append('<div class="section"><details open><summary>Rule</summary>')
            response_token_length = calculate_token_length(rules_text, tokenizer)
            html_content.append(f'<div class="response">{html.escape(rules_text)}</div>')
            html_content.append(f'<div style="text-align: right;"><span class="token-count">[{response_token_length} tokens]</span></div>')
            html_content.append('</details></div>')

        # Model Response (cleaned)
        html_content.append('<div class="section"><details open><summary>Model Response (Cleaned)</summary>')
        response_token_length = calculate_token_length(response_text, tokenizer)
        html_content.append(f'<div class="response">{html.escape(response_text)}</div>')
        html_content.append(f'<div style="text-align: right;"><span class="token-count">[{response_token_length} tokens]</span></div>')
        html_content.append('</details></div>')

        # Optional: Thinking
        thinking_content = extract_think_content(response_text)
        if thinking_content:
            html_content.append('<div class="section"><details><summary>Extracted Thinking</summary>')
            think_token_length = calculate_token_length(thinking_content, tokenizer)
            html_content.append(f'<div class="think">{html.escape(thinking_content)}</div>')
            html_content.append(f'<div style="text-align: right;"><span class="token-count">[{think_token_length} tokens]</span></div>')
            html_content.append('</details></div>')

        # Optional: Answer
        answer_content = extract_answer_content(response_text)
        if answer_content:
            html_content.append('<div class="section"><details><summary>Extracted Answer Tag</summary>')
            ans_token_length = calculate_token_length(answer_content, tokenizer)
            html_content.append(f'<div class="answer">{html.escape(answer_content)}</div>')
            html_content.append(f'<div style="text-align: right;"><span class="token-count">[{ans_token_length} tokens]</span></div>')
            html_content.append('</details></div>')

        html_content.append('</div>') # End sample div

    # --- Finalize HTML ---
    html_content.append("</body></html>")
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(html_content))
    except IOError as e:
         print(f"Error writing HTML file {output_file}: {e}")


def generate_txt_report(
    output_file: str,
    display_df: pd.DataFrame,
    metrics_dict: Dict[str, Any],
    sampling_info: Dict[str, Any],
    tokenizer: Any,
    input_file_name: str,
    regression_threshold: float
):
    """Generates the TXT report content and writes it to a file."""
    print(f"Generating TXT report: {output_file}")
    txt_content = []

    # --- Header ---
    txt_content.append("="*80)
    txt_content.append(f"ICL REASONING RESULTS: {Path(input_file_name).name}")
    txt_content.append("="*80 + "\n")

    # --- Summary Section ---
    txt_content.append("SUMMARY")
    txt_content.append("-"*80)
    txt_content.append(f"Total Samples: {metrics_dict['total_samples']}")
    if sampling_info["sampled"]:
         txt_content.append(f"Displayed Samples: {len(display_df)} (Balanced: {sampling_info['target_correct']} correct, {sampling_info['target_incorrect']} incorrect)")

    task_type = supported_datasets.get(display_df.iloc[0].get('data_source', ''), {}).get('type', 'unknown') if not display_df.empty else 'unknown'
    if task_type == "classification":
        txt_content.append(f"Overall Accuracy: {metrics_dict.get('overall_accuracy', 0):.2f}%")
        # txt_content.append(f"Accuracy Per Point: {metrics_dict.get('accuracy_per_point', 0):.2f}%")
        txt_content.append(f"Parseable Accuracy: {metrics_dict.get('parseable_accuracy', 0):.2f}%")
        # txt_content.append(f"Parseable Accuracy Per Point: {metrics_dict.get('parseable_accuracy_per_point', 0):.2f}%")
    elif task_type == "regression":
        txt_content.append(f"Overall MSE: {metrics_dict.get('overall_mse', 0):.4f}")
        txt_content.append(f"Parseable MSE: {metrics_dict.get('parseable_mse', 0):.4f}")
        txt_content.append(f"Overall R2 Score: {metrics_dict.get('r2_score', 0):.4f}")
        txt_content.append(f"Parseable R2 Score: {metrics_dict.get('parseable_r2_score', 0):.4f}")
        txt_content.append(f"Correct Threshold (>=): {regression_threshold}")
        txt_content.append(f"Overall Accuracy (Threshold): {metrics_dict.get('overall_accuracy_threshold', 0):.2f}%")
        txt_content.append(f"Parseable Accuracy (Threshold): {metrics_dict.get('parseable_accuracy_threshold', 0):.2f}%")

    txt_content.append(f"Parseable Proportion: {metrics_dict.get('parseable_proportion', 0):.2f}% ({metrics_dict.get('parseable_samples', 0)}/{metrics_dict.get('total_samples', 0)})")
    txt_content.append(f"Unparseable Samples: {metrics_dict.get('unparseable_samples', 0)}")
    txt_content.append(f"Wrong # Answers: {metrics_dict.get('wrong_number_of_answers', 0)}")
    txt_content.append("="*80 + "\n")

    # --- Individual Samples ---
    for idx, row in display_df.iterrows():
        # --- Extract data ---
        row_dict = safe_tolist(row.to_dict())
        data_source = row_dict.get('data_source', 'unknown')
        prompt_data = row_dict.get('prompt', [{}])[0]
        datasample_text = prompt_data.get('datasample')
        question_text = prompt_data.get('content')
        responses = row_dict.get('responses', [])
        if not isinstance(responses, list): responses = [responses]
        response_text = clean_response_text(responses[0]) if responses else ""

        reward_model_data = row_dict.get('reward_model', {})
        ground_truth_data = reward_model_data.get('ground_truth', {})
        gt_label = ground_truth_data.get('label', 'N/A')
        gt_features = ground_truth_data.get('features', 'N/A')
        extra_info = row_dict.get('extra_info', {})

        # --- Recalculate metric/prediction ---
        metric = 0.0
        is_correct = False
        prediction = None
        try:
             current_task_type = supported_datasets[data_source]['type']
             reward_fn = select_reward_fn(data_source)
             parse_fn = select_parse_fn(data_source)
             metric = reward_fn(response_text, ground_truth_data)
             prediction = parse_fn(response_text)
             if current_task_type == "classification": is_correct = (metric == 1.0)
             elif current_task_type == "regression": is_correct = (metric >= regression_threshold)
             if prediction is None or (isinstance(prediction, list) and prediction == []):
                 prediction = "UNPARSEABLE"
                 is_correct = False
             elif isinstance(prediction, list) and isinstance(gt_label, list) and len(prediction) != len(gt_label):
                  prediction = f"PARSE ERROR (Length mismatch: {prediction})"
                  is_correct = False
        except Exception as e:
             prediction = "ERROR"
             is_correct = False

        result_str = "CORRECT" if is_correct else "INCORRECT"

        # --- Build TXT for sample ---
        txt_content.append(f"=== Sample {extra_info.get('index', idx)+1} ({result_str}) ===")
        txt_content.append(f"Data Source: {data_source}")

        # Extra Info
        txt_content.append("\n--- Extra Info ---")
        for k, v in extra_info.items():
            txt_content.append(f"  {k}: {v}")

        # Prompt
        txt_content.append("\n--- Input Prompt ---")
        prompt_token_length = 0
        if datasample_text:
            ds_len = calculate_token_length(datasample_text, tokenizer)
            prompt_token_length += ds_len
            txt_content.append(f"[Datasample] [{ds_len} tokens]\n{datasample_text}\n{'-'*20}")
        if question_text:
            q_len = calculate_token_length(question_text, tokenizer)
            prompt_token_length += q_len
            txt_content.append(f"[Question] [{q_len} tokens]\n{question_text}")
        txt_content.append(f"--> Total Prompt Tokens: {prompt_token_length}")


        # Ground Truth
        txt_content.append("\n--- Ground Truth ---")
        txt_content.append(f"Label: {gt_label}")
        # Use format_features_for_display but replace <br> with newline
        features_display_text = format_features_for_display(gt_features).replace("<br>", "\n")
        txt_content.append(f"Features: {features_display_text}")

        # Prediction
        txt_content.append("\n--- Prediction Result ---")
        txt_content.append(f"Predicted: {prediction} ({result_str})")
        if task_type == "regression":
             txt_content.append(f"Metric (-MSE): {metric:.4f}")

        # Response
        response_token_length = calculate_token_length(response_text, tokenizer)
        txt_content.append(f"\n--- Model Response (Cleaned) [{response_token_length} tokens] ---")
        txt_content.append(response_text)

        # Optional: Thinking
        thinking_content = extract_think_content(response_text)
        if thinking_content:
            think_token_length = calculate_token_length(thinking_content, tokenizer)
            txt_content.append(f"\n--- Extracted Thinking [{think_token_length} tokens] ---")
            txt_content.append(thinking_content)

        # Optional: Answer
        answer_content = extract_answer_content(response_text)
        if answer_content:
             ans_token_length = calculate_token_length(answer_content, tokenizer)
             txt_content.append(f"\n--- Extracted Answer Tag [{ans_token_length} tokens] ---")
             txt_content.append(answer_content)

        txt_content.append("\n" + "="*80 + "\n")

    # --- Write to file ---
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(txt_content))
    except IOError as e:
         print(f"Error writing TXT file {output_file}: {e}")

def visualize_icl_reasoning_output_refactored(
    input_file: str,
    output_format: str = "html",
    save_dir: Optional[str] = None,
    max_samples: int = 100,
    regression_threshold: float = DEFAULT_REGRESSION_CORRECT_THRESHOLD,
    output_csv_dir: Optional[str] = None
) -> str:
    """
    Orchestrates the loading, analysis, sampling, and visualization of ICL output.

    Args:
        input_file: Path to the input parquet file.
        output_format: 'txt' or 'html'.
        save_dir: Directory to save the output file.
        max_samples: Max samples for visualization (0 for all).
        regression_threshold: Threshold for regression correctness.
        output_csv_dir: Directory to save metrics CSV.

    Returns:
        Path to the generated output file.
    """
    random.seed(42) # Set seed for reproducibility in sampling

    # 1. Load data and tokenizer
    df, tokenizer = load_data_and_tokenizer(input_file)

    # 2. Calculate metrics on the full dataset
    # Note: Pass necessary arguments if calculate_metrics needs them (e.g., selectors)
    metrics_dict, correctness_flags, _, _ = calculate_metrics(
        df, regression_threshold
    )
    print(f"Metrics calculated: {metrics_dict}")

    # 3. Append metrics to CSV if requested
    if output_csv_dir:
        csv_filename = "regression_metrics.csv" if metrics_dict.get("overall_mse") is not None else "classification_metrics.csv"
        output_csv_path = os.path.join(output_csv_dir, csv_filename)
        Path(output_csv_dir).mkdir(parents=True, exist_ok=True)
        append_metrics_to_csv(output_csv_path, input_file, metrics_dict)
        print(f"Metrics appended to: {output_csv_path}")

    # 4. Sample data for visualization
    display_df, sampling_info = sample_data_for_visualization(
        df, max_samples, correctness_flags
    )

    # 5. Determine output file path
    if save_dir:
        output_dir = Path(save_dir)
    else:
        output_dir = Path(input_file).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    output_filename = f"{Path(input_file).stem}_visualization.{output_format}"
    output_file = output_dir / output_filename

    # 6. Generate the report
    if output_format == "html":
        generate_html_report(
            str(output_file), display_df, metrics_dict, sampling_info, tokenizer,
            input_file, regression_threshold
        )
    elif output_format == "txt":
        generate_txt_report(
            str(output_file), display_df, metrics_dict, sampling_info, tokenizer,
            input_file, regression_threshold
        )
    else:
        raise ValueError(f"Unsupported output format: {output_format}")

    print(f"Visualization complete! Saved to: {output_file}")
    return str(output_file)

def main():
    parser = argparse.ArgumentParser(description='Visualize ICL reasoning output with model responses')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to the input parquet file with model responses')
    parser.add_argument('--format', type=str, choices=['txt', 'html'], default='html',
                        help='Output format (txt or html)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save the output file (default: same as input file)')
    parser.add_argument('--max-samples', type=int, default=100,
                        help='Maximum number of samples to visualize (default: 100, use 0 for all)')
    # Argument names should match the function parameters now
    parser.add_argument('--regression-threshold', type=float, default=DEFAULT_REGRESSION_CORRECT_THRESHOLD,
                        help=f'Threshold for regression correctness (default: {DEFAULT_REGRESSION_CORRECT_THRESHOLD})')
    parser.add_argument('--output-csv-dir', type=str, default=None, # Changed arg name slightly
                        help='Directory to save/append metrics to a CSV file')
    args = parser.parse_args()

    try:
        output_file = visualize_icl_reasoning_output_refactored(
            input_file=args.input,
            output_format=args.format,
            save_dir=args.output_dir,
            max_samples=args.max_samples,
            regression_threshold=args.regression_threshold, # Use updated arg name
            output_csv_dir=args.output_csv_dir # Use updated arg name
        )
        # print(f"Refactored visualization complete! Saved to: {output_file}") # Optional
    except FileNotFoundError as e:
         print(f"Error: {e}")
    except ValueError as e:
         print(f"Error: {e}")
    except Exception as e:
         print(f"An unexpected error occurred: {e}")
         import traceback
         traceback.print_exc() # Print full traceback for unexpected errors


if __name__ == "__main__":
    main()