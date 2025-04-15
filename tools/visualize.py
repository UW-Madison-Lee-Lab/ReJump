#!/usr/bin/env python3
"""
Visualize ICL reasoning output from parquet files with model responses
Default input file: /staging/szhang967/icl_dataset-output/blobs_50shot_n1.0_f0.0_test10_icl3_seed42.parquet
"""

import pandas as pd
import json
import os
import argparse
import re
import html
from pathlib import Path
import random
from typing import Dict, Any, List, Optional, Tuple
from constants import supported_datasets
import pdb
from verl.trainer.ppo.helper import _select_rm_score_fn as select_reward_fn
from examples.data_preprocess.helper import _select_parse_fn
try:
    from transformers import AutoTokenizer
except ImportError:
    print("Warning: Could not import AutoTokenizer from transformers")
    AutoTokenizer = None

# Initialize tokenizer
def get_tokenizer(tokenizer_name="Qwen/Qwen2.5-3B-Instruct"):
    """
    Get tokenizer for token length calculation
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        return tokenizer
    except Exception as e:
        print(f"Warning: Failed to load tokenizer {tokenizer_name}: {e}")
        try:
            print("Falling back to GPT-2 tokenizer")
            return AutoTokenizer.from_pretrained("gpt2")
        except:
            print("Failed to load any tokenizer. Token length will not be calculated.")
            return None

def calculate_token_length(text, tokenizer):
    """
    Calculate token length of text
    """
    if tokenizer is None or text is None:
        return 0
    try:
        tokens = tokenizer.encode(text)
        return len(tokens)
    except Exception as e:
        print(f"Error calculating token length: {e}")
        return 0

def extract_think_content(text) -> Optional[str]:
    """
    Extract content between <think> and </think> tags
    """
    # Handle None case
    if text is None:
        return None
        
    # Force convert to string if it's bytes or any other type
    if not isinstance(text, str):
        try:
            if isinstance(text, bytes):
                text = text.decode('utf-8')
            else:
                text = str(text)
        except Exception as e:
            print(f"Warning: Could not convert to string: {e}")
            text = str(text)  # Last resort: use str() representation
        
    pattern = r'<think>(.*?)</think>'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def clean_response_text(response_text: str, prompt_content: Optional[str]) -> str:
    """
    Return original response text without any cleaning
    
    Args:
        response_text: Model response text
        prompt_content: Input prompt content (not used)
        
    Returns:
        Original response text
    """
    # Simply filter out <|endoftext|> tags
    if response_text:
        return response_text.replace("<|endoftext|>", "")
    return response_text


def extract_answer_content(text) -> Optional[str]:
    """
    Extract content between <answer> and </answer> tags
    """
    # Handle None case
    if text is None:
        return None
        
    # Force convert to string if it's bytes or any other type
    if not isinstance(text, str):
        try:
            if isinstance(text, bytes):
                text = text.decode('utf-8')
            else:
                text = str(text)
        except Exception as e:
            print(f"Warning: Could not convert to string: {e}")
            text = str(text)  # Last resort: use str() representation
        
    pattern = r'<answer>(.*?)</answer>'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def get_prediction_result(response_text: Optional[str], ground_truth, task_type: str) -> Tuple[Optional[int], bool]:
    #should be the same with reward_fn in helper.py, but I am too lazy to copy it over
    """
    Evaluate prediction using main_eval approach
    
    Args:
        response_text: Full model response text
        ground_truth: The ground truth data containing label and features
        
    Returns:
        Tuple of (predicted_label, metric)
    """
    # Import classification_reward_fn from helper module
    if task_type == "classification":
        from examples.data_preprocess.helper import classification_reward_fn
        # Use the classification_reward_fn for evaluation
        metric = classification_reward_fn(response_text, ground_truth)
        
        nested_match = re.search(r'<answer><answer>(.*?)</answer></answer>', response_text, re.DOTALL)
        if nested_match:
            prediction_str = nested_match.group(1).strip()
            if ',' in prediction_str:
                try:
                    prediction = [int(x.strip()) for x in prediction_str.split(',')]
                except ValueError:
                    try:
                        prediction = [float(x.strip()) for x in prediction_str.split(',')]
                    except ValueError:
                        prediction = None
                if prediction is not None:
                    return prediction, metric
            else:
                if re.match(r'^-?\d+(\.\d+)?$', prediction_str):
                    if '.' in prediction_str:
                        prediction = float(prediction_str)
                    else:
                        prediction = int(prediction_str)
                    return prediction, metric
                
        # Extract the predicted label for display
        all_matches = list(re.finditer(r'<answer>(.*?)</answer>', response_text, re.DOTALL))
        if all_matches:
            response_extract = None
            for match in all_matches[::-1]:  # Check from last to first
                match_content = match.group(1).strip()

                # Match comma-separated integers or floats
                if ',' in match_content:
                    try:
                        prediction = [int(x.strip()) for x in match_content.split(',')]
                        return prediction, metric
                    except ValueError:
                        try:
                            prediction = [float(x.strip()) for x in match_content.split(',')]
                            return prediction, metric
                        except ValueError:
                            continue

                # Match integers and floats (including negative numbers)
                if re.match(r'^-?\d+(\.\d+)?$', match_content):
                    response_extract = match
                    break
            if response_extract is not None and re.match(r'^-?\d+(\.\d+)?$', response_extract.group(1).strip()):
                prediction_str = response_extract.group(1).strip()
                # Convert to int if it's an integer, otherwise float
                if '.' in prediction_str:
                    prediction = float(prediction_str)
                else:
                    prediction = int(prediction_str)
                return prediction, metric
            
            # Try direct pattern matching if the tags might have whitespace issues
            num_pattern = r'<answer>\s*(-?\d+(\.\d+)?)\s*</answer>'
            num_matches = re.findall(num_pattern, response_text)
            if num_matches:
                prediction_str = num_matches[-1][0]  # Use the last match
                # Convert to int if it's an integer, otherwise float
                if '.' in prediction_str:
                    prediction = float(prediction_str)
                else:
                    prediction = int(prediction_str)
                return prediction, metric
        
        # If metric but didn't find valid prediction, try more aggressive patterns
        if metric:
            # Look for numbers after "answer:" or "class:" patterns that might appear in text
            alternative_patterns = [
                r'answer:\s*(-?\d+(\.\d+)?)',
                r'class:\s*(-?\d+(\.\d+)?)',
                r'prediction:\s*(-?\d+(\.\d+)?)',
                r'label:\s*(-?\d+(\.\d+)?)',
                r'the answer is\s*(-?\d+(\.\d+)?)',
                r'class is\s*(-?\d+(\.\d+)?)'
            ]
            
            for pattern in alternative_patterns:
                matches = re.findall(pattern, response_text, re.IGNORECASE)
                if matches:
                    try:
                        prediction_str = matches[-1][0]  # Use the last match
                        # Convert to int if it's an integer, otherwise float
                        if '.' in prediction_str:
                            prediction = float(prediction_str)
                        else:
                            prediction = int(prediction_str)
                        return prediction, metric
                    except (ValueError, TypeError):
                        continue  # Try next pattern if this one didn't work
            
            # Final fallback: just find any number that could be a valid class
            if ground_truth and 'label' in ground_truth:
                # Get task type to determine number of classes
                num_classes = supported_datasets[ground_truth['data_source']]['num_classes']
                
                # Extract all number sequences and check if any could be a valid class
                num_matches = re.findall(r'\b(-?\d+(\.\d+)?)\b', response_text)
                for match in num_matches[::-1]:  # Check from last to first
                    try:
                        # Convert to int if it's an integer, otherwise float
                        if '.' in match[0]:
                            num_val = float(match[0])
                        else:
                            num_val = int(match[0])
                        
                        # For integer values, check if they're valid class indices
                        if isinstance(num_val, int) and 0 <= num_val < num_classes:
                            # Found a valid class index
                            return num_val, metric
                    except (ValueError, TypeError):
                        continue
                
            # If we know the answer is correct but couldn't extract it, use ground truth
            if ground_truth and 'label' in ground_truth:
                return ground_truth['label'], True
        
        # Otherwise, couldn't extract prediction
        return None, False
    else:
        # Fallback to original implementation if module not available
        if response_text is None:
            return None, False
        
        # Make sure ground_truth_label is available
        if ground_truth is None or 'label' not in ground_truth:
            return None, False
            
        ground_truth_label = ground_truth['label']
        
        # Try to parse a number from the answer
        answer = extract_answer_content(response_text)
        if answer is not None and re.match(r'^-?\d+(\.\d+)?$', answer.strip()):
            prediction_str = answer.strip()
            # Convert to int if it's an integer, otherwise float
            if '.' in prediction_str:
                prediction = float(prediction_str)
            else:
                prediction = int(prediction_str)
            return prediction, prediction == ground_truth_label
        
        # Look for number patterns in the full response
        num_pattern = r'<answer>\s*(-?\d+(\.\d+)?)\s*</answer>'
        num_matches = re.findall(num_pattern, response_text)
        if num_matches:
            prediction_str = num_matches[-1][0]  # Use the last match
            # Convert to int if it's an integer, otherwise float
            if '.' in prediction_str:
                prediction = float(prediction_str)
            else:
                prediction = int(prediction_str)
            return prediction, prediction == ground_truth_label
            
        # Try additional patterns for classification responses
        alternative_patterns = [
            r'answer:\s*(-?\d+(\.\d+)?)',
            r'class:\s*(-?\d+(\.\d+)?)',
            r'prediction:\s*(-?\d+(\.\d+)?)',
            r'label:\s*(-?\d+(\.\d+)?)',
            r'the answer is\s*(-?\d+(\.\d+)?)',
            r'class is\s*(-?\d+(\.\d+)?)'
        ]
        
        for pattern in alternative_patterns:
            matches = re.findall(pattern, response_text, re.IGNORECASE)
            if matches:
                try:
                    prediction_str = matches[-1][0]  # Use the last match
                    # Convert to int if it's an integer, otherwise float
                    if '.' in prediction_str:
                        prediction = float(prediction_str)
                    else:
                        prediction = int(prediction_str)
                    return prediction, prediction == ground_truth_label
                except (ValueError, TypeError):
                    continue  # Try next pattern if this one didn't work
        
        # Final fallback: just find any number that could be a valid class
        num_classes = supported_datasets[ground_truth['data_source']]['num_classes']
            
        # Extract all number sequences and check if any could be a valid class
        num_matches = re.findall(r'\b(-?\d+(\.\d+)?)\b', response_text)
        for match in num_matches[::-1]:  # Check from last to first
            try:
                # Convert to int if it's an integer, otherwise float
                if '.' in match[0]:
                    num_val = float(match[0])
                else:
                    num_val = int(match[0])
                
                # For integer values, check if they're valid class indices
                if isinstance(num_val, int) and 0 <= num_val < num_classes:
                    # Found a valid class index
                    return num_val, num_val == ground_truth_label
            except (ValueError, TypeError):
                continue
        
        # If no number found, return None
        return None, False
    
def get_experiment_name(exp_path):
    """
    Get experiment name from the path  
    """
    parts = exp_path.split('/')
    if len(parts) >= 3:
        first = parts[1]
        second = parts[2].split('_')[0]
        return f"{first}_{second}"
    return "Unknown_Experiment"

def visualize_icl_reasoning_output(input_file: str, output_format: str = "txt", save_dir: Optional[str] = None, max_samples: int = 100, REGRESSION_CORRECT_THRESHOLD=-0.01, output_csv_dir: Optional[str] = None) -> str:
    """
    Visualize ICL reasoning output from parquet files with model responses
    
    Args:
        input_file: Path to the input parquet file
        output_format: Output format, supports txt and html
        save_dir: Directory to save the output file (default: same as input file's directory)
        max_samples: Maximum number of samples to visualize (default: 100)
        
    Returns:
        Path to the output file
    """
    # Set random seed for refined accuracy calculation
    random.seed(42)
    
    # Initialize tokenizer
    tokenizer = get_tokenizer()
    
    # Read the parquet file
    print(f"Reading parquet file: {input_file}")

    df = pd.read_parquet(input_file)
    
    # Debug information
    print(f"DataFrame loaded successfully with {len(df)} rows")
    print(f"DataFrame columns: {df.columns.tolist()}")
    
    # Check if 'responses' column exists
    if 'responses' in df.columns:
        print("'responses' column found in DataFrame")
        # Check the type of the first response
        if len(df) > 0:
            first_response = df.iloc[0].get('responses')
            print(f"Type of first response: {type(first_response)}")
            if isinstance(first_response, list) and len(first_response) > 0:
                print(f"Type of first response item: {type(first_response[0])}")
        
    else:
        raise ValueError("'responses' column not found in DataFrame")
    
    
    
        
    # Calculate accuracy on full dataset first
    total_data_size = len(df)
    wrong_number_of_answer=0
    full_correct_predictions = 0
    full_MSE_accuracyPsamples = 0  # For storing the correct count for refined accuracy
    parseable_correct_predictions = 0
    parseable_MSE_accuracyPsamples= 0  # For storing the refined accuracy
    unparseable_predictions = 0  # Count of unparseable predictions
    parseable_correct = 0  # Count of correct predictions among parseable ones
    parseable_predictions = 0  # List to store parseable predictions
    #caculate R2
    predictions_list = []
    ground_truths_list = []
    parseable_predictions_list = []
    parseable_ground_truths_list = []

    print(f"Calculating accuracy on all {total_data_size} samples...")
    
    # Process all samples for accuracy calculation
    for _, row in df.iterrows():
        # Get ground truth label
        ground_truth = None
        data_source = row.get('data_source', 'blobs')
        task_type = supported_datasets[data_source]['type']
        if 'reward_model' in row and isinstance(row['reward_model'], dict):
            ground_truth_data = row['reward_model'].get('ground_truth', None)
            if isinstance(ground_truth_data, dict) and 'label' in ground_truth_data:
                ground_truth = ground_truth_data
        
        # Process responses
        responses = row.get('responses', [])
        
        # Handle different types of response data
        if responses is None:
            responses = ["No response available"]
        elif not isinstance(responses, list):
            # Try to convert to list if it's another iterable
            try:
                responses = list(responses)
            except (TypeError, ValueError):
                responses = [responses]
        
        # Get first response (assuming single response per example)
        response_text = responses[0] if responses and len(responses) > 0 else "No response generated"
        
        # Ensure response_text is a string
        if not isinstance(response_text, str):
            try:
                if isinstance(response_text, bytes):
                    response_text = response_text.decode('utf-8')
                else:
                    response_text = str(response_text)
            except Exception as e:
                print(f"Warning: Could not convert response to string: {e}")
                response_text = str(response_text)  # Last resort
        
        # Filter out <|endoftext|> tags
        response_text = response_text.replace("<|endoftext|>", "")
        
        # Get input prompt content
        input_prompt = row.get('prompt')
        input_prompt_content = input_prompt[0]['content']
        
        # Use original response without attempting to remove prompt
        cleaned_response_text = clean_response_text(response_text, input_prompt_content)
        
        # Remove prompt content from response
        if input_prompt_content:
            cleaned_response_text = cleaned_response_text.replace(input_prompt_content, "")
        
        # Preserve original content, don't extract thinking and answer
        raw_thinking = extract_think_content(cleaned_response_text)
        raw_answer = extract_answer_content(cleaned_response_text)
        
        # Calculate token length
        token_length = calculate_token_length(cleaned_response_text, tokenizer)
        
        # Get reward function for the data source
        reward_fn = select_reward_fn(data_source)
        extract_answer=_select_parse_fn(data_source)
        # Evaluate with the appropriate reward function
        metric = False
        can_parse_prediction = True
        
        # Use the dedicated reward function
        metric = reward_fn(cleaned_response_text, ground_truth)


        # Try to extract prediction result to check if it can be parsed
        parsed_prediction = extract_answer(cleaned_response_text)

        

        if parsed_prediction != [] and len(parsed_prediction) == len(ground_truth['label']):
            parseable_correct += 1  
            parseable_predictions += 1
            

            if task_type == "classification":
                # For classification, check if the prediction matches the ground truth
                if metric == 1.0:
                    full_correct_predictions += 1
                    parseable_correct_predictions += 1
                full_MSE_accuracyPsamples += metric
                parseable_MSE_accuracyPsamples += metric
                
            if task_type == "regression":
                # For regression, we can use the metric directly
                if metric >= REGRESSION_CORRECT_THRESHOLD:
                    full_correct_predictions += 1
                    parseable_correct_predictions += 1  
                
                full_MSE_accuracyPsamples += metric   
                parseable_MSE_accuracyPsamples += metric

                #Calculate R2
                if ground_truth and 'label' in ground_truth:
                    ground_truth_label = ground_truth['label']
                    parseable_predictions_list.append(parsed_prediction)
                    parseable_ground_truths_list.append(ground_truth_label)
                    predictions_list.append(parsed_prediction)
                    ground_truths_list.append(ground_truth_label)

  
        else: #a random prediction
            unparseable_predictions += 1
            if task_type == "classification":
                # For classification, check if the prediction matches the ground truth
                if metric == 1.0:
                    full_correct_predictions += 1
                full_MSE_accuracyPsamples += metric
            if task_type == "regression":
                # For regression, we can use the metric directly
                if metric >= REGRESSION_CORRECT_THRESHOLD:
                    full_correct_predictions += 1
                full_MSE_accuracyPsamples += metric 

                answers = [random.uniform(0, 10) for _ in range(len(ground_truth['label']))]
                predictions_list.append(answers)
                ground_truths_list.append(ground_truth['label'])

    
    # Calculate overall accuracy and refined accuracy
    accuracy = (full_correct_predictions / total_data_size) if total_data_size > 0 else 0
    refined_accuracy = (full_MSE_accuracyPsamples / total_data_size) if total_data_size > 0 else 0
    
    # Calculate accuracy for parseable predictions only
    parseable_accuracy = (parseable_correct_predictions / parseable_correct) if parseable_correct > 0 else 0
    parseable_refined_accuracy= (parseable_MSE_accuracyPsamples / parseable_correct) if parseable_correct > 0 else 0
    parseable_porporation = (parseable_correct / total_data_size) if total_data_size > 0 else 0
    if task_type == "regression":
        # Calculate R2 score
        from sklearn.metrics import r2_score
        r2 = r2_score(ground_truths_list, predictions_list)
        r2_parseable = r2_score(parseable_ground_truths_list, parseable_predictions_list) if parseable_predictions_list != [] else 0
    
    if task_type == "classification":
        metrics_dict = {
            "overall_accuracy": accuracy * 100,
            "accuracy_per_point": refined_accuracy * 100,
            "parseable_overall_accuracy": parseable_accuracy * 100,
            "parseable_accuracy_per_point": parseable_refined_accuracy * 100,
            "parseableporporation": parseable_porporation * 100
        }
    else:
        # 对回归任务：
        metrics_dict = {
            "overall_accuracy": accuracy * 100,  # 如果需要百分比，可以乘以 100
            "overall_mse": -refined_accuracy,
            "parseable_accuracy": parseable_accuracy * 100,
            "parseable_mse": -parseable_refined_accuracy,
            "r2_score": r2,
            "parseable_r2_score": r2_parseable,
            "parseableporporation": parseable_porporation * 100
        }
    import csv
    import os
    def append_experiment_metrics(csv_path, experiment_name, metrics):
        experiment_name = get_experiment_name(experiment_name)

        file_exists = os.path.isfile(csv_path)
        row = {"experiment": experiment_name}
        row.update(metrics)
        
        mode = 'a' if file_exists else 'w'
        with open(csv_path, mode, newline='') as csvfile:
            fieldnames = list(row.keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
        
    if task_type == "classification":
        print(f"Overall accuracy from all {total_data_size} samples: {accuracy*100:.2f}%")
        print(f"Accuracy for each point: {refined_accuracy*100:.2f}%")
        print(f"Parseable accuracy (excluding unparseable): {parseable_accuracy*100:.2f}% ({parseable_predictions}/{total_data_size} samples)")
        print(f"Parseable refined accuracy (excluding unparseable): {parseable_refined_accuracy*100:.2f}% ({parseable_predictions}/{total_data_size} samples)")
        if output_csv_dir is not None:
            output_csv = os.path.join(output_csv_dir, "classification_metrics.csv")
            append_experiment_metrics(output_csv, input_file, metrics_dict)
    else:
        print(f"Overall accuracy from all {total_data_size} samples: {accuracy*100:.2f}")
        print(f"Overall MSE: {-refined_accuracy:.4f} ({parseable_predictions}/{total_data_size} samples)")
        print(f"Parseable accuracy (excluding unparseable): {parseable_accuracy*100:.2f}% ({parseable_predictions}/{total_data_size} samples)")
        print(f"Parseable MSE: {-parseable_refined_accuracy:.4f} ({parseable_predictions}/{total_data_size} samples)")
        print(f"R2 Score: {r2:.4f}")
        print(f"Parseable R2 Score: {r2_parseable:.4f}")
        if output_csv_dir is not None:
            output_csv = os.path.join(output_csv_dir, "regression_metrics.csv")
            append_experiment_metrics(output_csv, input_file, metrics_dict)
    print(f"Unparseable predictions: {unparseable_predictions} ({unparseable_predictions/total_data_size*100:.2f}%)")
        
    # Sample the dataframe if needed for visualization, ensuring balanced correct/incorrect samples
    display_df = df
    target_correct = 0
    target_incorrect = 0
    balanced_sampling = False
    
    if max_samples > 0 and max_samples < total_data_size:
        # Create two separate dataframes for correct and incorrect samples
        correct_mask = []
        incorrect_mask = []
        
        print("Separating correct and incorrect samples for balanced sampling...")
        
        # Identify correct and incorrect samples
        for idx, row in df.iterrows():
            # Same logic as above to determine correctness
            ground_truth = None
            if 'reward_model' in row and isinstance(row['reward_model'], dict):
                ground_truth_data = row['reward_model'].get('ground_truth', None)
                if isinstance(ground_truth_data, dict) and 'label' in ground_truth_data:
                    ground_truth = ground_truth_data
            
            responses = row.get('responses', [])
            if responses is None:
                responses = ["No response available"]
            elif not isinstance(responses, list):
                try:
                    responses = list(responses)
                except (TypeError, ValueError):
                    responses = [responses]
            
            response_text = responses[0] if responses and len(responses) > 0 else "No response generated"
            if not isinstance(response_text, str):
                try:
                    if isinstance(response_text, bytes):
                        response_text = response_text.decode('utf-8')
                    else:
                        response_text = str(response_text)
                except Exception:
                    response_text = str(response_text)
            
            response_text = response_text.replace("<|endoftext|>", "")
            
            input_prompt = row.get('prompt')
            input_prompt_content = None
            if input_prompt and isinstance(input_prompt, list) and len(input_prompt) > 0:
                if isinstance(input_prompt[0], dict) and 'content' in input_prompt[0]:
                    input_prompt_content = input_prompt[0]['content']
            
            cleaned_response_text = clean_response_text(response_text, input_prompt_content)
            if input_prompt_content:
                cleaned_response_text = cleaned_response_text.replace(input_prompt_content, "")
            
            data_source = row.get('data_source', 'blobs')
            reward_fn = select_reward_fn(data_source)
            
            metric = False
            try:
                metric = reward_fn(cleaned_response_text, ground_truth)
            except Exception:
                _, metric = get_prediction_result(cleaned_response_text, ground_truth, task_type)
            
            if metric:
                correct_mask.append(idx)
            else:
                incorrect_mask.append(idx)
        
        correct_df = df.loc[correct_mask]
        incorrect_df = df.loc[incorrect_mask]
        
        print(f"Found {len(correct_df)} correct samples and {len(incorrect_df)} incorrect samples")
        
        # Calculate how many samples to take from each group
        total_correct = len(correct_df)
        total_incorrect = len(incorrect_df)
        
        # Aim for 50/50 split if possible
        target_correct = min(total_correct, max_samples // 2)
        target_incorrect = min(total_incorrect, max_samples - target_correct)
        
        # If one group is smaller than the target, take more from the other group
        if target_correct < max_samples // 2:
            target_incorrect = min(total_incorrect, max_samples - target_correct)
        if target_incorrect < max_samples - target_correct:
            target_correct = min(total_correct, max_samples - target_incorrect)
        
        # Sample from each group
        sampled_correct = correct_df.sample(n=target_correct, random_state=42) if target_correct > 0 else pd.DataFrame()
        sampled_incorrect = incorrect_df.sample(n=target_incorrect, random_state=43) if target_incorrect > 0 else pd.DataFrame()
        
        # Combine the samples
        display_df = pd.concat([sampled_correct, sampled_incorrect])
        
        # Shuffle the combined samples to avoid clustering
        display_df = display_df.sample(frac=1, random_state=44)
        
        balanced_sampling = True
        print(f"Sampled {target_correct} correct samples and {target_incorrect} incorrect samples for visualization")
        print(f"Total samples for visualization: {len(display_df)}")
    
    # Statistics for displayed samples
    displayed_samples = len(display_df)
    displayed_correct_predictions = 0
    
    # Determine the output file path
    if save_dir:
        output_dir = Path(save_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_filename = f"{Path(input_file).stem}_visualization.{output_format}"
        output_file = output_dir / output_filename
    else:
        output_dir = Path(input_file).parent
        output_filename = f"{Path(input_file).stem}_visualization.{output_format}"
        output_file = output_dir / output_filename
    
    print(f"Output file: {output_file}")
    
    if output_format == "html":
        # HTML output
        html_content = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            "    <meta charset=\"UTF-8\">",
            "    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">",
            "    <title>ICL Reasoning Results - Accuracy: " + f"{accuracy*100:.2f}%" + "</title>" if task_type == "classification" else "ICL Reasoning Results - MSE: " + f"{refined_accuracy:.4f}",
            "    <style>",
            "        body { font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }",
            "        .sample { border: 1px solid #ddd; padding: 15px; margin-bottom: 20px; border-radius: 5px; }",
            "        .section { margin-bottom: 15px; }",
            "        .section-title { font-weight: bold; background-color: #f5f5f5; padding: 5px; }",
            "        .prompt { white-space: pre-wrap; font-family: monospace; max-height: 200px; overflow-y: auto; }",
            "        .response { white-space: pre-wrap; font-family: monospace; }",
            "        .think { background-color: #f9f9f9; padding: 10px; border-left: 3px solid #ccc; }",
            "        .answer { font-weight: bold; }",
            "        .correct { color: green; }",
            "        .incorrect { color: red; }",
            "        .summary { background-color: #eef; padding: 15px; margin-bottom: 20px; border-radius: 5px; }",
            "        table { border-collapse: collapse; width: 100%; }",
            "        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
            "        th { background-color: #f2f2f2; }",
            "        /* Simple and clear accuracy style */",
            "        .accuracy-big { font-size: 24px; font-weight: bold; color: #333; margin: 20px 0; padding: 10px; background-color: #e9ffe9; border: 2px solid green; }",
            "        .refined-accuracy { color: #1565C0; }",
            "    </style>",
            "</head>",
            "<body>",
        ]
        
        # Add summary before all samples
        sampled_note = ""
        balance_note = ""
        if max_samples > 0 and max_samples < total_data_size:
            sampled_note = f" (showing {displayed_samples} out of {total_data_size})"
            if balanced_sampling:
                balance_note = f"<div>Balanced sampling: {target_correct} correct, {target_incorrect} incorrect</div>"
        
        # Directly add title and accuracy information at top of page without complex positioning or JavaScript
        top_content = [
            f'<!-- ACCURACY DATA: {accuracy:.2f}% | REFINED: {refined_accuracy:.2f}% | UNPARSEABLE: {unparseable_predictions} -->',
            f'<h1>ICL Reasoning Results: {Path(input_file).name}</h1>',
            f'<div class="accuracy-big">',
            f'Accuracy: {accuracy*100:.2f}% &nbsp;|&nbsp; Refined Accuracy: {refined_accuracy*100:.2f}%' if task_type == "classification" else f'Accuracy: {accuracy:.2f} &nbsp;|&nbsp; Refined MSE: {-refined_accuracy:.4f}',
            f'</div>',
            f'<div class="accuracy-big" style="background-color: #e9f0ff; border-color: #1565C0;">',
            f'Parseable Accuracy: {parseable_accuracy*100:.2f}% (excluding {unparseable_predictions} unparseable samples)' if task_type == "classification" else f'Parseable Accuracy: {-parseable_accuracy:.2f} (excluding {unparseable_predictions} unparseable samples)',
            f'</div>',
            f'<div>Unparseable Predictions: {unparseable_predictions} ({unparseable_predictions/total_data_size*100:.2f}%)</div>',
            f'</div>',
            f'<div>Wrong Number of Answers: {wrong_number_of_answer} ({wrong_number_of_answer/total_data_size*100:.2f}%)</div>',
            f'<div>Correct threshold: {REGRESSION_CORRECT_THRESHOLD} </div>' if task_type == "regression" else "",
            f'<hr style="margin: 20px 0; border: 0; height: 2px; background: #333;">',
        ]
        
        summary_html = [
            f'<div class="summary">',
            f'<h2>Results Summary</h2>',
            f'<table>',
            f'<tr><th>Metric</th><th>Value</th></tr>',
            f'<tr><td>Total Samples{sampled_note}</td><td>{total_data_size}</td></tr>',
            f'<tr><td>Correct Predictions (all data)</td><td>{full_correct_predictions}</td></tr>' if task_type == "classification" else "",
            f'<tr><td>Accuracy (all data)</td><td>{accuracy*100:.2f}%</td></tr>' 
            f'<tr><td>Refined Accuracy</td><td>{refined_accuracy*100:.2f}%</td></tr>' if task_type == "classification" else f'<tr><td>MSE</td><td>{-refined_accuracy:.4f}</td></tr>',
            f'<tr><td>Parseable Accuracy</td><td>{parseable_accuracy*100:.2f}% ({parseable_predictions}/{total_data_size} samples)</td></tr>',
            f'<tr><td>Parseable MSE</td><td>{-parseable_refined_accuracy:.4f} ({parseable_predictions}/{total_data_size} samples)</td></tr>'  if task_type == "regression" else f'<tr><td>Parseable Refined Accuracy</td><td>{parseable_refined_accuracy*100:.2f}% ({parseable_predictions}/{total_data_size} samples)</td></tr>', 
            f'<tr><td>Unparseable Predictions</td><td>{unparseable_predictions} ({unparseable_predictions/total_data_size*100:.2f}%)</td></tr>',
            f'</table>',
            f'{balance_note}',
            f'</div>'
        ]
        
        # Add content to HTML body
        html_content = html_content + top_content + summary_html
        
        # Process each sample
        for idx, row in display_df.iterrows():
            # Get ground truth label
            ground_truth = None
            if 'reward_model' in row and isinstance(row['reward_model'], dict):
                ground_truth_data = row['reward_model'].get('ground_truth', None)
                if isinstance(ground_truth_data, dict) and 'label' in ground_truth_data:
                    ground_truth = ground_truth_data
            
            # Get data source for reward function selection
            data_source = row.get('data_source')
            
            # Process responses
            responses = row.get('responses', [])
            
            # Handle different types of response data
            if responses is None:
                responses = ["No response available"]
            elif not isinstance(responses, list):
                # Try to convert to list if it's another iterable
                try:
                    responses = list(responses)
                except (TypeError, ValueError):
                    responses = [responses]
            
            # Get first response (assuming single response per example)
            response_text = responses[0] if responses and len(responses) > 0 else "No response generated"
            
            # Ensure response_text is a string
            if not isinstance(response_text, str):
                try:
                    if isinstance(response_text, bytes):
                        response_text = response_text.decode('utf-8')
                    else:
                        response_text = str(response_text)
                except Exception as e:
                    print(f"Warning: Could not convert response to string: {e}")
                    response_text = str(response_text)  # Last resort
            
            # Filter out <|endoftext|> tags
            response_text = response_text.replace("<|endoftext|>", "")
            
            # Get input prompt content
            input_prompt = row.get('prompt')
            input_prompt_content = None
            if input_prompt and isinstance(input_prompt, list) and len(input_prompt) > 0:
                if isinstance(input_prompt[0], dict) and 'content' in input_prompt[0]:
                    input_prompt_content = input_prompt[0]['content']
            
            # Clean response text
            cleaned_response_text = clean_response_text(response_text, input_prompt_content)
            
            # Remove prompt content from response
            if input_prompt_content:
                cleaned_response_text = cleaned_response_text.replace(input_prompt_content, "")
            
            # Preserve original content, don't extract thinking and answer
            raw_thinking = extract_think_content(cleaned_response_text)
            raw_answer = extract_answer_content(cleaned_response_text)
            
            # Calculate token length
            token_length = calculate_token_length(cleaned_response_text, tokenizer)
            
            # Get reward function for the data source
            reward_fn = select_reward_fn(data_source)
            
            # Extract prediction function
            extract_answer=_select_parse_fn(data_source)

            # Evaluate with the appropriate reward function
            metric = False
            prediction = None
            
            # First try with the dedicated reward function
            metric = reward_fn(cleaned_response_text, ground_truth)

            
            # Extract prediction for display with enhanced matching
            prediction = None
            # First check if there's a clean answer tag
            if raw_answer and raw_answer.strip().isdigit():
                prediction = int(raw_answer.strip())
            else:
                # Try to extract prediction using get_prediction_result function
                parsed_prediction = extract_answer(cleaned_response_text)
                if parsed_prediction is not None:
                    prediction = parsed_prediction
                elif ground_truth and 'label' in ground_truth and metric:
                    # If correct but can't parse prediction, use ground truth
                    prediction = ground_truth['label']
                
            
            if task_type == "classification" and metric: displayed_correct_predictions += 1
            
            # Start building the sample HTML
            html_content.append(f'<div class="sample">')
            html_content.append(f'<h2>Sample {idx+1}</h2>')
            
            # Add ICL and Test Data Configuration section
            html_content.append(f'<div class="section">')
            html_content.append(f'<div class="section-title">Configuration Information</div>')
            html_content.append(f'<details>')
            html_content.append(f'<summary>Show Configuration</summary>')
            
            # Display ICL Meta Info
            html_content.append(f'<div style="margin-top: 10px;">')
            html_content.append(f'<h4>ICL Example Meta Info</h4>')
            
            icl_meta_info = row.get('icl_example_meta_info', [])
            if icl_meta_info is not None:
                # Convert NumPy arrays to regular Python types
                if hasattr(icl_meta_info, '__iter__') and not isinstance(icl_meta_info, (str, dict)):
                    # For iterable objects like arrays and lists
                    html_content.append(f'<table style="width: 100%; border-collapse: collapse;">')
                    html_content.append(f'<tr style="background-color: #f2f2f2;"><th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Index</th><th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Value</th></tr>')
                    
                    for i, item in enumerate(icl_meta_info):
                        # Convert each item to regular python type if needed
                        if hasattr(item, 'item') and callable(getattr(item, 'item')):
                            try:
                                item = item.item()  # Convert NumPy scalar to Python scalar
                            except:
                                item = str(item)
                        elif hasattr(item, 'tolist') and callable(getattr(item, 'tolist')):
                            try:
                                item = item.tolist()  # Convert NumPy array to Python list
                            except:
                                item = str(item)
                        
                        if isinstance(item, dict):
                            # Handle dict items
                            for key, value in item.items():
                                # Convert numpy values to regular Python types
                                if hasattr(value, 'item') and callable(getattr(value, 'item')):
                                    try:
                                        value = value.item()
                                    except:
                                        value = str(value)
                                elif hasattr(value, 'tolist') and callable(getattr(value, 'tolist')):
                                    try:
                                        value = value.tolist()
                                    except:
                                        value = str(value)
                                html_content.append(f'<tr><td style="border: 1px solid #ddd; padding: 8px;">{key}</td><td style="border: 1px solid #ddd; padding: 8px;">{value}</td></tr>')
                        else:
                            # Handle non-dict items
                            html_content.append(f'<tr><td style="border: 1px solid #ddd; padding: 8px;">{i}</td><td style="border: 1px solid #ddd; padding: 8px;">{item}</td></tr>')
                    
                    html_content.append(f'</table>')
                elif isinstance(icl_meta_info, dict):
                    # If meta_info is a dictionary, show its key-value pairs
                    html_content.append(f'<table style="width: 100%; border-collapse: collapse;">')
                    html_content.append(f'<tr style="background-color: #f2f2f2;"><th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Property</th><th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Value</th></tr>')
                    
                    for key, value in icl_meta_info.items():
                        # Convert numpy values to regular Python types
                        if hasattr(value, 'item') and callable(getattr(value, 'item')):
                            try:
                                value = value.item()  # Convert NumPy scalar to Python scalar
                            except:
                                value = str(value)
                        elif hasattr(value, 'tolist') and callable(getattr(value, 'tolist')):
                            try:
                                value = value.tolist()  # Convert NumPy array to Python list
                            except:
                                value = str(value)
                        html_content.append(f'<tr><td style="border: 1px solid #ddd; padding: 8px;">{key}</td><td style="border: 1px solid #ddd; padding: 8px;">{value}</td></tr>')
                    
                    html_content.append(f'</table>')
                else:
                    # If it's not iterable or dict (scalar values)
                    value = icl_meta_info
                    if hasattr(value, 'item') and callable(getattr(value, 'item')):
                        try:
                            value = value.item()
                        except:
                            value = str(value)
                    elif hasattr(value, 'tolist') and callable(getattr(value, 'tolist')):
                        try:
                            value = value.tolist()
                        except:
                            value = str(value)
                    html_content.append(f'<div>{value}</div>')
            else:
                html_content.append(f'<div>No ICL meta information available</div>')
            
            html_content.append(f'</div>')
            
            # Display Test Data Configuration
            html_content.append(f'<div style="margin-top: 20px;">')
            html_content.append(f'<h4>Test Data Configuration</h4>')
            
            test_data_config = row.get('test_data', {})
            if test_data_config and isinstance(test_data_config, dict):
                html_content.append(f'<table style="width: 100%; border-collapse: collapse;">')
                html_content.append(f'<tr style="background-color: #f2f2f2;"><th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Property</th><th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Value</th></tr>')
                
                for key, value in test_data_config.items():
                    html_content.append(f'<tr><td style="border: 1px solid #ddd; padding: 8px;">{key}</td><td style="border: 1px solid #ddd; padding: 8px;">{value}</td></tr>')
                
                html_content.append(f'</table>')
            else:
                html_content.append(f'<div>No test data configuration available</div>')
            
            html_content.append(f'</div>')
            
            # Display extra info if available
            if 'extra_info' in row and isinstance(row['extra_info'], dict):
                html_content.append(f'<div style="margin-top: 20px;">')
                html_content.append(f'<h4>Extra Information</h4>')
                
                extra_info = row['extra_info']
                html_content.append(f'<table style="width: 100%; border-collapse: collapse;">')
                html_content.append(f'<tr style="background-color: #f2f2f2;"><th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Property</th><th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Value</th></tr>')
                
                for key, value in extra_info.items():
                    html_content.append(f'<tr><td style="border: 1px solid #ddd; padding: 8px;">{key}</td><td style="border: 1px solid #ddd; padding: 8px;">{value}</td></tr>')
                
                html_content.append(f'</table>')
                html_content.append(f'</div>')
            
            html_content.append(f'</details>')
            html_content.append(f'</div>')
            
            # Data source
            if 'data_source' in row:
                html_content.append(f'<div class="section">')
                html_content.append(f'<div class="section-title">Data Source</div>')
                html_content.append(f'<div>{row["data_source"]}</div>')
                html_content.append(f'</div>')
            
            # Add ICL Examples section (collapsed by default)
            if 'icl_examples' in row:
                html_content.append(f'<div class="section">')
                html_content.append(f'<div class="section-title">ICL Examples</div>')
                html_content.append(f'<details>')
                html_content.append(f'<summary>Show ICL Examples</summary>')
                
                icl_examples = row.get('icl_examples', [])
                
                # Handle various formats of ICL examples
                if isinstance(icl_examples, str):
                    try:
                        # Try to parse as JSON if it's a string representation of a list
                        if icl_examples.startswith('[') and icl_examples.endswith(']'):
                            icl_examples = json.loads(icl_examples)
                        else:
                            icl_examples = [icl_examples]
                    except:
                        icl_examples = [icl_examples]
                
                html_content.append(f'<div style="font-size: 0.9em; margin-bottom: 10px;"><b>Total ICL Examples: {len(icl_examples)}</b></div>')
                
                # Display each ICL example
                for i, example in enumerate(icl_examples):
                    html_content.append(f'<div style="border: 1px solid #ddd; margin-bottom: 10px; padding: 10px; border-radius: 5px;">')
                    html_content.append(f'<h4>Example {i+1}</h4>')
                    
                    # Try to parse the example if it's a string
                    example_data = None
                    if isinstance(example, str):
                        try:
                            example_data = json.loads(example)
                        except:
                            # If not valid JSON, just use as text
                            pass
                    else:
                        example_data = example
                    
                    if example_data and isinstance(example_data, dict):
                        # Extract prompt content
                        if 'prompt' in example_data:
                            prompt_content = None
                            if isinstance(example_data['prompt'], list) and len(example_data['prompt']) > 0:
                                if isinstance(example_data['prompt'][0], dict) and 'content' in example_data['prompt'][0]:
                                    prompt_content = example_data['prompt'][0]['content']
                            elif isinstance(example_data['prompt'], dict) and 'content' in example_data['prompt']:
                                prompt_content = example_data['prompt']['content']
                            
                            if prompt_content:
                                prompt_token_length = calculate_token_length(prompt_content, tokenizer)
                                html_content.append(f'<div><b>Prompt:</b> <span style="color:#666; font-size:0.9em;">[{prompt_token_length} tokens]</span></div>')
                                html_content.append(f'<div style="white-space: pre-wrap; font-family: monospace; background-color: #f5f5f5; padding: 5px; margin-bottom: 5px; max-height: 200px; overflow-y: auto;">{html.escape(prompt_content)}</div>')
                        
                        # Extract reasonings
                        if 'reasonings' in example_data and example_data['reasonings']:
                            reasonings = example_data['reasonings']
                            if isinstance(reasonings, list) and len(reasonings) > 0:
                                reasoning_text = reasonings[0]
                            else:
                                reasoning_text = str(reasonings)
                            
                            # 计算token数量
                            reasoning_token_length = calculate_token_length(reasoning_text, tokenizer)
                            html_content.append(f'<div><b>Reasoning:</b> <span style="color:#666; font-size:0.9em;">[{reasoning_token_length} tokens]</span></div>')
                            html_content.append(f'<div style="white-space: pre-wrap; font-family: monospace; background-color: #f5f5f5; padding: 5px; margin-bottom: 5px; max-height: 150px; overflow-y: auto;">{html.escape(reasoning_text)}</div>')
                        
                        # Extract responses
                        if 'responses' in example_data and example_data['responses']:
                            responses = example_data['responses']
                            if isinstance(responses, list) and len(responses) > 0:
                                response_text = responses[0]
                            else:
                                response_text = str(responses)
                            
                            # 计算token数量
                            response_token_length = calculate_token_length(response_text, tokenizer)
                            html_content.append(f'<div><b>Response:</b> <span style="color:#666; font-size:0.9em;">[{response_token_length} tokens]</span></div>')
                            html_content.append(f'<div style="white-space: pre-wrap; font-family: monospace; background-color: #f5f5f5; padding: 5px; max-height: 150px; overflow-y: auto;">{html.escape(response_text)}</div>')
                        
                        # Extract data source if available
                        example_data_source = None
                        if 'data_source' in example_data:
                            example_data_source = example_data['data_source']
                        elif data_source:  # Use the row's data_source as fallback
                            example_data_source = data_source
                        
                        if example_data_source:
                            html_content.append(f'<div><b>Data Source:</b> {example_data_source}</div>')
                        
                        # Extract features and label if available
                        if 'features' in example_data and isinstance(example_data['features'], list):
                            features = example_data['features']
                            features_str = ", ".join([f"{x:.3f}" for x in features])
                            html_content.append(f'<div><b>Features:</b> [{features_str}]</div>')
                        
                        if 'ground_truth' in example_data and isinstance(example_data['ground_truth'], dict):
                            ground_truth = example_data['ground_truth']
                            if 'label' in ground_truth:
                                html_content.append(f'<div><b>Label:</b> {ground_truth["label"]}</div>')
                    else:
                        # Just display the raw example text
                        html_content.append(f'<div style="white-space: pre-wrap; font-family: monospace; max-height: 300px; overflow-y: auto;">{html.escape(str(example))}</div>')
                    
                    html_content.append(f'</div>')
                
                html_content.append(f'</details>')
                html_content.append(f'</div>')
            
            # Add Test Examples section (collapsed by default)
            if 'test_examples' in row:
                html_content.append(f'<div class="section">')
                html_content.append(f'<div class="section-title">Test Examples</div>')
                html_content.append(f'<details>')
                html_content.append(f'<summary>Show Test Examples</summary>')
                
                test_examples = row.get('test_examples', '[]')
                
                # Parse test examples - expected to be a JSON string of lists of tuples
                try:
                    if isinstance(test_examples, str):
                        test_examples = json.loads(test_examples)
                except Exception as e:
                    test_examples = []
                    print(f"Error parsing test examples: {e}")
                
                html_content.append(f'<div style="font-size: 0.9em; margin-bottom: 10px;"><b>Total Test Examples: {len(test_examples)}</b></div>')
                
                # Display test examples in a table for better readability
                if test_examples:
                    html_content.append(f'<table style="width: 100%; border-collapse: collapse;">')
                    html_content.append(f'<tr style="background-color: #f2f2f2;"><th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Example</th><th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Features</th><th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Label</th></tr>')
                    
                    for i, example in enumerate(test_examples):
                        # Test examples are typically (features, label) tuples
                        if isinstance(example, (list, tuple)) and len(example) >= 2:
                            features, label = example[0], example[1]
                            
                            # Format features for display
                            if isinstance(features, (list, tuple)):
                                features_str = ", ".join([f"{x:.3f}" for x in features])
                                features_display = f"[{features_str}]"
                            else:
                                features_display = str(features)
                            
                            label_display = str(label)
                            
                            html_content.append(f'<tr><td style="border: 1px solid #ddd; padding: 8px;">{i+1}</td><td style="border: 1px solid #ddd; padding: 8px; font-family: monospace;">{features_display}</td><td style="border: 1px solid #ddd; padding: 8px;">{label_display}</td></tr>')
                    
                    html_content.append(f'</table>')
                else:
                    html_content.append(f'<div>No test examples found or unable to parse</div>')
                
                html_content.append(f'</details>')
                html_content.append(f'</div>')
            
            # Input Prompt
            html_content.append(f'<div class="section">')
            html_content.append(f'<div class="section-title">Input Prompt</div>')
            
            prompt = row.get('prompt', None)
            if prompt is not None:
                if isinstance(prompt, list):
                    # Handle list of prompt items
                    html_content.append(f'<details>')
                    html_content.append(f'<summary>Show Input Prompt</summary>')
                    html_content.append(f'<div class="prompt">')
                    for prompt_item in prompt:
                        if isinstance(prompt_item, dict):
                            if 'role' in prompt_item:
                                html_content.append(f'<b>{prompt_item["role"]}:</b><br>')
                            if 'content' in prompt_item:
                                content = prompt_item["content"]
                                
                                if content:
                                    # Escape HTML content
                                    escaped_content = html.escape(content)
                                    html_content.append(f'{escaped_content}<br><br>')
                        else:
                            html_content.append(f'{str(prompt_item)}<br>')
                    html_content.append(f'</div>')
                    html_content.append(f'</details>')
                else:
                    # Handle string or other type of prompt
                    html_content.append(f'<details>')
                    html_content.append(f'<summary>Show Input Prompt</summary>')
                    # Escape HTML content
                    escaped_prompt = html.escape(str(prompt))
                    html_content.append(f'<div class="prompt">{escaped_prompt}</div>')
                    html_content.append(f'</details>')
            else:
                html_content.append(f'<div>No prompt available</div>')
            
            html_content.append(f'</div>')
            
            # Ground truth and features
            html_content.append(f'<div class="section">')
            html_content.append(f'<div class="section-title">Ground Truth</div>')
            if ground_truth is not None:
                features_str = ""
                if 'features' in ground_truth:
                    features = ground_truth['features']
                    if isinstance(features, list):
                        features_str = ", ".join([f"{x:.3f}" for x in features])
                        features_str = f"[{features_str}]"
                    else:
                        features_str = str(features)
                html_content.append(f'<div>Label: {ground_truth.get("label", "N/A")}</div>')
                if features_str:
                    html_content.append(f'<div>Features: {features_str}</div>')
            else:
                html_content.append(f'<div>Not available</div>')
            html_content.append(f'</div>')
            
            # Prediction result (still displayed because it's useful to the user)
            html_content.append(f'<div class="section">')
            html_content.append(f'<div class="section-title">Prediction Result</div>')
            if prediction is not None:
                if task_type == "classification":
                    result_class = "correct" if metric==1 else "incorrect"
                    html_content.append(f'<div class="{result_class}">Predicted: {prediction} ({("CORRECT" if metric else "INCORRECT")})</div>')
                else:
                    result_class = "correct" if metric>REGRESSION_CORRECT_THRESHOLD else "incorrect"
                    html_content.append(f'<div class="{result_class}">Predicted: {prediction} ({("CORRECT" if (result_class=="correct") else "INCORRECT")})</div>')
                    html_content.append(f'<div class="{result_class}">MSE: {metric}</div>')
            else:
                html_content.append(f'<div class="incorrect">Unable to parse prediction</div>')
            
            # Add token length information
            html_content.append(f'<div style="margin-top: 5px; color: #666;">Response Token Length: {token_length}</div>')
            
            html_content.append(f'</div>')
            
            # Restore collapsible full response, but don't extract tag content
            html_content.append(f'<details open>')
            html_content.append(f'<summary>Model Response (Cleaned)</summary>')
            html_content.append(f'<div class="section">')
            # Use html.escape to ensure tags display correctly, not interpreted as HTML tags
            escaped_response = html.escape(cleaned_response_text)
            html_content.append(f'<div class="response" style="white-space: pre-wrap; font-family: monospace;">{escaped_response}</div>')
            html_content.append(f'</div>')
            html_content.append(f'</details>')
            
            # End sample div
            html_content.append(f'</div>')
        
        # Close HTML tags
        html_content.append("</body>")
        html_content.append("</html>")
        
        # Write to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(html_content))
    
    else:  # Text output
        with open(output_file, 'w', encoding='utf-8') as f:
            # Write summary header
            f.write("="*80 + "\n")
            f.write(f"ICL REASONING RESULTS: {Path(input_file).name}\n")
            f.write("="*80 + "\n\n")
            
            # Process each sample
            for idx, row in display_df.iterrows():
                f.write(f"=== Sample {idx+1} ===\n\n")
                
                # Write data source if available
                if 'data_source' in row:
                    f.write(f"--- Data Source ---\n")
                    f.write(f"{row['data_source']}\n\n")
                
                # Write input prompt
                f.write("--- Input Prompt ---\n")
                prompt = row.get('prompt', None)
                if prompt is not None:
                    if isinstance(prompt, list):
                        # Handle list of prompt items
                        for prompt_item in prompt:
                            if isinstance(prompt_item, dict):
                                if 'role' in prompt_item:
                                    f.write(f"{prompt_item['role']}:\n")
                                if 'content' in prompt_item:
                                    content = prompt_item["content"]
                                    if content:
                                        f.write(f"{content}\n\n")

                            else:
                                f.write(f"{str(prompt_item)}\n")
                    else:
                        # Handle string or other type of prompt
                        f.write(f"{str(prompt)}\n")
                else:
                    f.write("No prompt available\n")
                f.write("\n")
                
                # Add ICL Examples section in text format
                if 'icl_examples' in row:
                    f.write("--- ICL Examples ---\n")
                    
                    icl_examples = row.get('icl_examples', [])
                    
                    # Handle various formats of ICL examples
                    if isinstance(icl_examples, str):
                        try:
                            # Try to parse as JSON if it's a string representation of a list
                            if icl_examples.startswith('[') and icl_examples.endswith(']'):
                                icl_examples = json.loads(icl_examples)
                            else:
                                icl_examples = [icl_examples]
                        except:
                            icl_examples = [icl_examples]
                    
                    f.write(f"Total ICL Examples: {len(icl_examples)}\n")
                    
                    # Display each ICL example (summarized version for text format)
                    for i, example in enumerate(icl_examples):
                        f.write(f"\nExample {i+1}:\n")
                        f.write("-" * 40 + "\n")
                        
                        # Try to parse the example if it's a string
                        example_data = None
                        if isinstance(example, str):
                            try:
                                example_data = json.loads(example)
                            except:
                                # If not valid JSON, just use as text
                                pass
                        else:
                            example_data = example
                        
                        if example_data and isinstance(example_data, dict):
                            # Extract key information (brief summary for text format)
                            # Features and label if available
                            if 'features' in example_data and isinstance(example_data['features'], list):
                                features = example_data['features']
                                features_str = ", ".join([f"{x:.3f}" for x in features])
                                f.write(f"Features: [{features_str}]\n")
                            
                            # Add data source if available
                            example_data_source = None
                            if 'data_source' in example_data:
                                example_data_source = example_data['data_source']
                            elif data_source:  # Use the row's data_source as fallback
                                example_data_source = data_source
                            
                            if example_data_source:
                                f.write(f"Data Source: {example_data_source}\n")
                            
                            if 'ground_truth' in example_data and isinstance(example_data['ground_truth'], dict):
                                ground_truth = example_data['ground_truth']
                                if 'label' in ground_truth:
                                    f.write(f"Label: {ground_truth['label']}\n")
                            
                            # Reasonable length for reasoning and response (truncated)
                            if 'reasonings' in example_data and example_data['reasonings']:
                                reasonings = example_data['reasonings']
                                if isinstance(reasonings, list) and len(reasonings) > 0:
                                    reasoning_text = reasonings[0]
                                else:
                                    reasoning_text = str(reasonings)
                                
                                # 计算token数量
                                reasoning_token_length = calculate_token_length(reasoning_text, tokenizer)
                                
                                # Truncate if too long for text format
                                if len(reasoning_text) > 300:
                                    reasoning_text = reasoning_text[:300] + "... [truncated]"
                                
                                f.write(f"\nReasoning: [{reasoning_token_length} tokens] {reasoning_text}\n")
                            
                            # Brief response summary
                            if 'responses' in example_data and example_data['responses']:
                                responses = example_data['responses']
                                if isinstance(responses, list) and len(responses) > 0:
                                    response_text = responses[0]
                                else:
                                    response_text = str(responses)
                                
                                # 计算token数量
                                response_token_length = calculate_token_length(response_text, tokenizer)
                                
                                # Truncate if too long for text format
                                if len(response_text) > 200:
                                    response_text = response_text[:200] + "... [truncated]"
                                
                                f.write(f"\nResponse: [{response_token_length} tokens] {response_text}\n")
                        else:
                            # Just display a brief summary for text
                            example_str = str(example)
                            if len(example_str) > 500:
                                example_str = example_str[:500] + "... [truncated]"
                            f.write(f"{example_str}\n")
                    
                    f.write("\n")
                
                # Add Test Examples section in text format
                if 'test_examples' in row:
                    f.write("--- Test Examples ---\n")
                    
                    test_examples = row.get('test_examples', '[]')
                    
                    # Parse test examples - expected to be a JSON string of lists of tuples
                    try:
                        if isinstance(test_examples, str):
                            test_examples = json.loads(test_examples)
                    except Exception as e:
                        test_examples = []
                        print(f"Error parsing test examples: {e}")
                    
                    f.write(f"Total Test Examples: {len(test_examples)}\n\n")
                    
                    # Display test examples in a simple list format
                    for i, example in enumerate(test_examples):
                        # Test examples are typically (features, label) tuples
                        if isinstance(example, (list, tuple)) and len(example) >= 2:
                            features, label = example[0], example[1]
                            
                            # Format features for display
                            if isinstance(features, (list, tuple)):
                                features_str = ", ".join([f"{x:.3f}" for x in features])
                                features_display = f"[{features_str}]"
                            else:
                                features_display = str(features)
                            
                            label_display = str(label)
                            
                            f.write(f"Example {i+1}: Features: {features_display}, Label: {label_display}\n")
                        else:
                            f.write(f"Example {i+1}: {str(example)}\n")
                    
                    f.write("\n")
                
                # Get ground truth label
                ground_truth = None
                if 'reward_model' in row and isinstance(row['reward_model'], dict):
                    ground_truth_data = row['reward_model'].get('ground_truth', None)
                    if isinstance(ground_truth_data, dict) and 'label' in ground_truth_data:
                        ground_truth = ground_truth_data
                
                # Add Configuration Information section in text format
                f.write("--- Configuration Information ---\n")
                
                # ICL Meta Info
                f.write("ICL Example Meta Info:\n")
                icl_meta_info = row.get('icl_example_meta_info', [])
                if icl_meta_info is not None:
                    # Convert NumPy arrays to regular Python types
                    if hasattr(icl_meta_info, '__iter__') and not isinstance(icl_meta_info, (str, dict)):
                        # For iterable objects like arrays and lists
                        for i, item in enumerate(icl_meta_info):
                            # Convert each item to regular python type if needed
                            if hasattr(item, 'item') and callable(getattr(item, 'item')):
                                try:
                                    item = item.item()  # Convert NumPy scalar to Python scalar
                                except:
                                    item = str(item)
                            elif hasattr(item, 'tolist') and callable(getattr(item, 'tolist')):
                                try:
                                    item = item.tolist()  # Convert NumPy array to Python list
                                except:
                                    item = str(item)
                            
                            if isinstance(item, dict):
                                # Handle dict items
                                for key, value in item.items():
                                    # Convert numpy values to regular Python types
                                    if hasattr(value, 'item') and callable(getattr(value, 'item')):
                                        try:
                                            value = value.item()
                                        except:
                                            value = str(value)
                                    elif hasattr(value, 'tolist') and callable(getattr(value, 'tolist')):
                                        try:
                                            value = value.tolist()
                                        except:
                                            value = str(value)
                                    f.write(f"  {key}: {value}\n")
                            else:
                                # Handle non-dict items
                                f.write(f"  Item {i}: {item}\n")
                    elif isinstance(icl_meta_info, dict):
                        # If meta_info is a dictionary, show its key-value pairs
                        for key, value in icl_meta_info.items():
                            # Convert numpy values to regular Python types
                            if hasattr(value, 'item') and callable(getattr(value, 'item')):
                                try:
                                    value = value.item()  # Convert NumPy scalar to Python scalar
                                except:
                                    value = str(value)
                            elif hasattr(value, 'tolist') and callable(getattr(value, 'tolist')):
                                try:
                                    value = value.tolist()  # Convert NumPy array to Python list
                                except:
                                    value = str(value)
                            f.write(f"  {key}: {value}\n")
                    else:
                        # If it's not iterable or dict (scalar values)
                        value = icl_meta_info
                        if hasattr(value, 'item') and callable(getattr(value, 'item')):
                            try:
                                value = value.item()
                            except:
                                value = str(value)
                        elif hasattr(value, 'tolist') and callable(getattr(value, 'tolist')):
                            try:
                                value = value.tolist()
                            except:
                                value = str(value)
                        f.write(f"  Data: {value}\n")
                else:
                    f.write("  No ICL meta information available\n")
                
                f.write("\n")
                
                # Test Data Configuration
                f.write("Test Data Configuration:\n")
                test_data_config = row.get('test_data', {})
                if test_data_config and isinstance(test_data_config, dict):
                    for key, value in test_data_config.items():
                        f.write(f"  {key}: {value}\n")
                else:
                    f.write("  No test data configuration available\n")
                
                f.write("\n")
                
                # Extra Info
                if 'extra_info' in row and isinstance(row['extra_info'], dict):
                    f.write("Extra Information:\n")
                    extra_info = row['extra_info']
                    for key, value in extra_info.items():
                        f.write(f"  {key}: {value}\n")
                    f.write("\n")
                
                # Write ground truth information
                f.write("--- Ground Truth ---\n")
                if ground_truth is not None:
                    f.write(f"Label: {ground_truth.get('label', 'N/A')}\n")
                    if 'features' in ground_truth:
                        features = ground_truth['features']
                        if isinstance(features, list):
                            features_str = ", ".join([f"{x:.3f}" for x in features])
                            f.write(f"Features: [{features_str}]\n")
                        else:
                            f.write(f"Features: {features}\n")
                else:
                    f.write("Not available\n")
                f.write("\n")
                
                # Process responses
                responses = row.get('responses', [])
                
                # Handle different types of response data
                if responses is None:
                    responses = ["No response available"]
                elif not isinstance(responses, list):
                    # Try to convert to list if it's another iterable
                    try:
                        responses = list(responses)
                    except (TypeError, ValueError):
                        responses = [responses]
                
                # Get first response (assuming single response per example)
                response_text = responses[0] if responses and len(responses) > 0 else "No response generated"
                
                # Ensure response_text is a string
                if not isinstance(response_text, str):
                    try:
                        if isinstance(response_text, bytes):
                            response_text = response_text.decode('utf-8')
                        else:
                            response_text = str(response_text)
                    except Exception as e:
                        print(f"Warning: Could not convert response to string: {e}")
                        response_text = str(response_text)  # Last resort
                
                # Filter out <|endoftext|> tags
                response_text = response_text.replace("<|endoftext|>", "")
                
                # Get input prompt content
                input_prompt = row.get('prompt')
                input_prompt_content = None
                if input_prompt and isinstance(input_prompt, list) and len(input_prompt) > 0:
                    if isinstance(input_prompt[0], dict) and 'content' in input_prompt[0]:
                        input_prompt_content = input_prompt[0]['content']
                
                # Clean response text
                cleaned_response_text = clean_response_text(response_text, input_prompt_content)
                
                # Remove prompt content from response
                if input_prompt_content:
                    cleaned_response_text = cleaned_response_text.replace(input_prompt_content, "")
                
                # Preserve original content, don't extract thinking and answer
                raw_thinking = extract_think_content(cleaned_response_text)
                raw_answer = extract_answer_content(cleaned_response_text)
                
                # Calculate token length
                token_length = calculate_token_length(cleaned_response_text, tokenizer)
                
                # Get data source for reward function selection
                data_source = row.get('data_source', 'blobs')
                
                # Get reward function for the data source
                reward_fn = select_reward_fn(data_source)
                
                # Evaluate with the appropriate reward function
                metric = False
                prediction = None
                
                
                # First try with the dedicated reward function
                metric = reward_fn(cleaned_response_text, ground_truth)

                
                # Extract prediction for display with enhanced matching
                prediction = None
                # First check if there's a clean answer tag
                if raw_answer and raw_answer.strip().isdigit():
                    prediction = int(raw_answer.strip())
                else:
                    # Try to extract prediction using get_prediction_result function
                    parsed_prediction, _ = extract_answer(cleaned_response_text, ground_truth, task_type)
                    if parsed_prediction is not None:
                        prediction = parsed_prediction
                    elif ground_truth and 'label' in ground_truth and metric:
                        # If correct but can't parse prediction, use ground truth
                        prediction = ground_truth['label']

                if metric:
                    displayed_correct_predictions += 1
                
                # Write prediction result
                f.write("--- Prediction Result ---\n")
                if prediction is not None:
                    result_str = "CORRECT" if metric else "INCORRECT"
                    f.write(f"Predicted: {prediction} ({result_str})\n")
                else:
                    f.write("Unable to parse prediction\n")
                
                # Add token length information
                f.write(f"Response Token Length: {token_length}\n")
                
                f.write("\n")
                
                # Write full response (don't extract tag content)
                f.write(f"--- Model Response (Cleaned) ---\n")
                f.write(f"{cleaned_response_text}\n")
                f.write("\n")
                
                # Write separator
                f.write("="*80 + "\n\n")
            
            # Write summary at the end
            # Note: Use the previously calculated accuracy and refined_accuracy variables
            
            sampled_note = ""
            balance_note = ""
            if max_samples > 0 and max_samples < total_data_size:
                sampled_note = f" (showing {displayed_samples} out of {total_data_size})"
                if balanced_sampling:
                    balance_note = f"Balanced sampling: {target_correct} correct, {target_incorrect} incorrect"
            f.write("="*80 + "\n")
            f.write("SUMMARY\n")
            f.write("="*80 + "\n")
            f.write(f"Total samples{sampled_note}: {total_data_size}\n")
            f.write(f"Correct predictions (all data): {full_correct_predictions}\n")
            f.write(f"Accuracy (all data): {accuracy*100:.2f}%\n")
            f.write(f"Refined accuracy: {refined_accuracy*100:.2f}%\n")
            f.write(f"Parseable accuracy: {parseable_accuracy*100:.2f}% ({parseable_predictions}/{total_data_size} samples)\n")
            f.write(f"Unparseable predictions: {unparseable_predictions} ({unparseable_predictions/total_data_size*100:.2f}%)\n")
            if balance_note:
                f.write(f"{balance_note}\n")
            f.write("="*80 + "\n")
    
    print(f"Visualization saved to: {output_file}")
    
    return str(output_file)


def main():
    parser = argparse.ArgumentParser(description='Visualize ICL reasoning output with model responses')
    parser.add_argument('--input', type=str, 
                        required=True,
                        help='Path to the input parquet file with model responses')
    parser.add_argument('--format', type=str, choices=['txt', 'html'], default='html',
                        help='Output format (txt or html)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save the output file (default: same as input file)')
    parser.add_argument('--max-samples', type=int, default=100,
                        help='Maximum number of samples to visualize (default: 100, use 0 for all)')
    parser.add_argument('--REGRESSION_CORRECT_THRESHOLD', type=float, default=-0.01,
                        help='Threshold for regression correctness (default:-0.01)')
    parser.add_argument('--output-csv', type=str, default=None,
                        help='Output results to CSV file')
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: File '{args.input}' not found")
        return
    
    # Visualize the output
    output_file = visualize_icl_reasoning_output(args.input, args.format, args.output_dir, args.max_samples, args.REGRESSION_CORRECT_THRESHOLD, args.output_csv)
    print(f"Visualization complete! Saved to: {output_file}")


if __name__ == "__main__":
    main() 