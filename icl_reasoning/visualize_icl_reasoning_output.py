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

# Import function to get the number of classes
try:
    from liftr.icl_reasoning.icl_reasoning import get_num_classes
    from transformers import AutoTokenizer
except ImportError:
    # If unable to import, provide a simple fallback function
    def get_num_classes(task_type: str) -> int:
        """
        Return the number of classes for each task type
        """
        num_classes = {
            "blobs": 3,
            "circles": 2,
            "linear": 2,
            "moons": 2
        }
        if task_type not in num_classes:
            return 2  # Default value
        
        return num_classes[task_type]
    
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


def get_prediction_result(response_text: Optional[str], ground_truth) -> Tuple[Optional[int], bool]:
    """
    Evaluate prediction using main_eval approach
    
    Args:
        response_text: Full model response text
        ground_truth: The ground truth data containing label and features
        
    Returns:
        Tuple of (predicted_label, is_correct)
    """
    # Import classification_reward_fn from helper module
    try:
        from examples.data_preprocess.helper import classification_reward_fn
        # Use the classification_reward_fn for evaluation
        is_correct = classification_reward_fn(response_text, ground_truth)
        
        # Extract the predicted label for display
        all_matches = list(re.finditer(r'<answer>(.*?)</answer>', response_text, re.DOTALL))
        if all_matches:
            response_extract = None
            for match in all_matches[::-1]:  # Check from last to first
                if match.group(1).strip().isdigit():
                    response_extract = match
                    break
            if response_extract is not None and response_extract.group(1).strip().isdigit():
                prediction = int(response_extract.group(1).strip())
                return prediction, is_correct
        # If no valid prediction found but potentially marked as correct
        if is_correct:
            # This shouldn't happen with the classification_reward_fn logic
            return ground_truth['label'], True
        # Otherwise, couldn't extract prediction
        return None, False
    except ImportError:
        # Fallback to original implementation if module not available
        if response_text is None:
            return None, False
        
        # Make sure ground_truth_label is available
        if ground_truth is None or 'label' not in ground_truth:
            return None, False
            
        ground_truth_label = ground_truth['label']
        
        # Try to parse an integer from the answer
        answer = extract_answer_content(response_text)
        if answer is not None and answer.strip().isdigit():
            prediction = int(answer.strip())
            return prediction, prediction == ground_truth_label
        
        # Look for integer patterns in the full response
        int_pattern = r'<answer>\s*(\d+)\s*</answer>'
        int_matches = re.findall(int_pattern, response_text)
        if int_matches:
            prediction = int(int_matches[0])
            return prediction, prediction == ground_truth_label
        
        # If no integer found, return None
        return None, False


def visualize_icl_reasoning_output(input_file: str, output_format: str = "txt", save_dir: Optional[str] = None, max_samples: int = 100):
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
    try:
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
        
    except Exception as e:
        print(f"Error reading parquet file: {e}")
        raise
    
    # Import reward function selection from main_eval
    try:
        from verl.trainer.ppo.helper import _select_rm_score_fn as select_reward_fn
        print("Successfully imported select_reward_fn from verl.trainer.ppo.helper")
    except ImportError:
        # Define a basic fallback if imports not available
        def select_reward_fn(data_source):
            if "blobs" in data_source:
                try:
                    from examples.data_preprocess.blobs import blobs_reward_fn
                    return blobs_reward_fn
                except ImportError:
                    print("Warning: Could not import blobs_reward_fn")
            
            # Default reward function (will use our get_prediction_result)
            return lambda solution_str, ground_truth: get_prediction_result(solution_str, ground_truth)[1]
        
        print("Using fallback select_reward_fn function")
        
    # Calculate accuracy on full dataset first
    total_data_size = len(df)
    full_correct_predictions = 0
    full_refined_correct = 0  # For storing the correct count for refined accuracy
    unparseable_predictions = 0  # Count of unparseable predictions
    parseable_predictions = 0  # Count of parseable predictions
    parseable_correct = 0  # Count of correct predictions among parseable ones
    
    print(f"Calculating accuracy on all {total_data_size} samples...")
    
    # Process all samples for accuracy calculation
    for _, row in df.iterrows():
        # Get ground truth label
        ground_truth = None
        data_source = row.get('data_source', 'blobs')
        
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
        input_prompt_content = None
        if input_prompt and isinstance(input_prompt, list) and len(input_prompt) > 0:
            if isinstance(input_prompt[0], dict) and 'content' in input_prompt[0]:
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
        
        # Evaluate with the appropriate reward function
        is_correct = False
        can_parse_prediction = True
        
        try:
            # Use the dedicated reward function
            is_correct = reward_fn(cleaned_response_text, ground_truth)
            
            # Try to extract prediction result to check if it can be parsed
            parsed_prediction, _ = get_prediction_result(cleaned_response_text, ground_truth)
            can_parse_prediction = parsed_prediction is not None
            
        except Exception as e:
            # Fallback to simple prediction extraction
            parsed_prediction, is_correct = get_prediction_result(cleaned_response_text, ground_truth)
            can_parse_prediction = parsed_prediction is not None
        
        if is_correct:
            full_correct_predictions += 1
            full_refined_correct += 1  # Correct predictions also count as correct in refined accuracy
            if can_parse_prediction:
                parseable_correct += 1
        
        # Track parseable predictions
        if can_parse_prediction:
            parseable_predictions += 1
        elif ground_truth is not None and 'label' in ground_truth:
            # For samples with unparseable predictions, calculate refined accuracy
            unparseable_predictions += 1
            
            # Get task type to determine number of classes
            num_classes = get_num_classes(data_source)
            
            # Randomly select a label (fixed seed set at function start)
            random_label = random.randint(0, num_classes - 1)
            
            # If randomly selected label matches the true label, count as correct for refined accuracy
            if random_label == ground_truth['label']:
                full_refined_correct += 1
    
    # Calculate overall accuracy and refined accuracy
    accuracy = (full_correct_predictions / total_data_size) * 100 if total_data_size > 0 else 0
    refined_accuracy = (full_refined_correct / total_data_size) * 100 if total_data_size > 0 else 0
    
    # Calculate accuracy for parseable predictions only
    parseable_accuracy = (parseable_correct / parseable_predictions) * 100 if parseable_predictions > 0 else 0
    
    print(f"Overall accuracy from all {total_data_size} samples: {accuracy:.2f}%")
    print(f"Refined accuracy (random guess for unparseable): {refined_accuracy:.2f}%")
    print(f"Parseable accuracy (excluding unparseable): {parseable_accuracy:.2f}% ({parseable_predictions}/{total_data_size} samples)")
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
            
            is_correct = False
            try:
                is_correct = reward_fn(cleaned_response_text, ground_truth)
            except Exception:
                _, is_correct = get_prediction_result(cleaned_response_text, ground_truth)
            
            if is_correct:
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
            "    <title>ICL Reasoning Results - Accuracy: " + f"{accuracy:.2f}%" + "</title>",
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
            f'Accuracy: {accuracy:.2f}% &nbsp;|&nbsp; Refined Accuracy: {refined_accuracy:.2f}%',
            f'</div>',
            f'<div class="accuracy-big" style="background-color: #e9f0ff; border-color: #1565C0;">',
            f'Parseable Accuracy: {parseable_accuracy:.2f}% (excluding {unparseable_predictions} unparseable samples)',
            f'</div>',
            f'<div>Unparseable Predictions: {unparseable_predictions} ({unparseable_predictions/total_data_size*100:.2f}%)</div>',
            f'<hr style="margin: 20px 0; border: 0; height: 2px; background: #333;">',
        ]
        
        summary_html = [
            f'<div class="summary">',
            f'<h2>Results Summary</h2>',
            f'<table>',
            f'<tr><th>Metric</th><th>Value</th></tr>',
            f'<tr><td>Total Samples{sampled_note}</td><td>{total_data_size}</td></tr>',
            f'<tr><td>Correct Predictions (all data)</td><td>{full_correct_predictions}</td></tr>',
            f'<tr><td>Accuracy (all data)</td><td>{accuracy:.2f}%</td></tr>',
            f'<tr><td>Refined Accuracy</td><td>{refined_accuracy:.2f}%</td></tr>',
            f'<tr><td>Parseable Accuracy</td><td>{parseable_accuracy:.2f}% ({parseable_predictions}/{total_data_size} samples)</td></tr>',
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
            
            # Evaluate with the appropriate reward function
            is_correct = False
            prediction = None
            
            try:
                # First try with the dedicated reward function
                is_correct = reward_fn(cleaned_response_text, ground_truth)
                
                # Extract prediction for display
                if raw_answer and raw_answer.strip().isdigit():
                    prediction = int(raw_answer.strip())
                elif ground_truth and 'label' in ground_truth and is_correct:
                    # If correct but can't parse prediction, use ground truth
                    prediction = ground_truth['label']
                
            except Exception as e:
                print(f"Warning: Error using reward function: {e}")
                # Fallback to simple prediction extraction
                prediction, is_correct = get_prediction_result(cleaned_response_text, ground_truth)
            
            if is_correct:
                displayed_correct_predictions += 1
            
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
                                html_content.append(f'<div><b>Prompt:</b></div>')
                                html_content.append(f'<div style="white-space: pre-wrap; font-family: monospace; background-color: #f5f5f5; padding: 5px; margin-bottom: 5px; max-height: 200px; overflow-y: auto;">{html.escape(prompt_content)}</div>')
                        
                        # Extract reasonings
                        if 'reasonings' in example_data and example_data['reasonings']:
                            reasonings = example_data['reasonings']
                            if isinstance(reasonings, list) and len(reasonings) > 0:
                                reasoning_text = reasonings[0]
                            else:
                                reasoning_text = str(reasonings)
                            
                            html_content.append(f'<div><b>Reasoning:</b></div>')
                            html_content.append(f'<div style="white-space: pre-wrap; font-family: monospace; background-color: #f5f5f5; padding: 5px; margin-bottom: 5px; max-height: 150px; overflow-y: auto;">{html.escape(reasoning_text)}</div>')
                        
                        # Extract responses
                        if 'responses' in example_data and example_data['responses']:
                            responses = example_data['responses']
                            if isinstance(responses, list) and len(responses) > 0:
                                response_text = responses[0]
                            else:
                                response_text = str(responses)
                            
                            html_content.append(f'<div><b>Response:</b></div>')
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
                result_class = "correct" if is_correct else "incorrect"
                html_content.append(f'<div class="{result_class}">Predicted: {prediction} ({("CORRECT" if is_correct else "INCORRECT")})</div>')
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
                                
                                # Truncate if too long for text format
                                if len(reasoning_text) > 300:
                                    reasoning_text = reasoning_text[:300] + "... [truncated]"
                                
                                f.write(f"\nReasoning: {reasoning_text}\n")
                            
                            # Brief response summary
                            if 'responses' in example_data and example_data['responses']:
                                responses = example_data['responses']
                                if isinstance(responses, list) and len(responses) > 0:
                                    response_text = responses[0]
                                else:
                                    response_text = str(responses)
                                
                                # Truncate if too long for text format
                                if len(response_text) > 200:
                                    response_text = response_text[:200] + "... [truncated]"
                                
                                f.write(f"\nResponse: {response_text}\n")
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
                is_correct = False
                prediction = None
                
                try:
                    # First try with the dedicated reward function
                    is_correct = reward_fn(cleaned_response_text, ground_truth)
                    
                    # Extract prediction for display
                    if raw_answer and raw_answer.strip().isdigit():
                        prediction = int(raw_answer.strip())
                    elif ground_truth and 'label' in ground_truth and is_correct:
                        # If correct but can't parse prediction, use ground truth
                        prediction = ground_truth['label']
                    
                except Exception as e:
                    print(f"Warning: Error using reward function: {e}")
                    # Fallback to simple prediction extraction
                    prediction, is_correct = get_prediction_result(cleaned_response_text, ground_truth)
                
                if is_correct:
                    displayed_correct_predictions += 1
                
                # Write prediction result
                f.write("--- Prediction Result ---\n")
                if prediction is not None:
                    result_str = "CORRECT" if is_correct else "INCORRECT"
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
            f.write(f"Accuracy (all data): {accuracy:.2f}%\n")
            f.write(f"Refined accuracy: {refined_accuracy:.2f}%\n")
            f.write(f"Parseable accuracy: {parseable_accuracy:.2f}% ({parseable_predictions}/{total_data_size} samples)\n")
            f.write(f"Unparseable predictions: {unparseable_predictions} ({unparseable_predictions/total_data_size*100:.2f}%)\n")
            if balance_note:
                f.write(f"{balance_note}\n")
            f.write("="*80 + "\n")
    
    print(f"Visualization saved to: {output_file}")
    print(f"Total samples{sampled_note}: {total_data_size}")
    print(f"Correct predictions (all data): {full_correct_predictions}")
    print(f"Accuracy (all data): {accuracy:.2f}%")
    print(f"Refined accuracy: {refined_accuracy:.2f}%")
    print(f"Parseable accuracy: {parseable_accuracy:.2f}% ({parseable_predictions}/{total_data_size} samples)")
    print(f"Unparseable predictions: {unparseable_predictions} ({unparseable_predictions/total_data_size*100:.2f}%)")
    
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
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: File '{args.input}' not found")
        return
    
    # Visualize the output
    output_file = visualize_icl_reasoning_output(args.input, args.format, args.output_dir, args.max_samples)
    print(f"Visualization complete! Saved to: {output_file}")


if __name__ == "__main__":
    main() 