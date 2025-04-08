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
from typing import Dict, Any, List, Optional, Tuple, Callable
import math

# Import helper functions from data_processing
from data_processing import (
    process_icl_examples, 
    process_test_examples, 
    add_sample_data_fields,
    process_icl_examples_for_display,
    process_test_examples_for_display
)


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
                match_content = match.group(1).strip()
                if match_content.isdigit():
                    response_extract = match
                    break
            if response_extract is not None and response_extract.group(1).strip().isdigit():
                prediction = int(response_extract.group(1).strip())
                return prediction, is_correct
            
            # Try direct pattern matching if the tags might have whitespace issues
            int_pattern = r'<answer>\s*(\d+)\s*</answer>'
            int_matches = re.findall(int_pattern, response_text)
            if int_matches:
                prediction = int(int_matches[-1])  # Use the last match
                return prediction, is_correct
        
        # If is_correct but didn't find valid prediction, try more aggressive patterns
        if is_correct:
            # Look for numbers after "answer:" or "class:" patterns that might appear in text
            alternative_patterns = [
                r'answer:\s*(\d+)',
                r'class:\s*(\d+)',
                r'prediction:\s*(\d+)',
                r'label:\s*(\d+)',
                r'the answer is\s*(\d+)',
                r'class is\s*(\d+)'
            ]
            
            for pattern in alternative_patterns:
                matches = re.findall(pattern, response_text, re.IGNORECASE)
                if matches:
                    try:
                        prediction = int(matches[-1])  # Use the last match
                        return prediction, is_correct
                    except (ValueError, TypeError):
                        continue  # Try next pattern if this one didn't work
            
            # Final fallback: just find any number that could be a valid class
            if ground_truth and 'label' in ground_truth:
                # Get task type to determine number of classes
                num_classes = 0
                try:
                    data_source = ground_truth.get('data_source', 'blobs')
                    num_classes = get_num_classes(data_source)
                except:
                    num_classes = 3  # Default for most classification tasks
                
                # Extract all digit sequences and check if any could be a valid class
                digit_matches = re.findall(r'\b(\d+)\b', response_text)
                for match in digit_matches[::-1]:  # Check from last to first
                    try:
                        digit_val = int(match)
                        if 0 <= digit_val < num_classes:
                            # Found a valid class index
                            return digit_val, is_correct
                    except (ValueError, TypeError):
                        continue
                
            # If we know the answer is correct but couldn't extract it, use ground truth
            if ground_truth and 'label' in ground_truth:
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
            prediction = int(int_matches[-1])  # Use the last match
            return prediction, prediction == ground_truth_label
            
        # Try additional patterns for classification responses
        alternative_patterns = [
            r'answer:\s*(\d+)',
            r'class:\s*(\d+)',
            r'prediction:\s*(\d+)',
            r'label:\s*(\d+)',
            r'the answer is\s*(\d+)',
            r'class is\s*(\d+)'
        ]
        
        for pattern in alternative_patterns:
            matches = re.findall(pattern, response_text, re.IGNORECASE)
            if matches:
                try:
                    prediction = int(matches[-1])  # Use the last match
                    return prediction, prediction == ground_truth_label
                except (ValueError, TypeError):
                    continue  # Try next pattern if this one didn't work
        
        # Final fallback: just find any number that could be a valid class
        num_classes = 0
        try:
            data_source = ground_truth.get('data_source', 'blobs')
            num_classes = get_num_classes(data_source)
        except:
            num_classes = 3  # Default for most classification tasks
            
        # Extract all digit sequences and check if any could be a valid class
        digit_matches = re.findall(r'\b(\d+)\b', response_text)
        for match in digit_matches[::-1]:  # Check from last to first
            try:
                digit_val = int(match)
                if 0 <= digit_val < num_classes:
                    # Found a valid class index
                    return digit_val, digit_val == ground_truth_label
            except (ValueError, TypeError):
                continue
        
        # If no integer found, return None
        return None, False


def extract_and_execute_model_functions(extracted_json: str, ground_truth: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract model functions from Claude analysis JSON and execute them on in-context samples
    
    Args:
        extracted_json: JSON string containing model functions
        ground_truth: Ground truth data containing in_context_samples
        
    Returns:
        List of dictionaries with model info and accuracy
    """
    if not extracted_json or not ground_truth or 'in_context_samples' not in ground_truth:
        return []
    
    # Parse JSON if it's a string
    models = []
    try:
        if isinstance(extracted_json, str):
            models = json.loads(extracted_json)
        else:
            models = extracted_json
    except Exception as e:
        print(f"Error parsing model JSON: {e}")
        return []
    
    if not isinstance(models, list):
        print(f"Expected list of models, got {type(models)}")
        return []
    
    # Get in-context samples
    in_context_samples = ground_truth.get('in_context_samples', [])
    
    # Check if in_context_samples is valid and convert to list if it's a NumPy array
    try:
        # If it's a NumPy array
        if hasattr(in_context_samples, 'tolist'):
            in_context_samples = in_context_samples.tolist()
        
        # Check if it's empty
        if len(in_context_samples) == 0:
            return []
    except Exception as e:
        print(f"Error processing in_context_samples: {e}")
        return []
    
    # Results to return
    model_results = []
    
    # Process each model
    for model_idx, model in enumerate(models):
        if not isinstance(model, dict):
            continue
        
        # Extract model details
        model_desc = model.get('description', f'Model {model_idx}')
        model_func_str = model.get('function', '')
        
        if not model_func_str:
            continue
        
        # Build a safe execution environment
        model_namespace = {
            'math': math,
            'model_func': None,
            'accuracy': 0.0,
            'correct_count': 0,
            'total_count': 0,
            'predictions': []
        }
        
        # Extract the function name and prepare for execution
        func_name = "model"
        try:
            # Look for the function definition
            func_match = re.search(r'def\s+(\w+)\s*\(', model_func_str)
            if func_match:
                func_name = func_match.group(1)
            
            # Clean up the function code
            # Remove any unsafe operations
            clean_func_str = model_func_str
            
            # Execute the function code in the namespace
            exec(clean_func_str, model_namespace)
            
            # Get the model function
            model_func = model_namespace.get(func_name)
            
            # Skip if the function wasn't properly defined
            if not callable(model_func):
                model_results.append({
                    'model_desc': model_desc,
                    'model_func': model_func_str,
                    'accuracy': 0.0,
                    'error': f"Function '{func_name}' not callable"
                })
                continue
            
            # Evaluate on in-context samples
            correct_count = 0
            predictions = []
            
            for sample in in_context_samples:
                if not isinstance(sample, dict):
                    if hasattr(sample, 'item') and callable(getattr(sample, 'item')):
                        # Try to convert NumPy item to Python native type
                        try:
                            sample = sample.item()
                        except:
                            continue
                    else:
                        continue
                
                sample_features = sample.get('features', [])
                # Convert to list if it's a NumPy array
                if hasattr(sample_features, 'tolist'):
                    sample_features = sample_features.tolist()
                
                if len(sample_features) < 2:
                    continue
                    
                true_label = sample.get('label')
                if true_label is None:
                    continue
                
                # Convert to Python native type if it's a NumPy scalar
                if hasattr(true_label, 'item') and callable(getattr(true_label, 'item')):
                    true_label = true_label.item()
                
                # Try to call the model function with the sample features
                try:
                    # Most functions expect x, y as separate args
                    x, y = sample_features[0], sample_features[1]
                    #element in the list is a tuple with (x, y, label)
                    all_samples = [(each.get('features')[0], each.get('features')[1], each.get('label')) for each in in_context_samples]
                    pred = model_func(x, y, all_samples)
                    
                    # Ensure prediction is an integer
                    if pred is not None:
                        pred = int(pred)
                        
                    is_correct = pred == true_label
                    if is_correct:
                        correct_count += 1
                        
                    predictions.append({
                        'features': sample_features,
                        'true_label': true_label,
                        'predicted': pred,
                        'correct': is_correct
                    })
                except Exception as e:
                    # If the function fails, try alternate argument patterns
                    try:
                        # Try calling with features as a single arg
                        all_samples = [(each.get('features')[0], each.get('features')[1], each.get('label')) for each in in_context_samples]
                        pred = model_func(all_samples)
                        
                        # Ensure prediction is an integer
                        if pred is not None:
                            pred = int(pred)
                            
                        is_correct = pred == true_label
                        if is_correct:
                            correct_count += 1
                            
                        predictions.append({
                            'features': sample_features,
                            'true_label': true_label,
                            'predicted': pred,
                            'correct': is_correct
                        })
                    except Exception as e2:
                        # If all attempts fail, log the error
                        predictions.append({
                            'features': sample_features,
                            'true_label': true_label,
                            'predicted': None,
                            'correct': False,
                            'error': str(e2)
                        })
            
            # Calculate accuracy
            total_count = len(in_context_samples)
            accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0
            
            # Add result
            model_results.append({
                'model_desc': model_desc,
                'model_func': model_func_str,
                'accuracy': accuracy,
                'correct_count': correct_count,
                'total_count': total_count,
                'predictions': predictions
            })
            
        except Exception as e:
            # If there's any error in processing the model
            model_results.append({
                'model_desc': model_desc,
                'model_func': model_func_str,
                'accuracy': 0.0,
                'error': str(e)
            })
    
    return model_results


def visualize_icl_reasoning_output(input_file: str, output_format: str = "html", save_dir: Optional[str] = None, max_samples: int = 100, save_json: bool = True):
    """
    Visualize ICL reasoning output from parquet files with model responses
    
    Args:
        input_file: Path to the input parquet file
        output_format: Output format, supports txt and html
        save_dir: Directory to save the output file (default: same as input file's directory)
        max_samples: Maximum number of samples to visualize (default: 100)
        save_json: Whether to save a JSON file with all visualization data (default: True)
        
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
            prediction, is_correct = get_prediction_result(cleaned_response_text, ground_truth)
            can_parse_prediction = prediction is not None
        
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
    
    # Save to JSON file if requested
    if save_json:
        # Prepare JSON data
        json_data = {
            "metadata": {
                "input_file": str(input_file),
                "timestamp": pd.Timestamp.now().isoformat(),
                "total_samples": total_data_size,
                "displayed_samples": displayed_samples,
                "accuracy": accuracy,
                "refined_accuracy": refined_accuracy,
                "parseable_accuracy": parseable_accuracy,
                "parseable_predictions": parseable_predictions,
                "unparseable_predictions": unparseable_predictions
            },
            "samples": []
        }
        
        # Add all samples that will be displayed
        for idx, row in display_df.iterrows():
            # Get ground truth label
            ground_truth = None
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
            
            # Clean response text
            cleaned_response_text = clean_response_text(response_text, input_prompt_content)
            
            # Remove prompt content from response
            if input_prompt_content:
                cleaned_response_text = cleaned_response_text.replace(input_prompt_content, "")
            
            # Get data source for reward function selection
            data_source = row.get('data_source', 'blobs')
            
            # Get reward function for the data source
            reward_fn = select_reward_fn(data_source)
            
            # Extract thinking and answer
            raw_thinking = extract_think_content(cleaned_response_text)
            raw_answer = extract_answer_content(cleaned_response_text)
            
            # Evaluate with the appropriate reward function
            is_correct = False
            prediction = None
            
            try:
                # First try with the dedicated reward function
                is_correct = reward_fn(cleaned_response_text, ground_truth)
                
                # Extract prediction for display with enhanced matching
                prediction = None
                # First check if there's a clean answer tag
                if raw_answer and raw_answer.strip().isdigit():
                    prediction = int(raw_answer.strip())
                else:
                    # Try to extract prediction using get_prediction_result function
                    parsed_prediction, _ = get_prediction_result(cleaned_response_text, ground_truth)
                    if parsed_prediction is not None:
                        prediction = parsed_prediction
                    elif ground_truth and 'label' in ground_truth and is_correct:
                        # If correct but can't parse prediction, use ground truth
                        prediction = ground_truth['label']
            
            except Exception:
                # Fallback to simple prediction extraction
                prediction, is_correct = get_prediction_result(cleaned_response_text, ground_truth)
                
            # Create sample data
            sample_data = {
                "index": int(idx),
                "data_source": data_source,
                "ground_truth": ground_truth,
                "prediction": prediction,
                "is_correct": is_correct,
                "cleaned_response": cleaned_response_text,
                "raw_thinking": raw_thinking,
                "raw_answer": raw_answer
            }
            
            # 使用helper函数添加额外的数据字段
            sample_data = add_sample_data_fields(sample_data, row)
            
            # Add to samples list
            json_data["samples"].append(sample_data)
            
            # Add additional data fields
            sample_data = add_sample_data_fields(sample_data, row)
        
        # Save to JSON file
        json_output_file = output_dir / f"{Path(input_file).stem}_visualization_data.json"
        try:
            with open(json_output_file, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, default=str)
            print(f"JSON data saved to: {json_output_file}")
        except Exception as e:
            print(f"Error saving JSON data: {e}")
    
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
                
                # Extract prediction for display with enhanced matching
                prediction = None
                # First check if there's a clean answer tag
                if raw_answer and raw_answer.strip().isdigit():
                    prediction = int(raw_answer.strip())
                else:
                    # Try to extract prediction using get_prediction_result function
                    parsed_prediction, _ = get_prediction_result(cleaned_response_text, ground_truth)
                    if parsed_prediction is not None:
                        prediction = parsed_prediction
                    elif ground_truth and 'label' in ground_truth and is_correct:
                        # If correct but can't parse prediction, use ground truth
                        prediction = ground_truth['label']
                
            except Exception:
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
                
                # 使用辅助函数处理ICL示例
                icl_examples, icl_examples_count = process_icl_examples_for_display(row)
                
                html_content.append(f'<div style="font-size: 0.9em; margin-bottom: 10px;"><b>Total ICL Examples: {icl_examples_count}</b></div>')
                
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
                                # 计算token数量
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
                
                # 使用辅助函数处理测试示例
                test_examples, test_examples_count = process_test_examples_for_display(row)
                
                html_content.append(f'<div style="font-size: 0.9em; margin-bottom: 10px;"><b>Total Test Examples: {test_examples_count}</b></div>')
                
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
            
            # Add Claude raw output section if available
            if 'claude_analysis_raw_output' in row and row['claude_analysis_raw_output'] is not None:
                html_content.append(f'<div class="section">')
                html_content.append(f'<div class="section-title">Claude Analysis Raw Output</div>')
                html_content.append(f'<details>')
                html_content.append(f'<summary>Show Claude Analysis Raw Output</summary>')
                claude_raw = row['claude_analysis_raw_output']
                if not isinstance(claude_raw, str):
                    claude_raw = str(claude_raw)
                # Escape HTML content
                escaped_claude_raw = html.escape(claude_raw)
                html_content.append(f'<div class="response" style="white-space: pre-wrap; font-family: monospace; max-height: 400px; overflow-y: auto;">{escaped_claude_raw}</div>')
                html_content.append(f'</details>')
                html_content.append(f'</div>')
            
            # Add Claude extracted JSON section if available
            if 'claude_analysis_extracted_json' in row and row['claude_analysis_extracted_json'] is not None:
                html_content.append(f'<div class="section">')
                html_content.append(f'<div class="section-title">Claude Analysis Extracted JSON</div>')
                html_content.append(f'<details>')
                html_content.append(f'<summary>Show Claude Analysis Extracted JSON</summary>')
                claude_json = row['claude_analysis_extracted_json']
                if isinstance(claude_json, str):
                    try:
                        # Try to parse and prettify if it's a JSON string
                        parsed_json = json.loads(claude_json)
                        pretty_json = json.dumps(parsed_json, indent=2)
                        claude_json = pretty_json
                    except:
                        pass  # Keep as is if not valid JSON
                elif isinstance(claude_json, (dict, list)):
                    # Convert to pretty JSON string if it's already a dict or list
                    claude_json = json.dumps(claude_json, indent=2)
                else:
                    claude_json = str(claude_json)
                # Escape HTML content
                escaped_claude_json = html.escape(claude_json)
                html_content.append(f'<div class="response" style="white-space: pre-wrap; font-family: monospace; max-height: 400px; overflow-y: auto;">{escaped_claude_json}</div>')
                html_content.append(f'</details>')
                
                # Add Model Function Evaluation section
                if ground_truth is not None and 'in_context_samples' in ground_truth:
                    html_content.append(f'<div class="section">')
                    html_content.append(f'<div class="section-title">Model Function Evaluation</div>')
                    
                    # Extract and evaluate model functions
                    model_results = extract_and_execute_model_functions(claude_json, ground_truth)
                    
                    # Parse the JSON to get the original models data
                    try:
                        if isinstance(claude_json, str):
                            models_data = json.loads(claude_json)
                        else:
                            models_data = claude_json
                        if not isinstance(models_data, list):
                            models_data = []
                    except Exception:
                        models_data = []
                    
                    if model_results:
                        # Add model evaluation results table
                        html_content.append(f'<table style="width: 100%; border-collapse: collapse; margin-top: 10px;">')
                        html_content.append(f'<tr style="background-color: #f2f2f2;"><th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Order</th><th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Model</th><th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Accuracy</th><th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Details</th></tr>')
                        
                        for idx, result in enumerate(model_results):
                            model_desc = result.get('model_desc', f'Model {idx}')
                            accuracy = result.get('accuracy', 0.0)
                            correct_count = result.get('correct_count', 0)
                            total_count = result.get('total_count', 0)
                            has_error = 'error' in result
                            
                            # Get order directly from the original model object
                            model_order = "ERROR: No order"
                            original_model = None
                            
                            # Find the original model in models_data list
                            for model in models_data:
                                if isinstance(model, dict) and model.get('description') == model_desc:
                                    original_model = model
                                    break
                            
                            # Get the order from original model
                            if original_model and 'order' in original_model:
                                model_order = original_model['order']
                            
                            # Determine row color based on accuracy
                            row_class = ''
                            if has_error:
                                row_class = 'style="background-color: #ffeeee;"'
                            elif accuracy >= 90:
                                row_class = 'style="background-color: #eeffee;"'
                            elif accuracy >= 75:
                                row_class = 'style="background-color: #ffffee;"'
                                
                            # Create table row
                            html_content.append(f'<tr {row_class}>')
                            html_content.append(f'<td style="border: 1px solid #ddd; padding: 8px; text-align: center;">{model_order}</td>')
                            html_content.append(f'<td style="border: 1px solid #ddd; padding: 8px;">{model_desc}</td>')
                            
                            if has_error:
                                html_content.append(f'<td style="border: 1px solid #ddd; padding: 8px; color: red;">Error</td>')
                                html_content.append(f'<td style="border: 1px solid #ddd; padding: 8px; color: red;">{result.get("error", "Unknown error")}</td>')
                            else:
                                html_content.append(f'<td style="border: 1px solid #ddd; padding: 8px;">{accuracy:.2f}%</td>')
                                html_content.append(f'<td style="border: 1px solid #ddd; padding: 8px;">{correct_count} of {total_count} correct</td>')
                            
                            html_content.append(f'</tr>')
                        
                        html_content.append(f'</table>')
                        
                        # Add collapsible details with prediction results
                        html_content.append(f'<details>')
                        html_content.append(f'<summary style="margin-top: 10px; cursor: pointer; color: #0066cc;">Show Detailed Prediction Results</summary>')
                        
                        for idx, result in enumerate(model_results):
                            if 'error' in result:
                                continue  # Skip models with errors
                                
                            model_desc = result.get('model_desc', f'Model {idx}')
                            model_type = result.get('model_type', 'unknown')
                            predictions = result.get('predictions', [])
                            
                            if not predictions:
                                continue
                                
                            html_content.append(f'<h4 style="margin-top: 15px; margin-bottom: 5px;">{model_desc} ({model_type})</h4>')
                            html_content.append(f'<div style="margin-bottom: 15px;">')
                            html_content.append(f'<div style="font-family: monospace; white-space: pre-wrap; background-color: #f5f5f5; padding: 8px; margin-bottom: 10px;">{html.escape(result.get("model_func", ""))}</div>')
                            
                            # Display first 10 predictions in a table
                            html_content.append(f'<table style="width: 100%; border-collapse: collapse;">')
                            html_content.append(f'<tr style="background-color: #f2f2f2;"><th style="border: 1px solid #ddd; padding: 6px; text-align: left;">Features</th><th style="border: 1px solid #ddd; padding: 6px; text-align: left;">True Label</th><th style="border: 1px solid #ddd; padding: 6px; text-align: left;">Predicted</th><th style="border: 1px solid #ddd; padding: 6px; text-align: left;">Result</th></tr>')
                            
                            for pred_idx, pred in enumerate(predictions[:10]):  # Limit to first 10
                                features = pred.get('features', [])
                                true_label = pred.get('true_label')
                                predicted = pred.get('predicted')
                                is_correct = pred.get('correct', False)
                                
                                features_str = "[" + ", ".join([f"{x:.3f}" for x in features]) + "]"
                                result_class = 'style="color: green; font-weight: bold;"' if is_correct else 'style="color: red;"'
                                result_text = "✓ CORRECT" if is_correct else "✗ WRONG"
                                
                                html_content.append(f'<tr>')
                                html_content.append(f'<td style="border: 1px solid #ddd; padding: 6px; font-family: monospace;">{features_str}</td>')
                                html_content.append(f'<td style="border: 1px solid #ddd; padding: 6px; text-align: center;">{true_label}</td>')
                                html_content.append(f'<td style="border: 1px solid #ddd; padding: 6px; text-align: center;">{predicted if predicted is not None else "ERROR"}</td>')
                                html_content.append(f'<td style="border: 1px solid #ddd; padding: 6px;" {result_class}>{result_text}</td>')
                                html_content.append(f'</tr>')
                            
                            html_content.append(f'</table>')
                            
                            # Show pagination message if more than 10 predictions
                            if len(predictions) > 10:
                                html_content.append(f'<div style="text-align: center; margin-top: 5px; color: #666;">Showing 10 of {len(predictions)} predictions</div>')
                            
                            html_content.append(f'</div>')
                        
                        html_content.append(f'</details>')
                    else:
                        html_content.append(f'<div>No valid model functions found or no in-context samples available for evaluation.</div>')
                    
                    html_content.append(f'</div>')
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
        raise NotImplementedError("Text output is not implemented yet")
        
    
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
    parser.add_argument('--save-json', action='store_true', default=True,
                        help='Save a JSON file with visualization data (default: True)')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: File '{args.input}' not found")
        return
    
    # Visualize the output
    output_file = visualize_icl_reasoning_output(args.input, args.format, args.output_dir, args.max_samples, args.save_json)
    print(f"Visualization complete! Saved to: {output_file}")


if __name__ == "__main__":
    main() 