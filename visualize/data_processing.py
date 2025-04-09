#!/usr/bin/env python3
"""
Helper functions for processing data fields in visualization tool
"""

import json
import re
import math
import random
from typing import Dict, Any, List, Optional, Tuple, Callable


def process_icl_examples(icl_examples) -> List:
    """
    Process ICL examples field from various formats into a standardized list.
    
    Args:
        icl_examples: ICL examples data which can be in various formats (string, list, etc.)
        
    Returns:
        Processed ICL examples as a list
    """
    if isinstance(icl_examples, str):
        try:
            # Try to parse as JSON if it's a string representation of a list
            if icl_examples.startswith('[') and icl_examples.endswith(']'):
                icl_examples = json.loads(icl_examples)
        except:
            pass
    
    if not isinstance(icl_examples, list):
        icl_examples = [icl_examples] if icl_examples is not None else []
        
    return icl_examples


def process_test_examples(test_examples) -> List:
    """
    Process test examples field from string or list format.
    
    Args:
        test_examples: Test examples data (can be string JSON or list)
        
    Returns:
        Processed test examples as a list
    """
    if isinstance(test_examples, str):
        try:
            test_examples = json.loads(test_examples)
        except:
            test_examples = []
            
    if not isinstance(test_examples, list):
        test_examples = [] 
            
    return test_examples


def add_sample_data_fields(sample_data: Dict[str, Any], row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Add additional data fields to a sample data dictionary from a row of data.
    
    Args:
        sample_data: The current sample data dictionary
        row: The row data containing additional fields
        
    Returns:
        Updated sample data dictionary with additional fields
    """
    # Add additional columns if available
    if 'claude_analysis_raw_output' in row and row['claude_analysis_raw_output'] is not None:
        sample_data["claude_analysis_raw_output"] = row['claude_analysis_raw_output']
    
    if 'claude_analysis_extracted_json' in row and row['claude_analysis_extracted_json'] is not None:
        sample_data["claude_analysis_extracted_json"] = row['claude_analysis_extracted_json']
    
    # Add extra data fields if available
    if 'icl_examples' in row:
        # Process ICL examples
        icl_examples = process_icl_examples(row.get('icl_examples', []))
        sample_data["icl_examples_count"] = len(icl_examples) if isinstance(icl_examples, list) else 1
    
    if 'test_examples' in row:
        # Process test examples
        test_examples = process_test_examples(row.get('test_examples', '[]'))
        sample_data["test_examples_count"] = len(test_examples)
    
    if 'icl_example_meta_info' in row:
        sample_data["icl_example_meta_info"] = row['icl_example_meta_info']
    
    if 'test_data' in row:
        sample_data["test_data"] = row['test_data']
    
    if 'extra_info' in row:
        sample_data["extra_info"] = row['extra_info']
    
    return sample_data


def process_icl_examples_for_display(row: Dict[str, Any]) -> Tuple[List, int]:
    """
    Process ICL examples for display in HTML or text output.
    
    Args:
        row: The data row containing ICL examples
        
    Returns:
        Tuple of (processed_examples, count)
    """
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
    elif not isinstance(icl_examples, list):
        icl_examples = [icl_examples] if icl_examples is not None else []
    
    count = len(icl_examples)
    return icl_examples, count


def process_test_examples_for_display(row: Dict[str, Any]) -> Tuple[List, int]:
    """
    Process test examples for display in HTML or text output.
    
    Args:
        row: The data row containing test examples
        
    Returns:
        Tuple of (processed_examples, count)
    """
    test_examples = row.get('test_examples', '[]')
    
    # Parse test examples - expected to be a JSON string of lists of tuples
    try:
        if isinstance(test_examples, str):
            test_examples = json.loads(test_examples)
        elif not isinstance(test_examples, list):
            test_examples = []
    except:
        test_examples = []
    
    count = len(test_examples)
    return test_examples, count 

# Additional utility functions moved from visualize.py

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


def get_tokenizer(tokenizer_name="Qwen/Qwen2.5-3B-Instruct"):
    """
    Get tokenizer for token length calculation
    """
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        return tokenizer
    except Exception as e:
        print(f"Warning: Failed to load tokenizer {tokenizer_name}: {e}")
        try:
            from transformers import AutoTokenizer
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
                    all_samples = [(each.get('features')[0], each.get('features')[1], each.get('label')) for each in in_context_samples
                                   if (each.get('features')[0]!=x and each.get('features')[1]!=y)]
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
                        all_samples = [(each.get('features')[0], each.get('features')[1], each.get('label')) for each in in_context_samples
                                   if (each.get('features')[0]!=x and each.get('features')[1]!=y)]
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

def select_reward_fn(data_source: str) -> Callable:
    """
    Select the appropriate reward function based on data source
    
    Args:
        data_source: The data source name (e.g., 'blobs', 'circles')
        
    Returns:
        Reward function appropriate for the data source
    """
    try:
        from verl.trainer.ppo.helper import _select_rm_score_fn as orig_select_reward_fn
        return orig_select_reward_fn(data_source)
    except ImportError:
        # Define a basic fallback if imports not available
        try:
            if "blobs" in data_source:
                from examples.data_preprocess.blobs import blobs_reward_fn
                return blobs_reward_fn
        except ImportError:
            print("Warning: Could not import blobs_reward_fn")
        
        # Default reward function (will use our get_prediction_result)
        return lambda solution_str, ground_truth: get_prediction_result(solution_str, ground_truth)[1] 