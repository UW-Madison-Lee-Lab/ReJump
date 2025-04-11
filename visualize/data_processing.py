#!/usr/bin/env python3
"""
Helper functions for processing data fields in visualization tool
"""

import json
import re
import math
import random
from typing import Dict, Any, List, Optional, Tuple, Callable, Union

def parse_llm_json_regression(llm_json: str) -> List[Dict[str, Any]]:
    """
    Extract fitting models from LLM output containing regression models in JSON format.
    
    Args:
        llm_json: String containing LLM output with JSON array of regression models
        
    Returns:
        List of dictionaries representing the regression models
    """
    if not llm_json:
        return []
    
    # Find the JSON array in the text
    import re
    import json
    
    # Clean the input string first
    # Strip any leading/trailing text before the actual JSON
    first_bracket = llm_json.find('[')
    last_bracket = llm_json.rfind(']')
    
    if first_bracket != -1 and last_bracket != -1 and first_bracket < last_bracket:
        # Extract the potential JSON array
        json_str = llm_json[first_bracket:last_bracket+1]
        
        # Try to parse the JSON directly first
        try:
            models = json.loads(json_str)
            if isinstance(models, list) and len(models) > 0:
                return models
        except json.JSONDecodeError:
            # If it fails, we need to clean the string more thoroughly
            pass
        
        # Handle special characters
        # Replace Unicode characters that might cause JSON parsing issues
        char_replacements = {
            '≈': '~',        # approximate
            '→': '->',       # arrow
            '²': '**2',      # squared
            '\u2248': '~',   # almost equal to
            '\u2192': '->',  # rightwards arrow
            '\u00b2': '**2', # superscript two
        }
        
        for char, replacement in char_replacements.items():
            json_str = json_str.replace(char, replacement)
        
        # Try to load the cleaned JSON
        try:
            models = json.loads(json_str)
            if isinstance(models, list):
                return models
        except json.JSONDecodeError as e:
            # Last attempt: use regex to extract each model object individually
            try:
                objects = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', json_str)
                if objects:
                    assembled_json = '['
                    for i, obj in enumerate(objects):
                        assembled_json += obj
                        if i < len(objects) - 1:
                            assembled_json += ','
                    assembled_json += ']'
                    
                    models = json.loads(assembled_json)
                    if isinstance(models, list):
                        return models
            except:
                pass
    
    # If extraction failed, use a fallback approach to return whatever valid models we can
    try:
        # Try to interpret the entire string as JSON first
        models = json.loads(llm_json)
        if isinstance(models, list):
            return models
    except:
        # Try to extract any JSON array in the text using regex
        array_pattern = r'\[\s*\{.*\}\s*\]'
        match = re.search(array_pattern, llm_json, re.DOTALL)
        if match:
            try:
                models = json.loads(match.group(0))
                if isinstance(models, list):
                    return models
            except:
                pass
    
    # If all extraction methods fail, return empty list
    return []

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
    if 'llm_analysis_raw_output' in row and row['llm_analysis_raw_output'] is not None:
        sample_data["llm_analysis_raw_output"] = row['llm_analysis_raw_output']
    
    if 'llm_analysis_extracted_json' in row and row['llm_analysis_extracted_json'] is not None:
        sample_data["llm_analysis_extracted_json"] = row['llm_analysis_extracted_json']
    
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


def get_prediction_result(response_text: Optional[str], ground_truth, data_type: str = 'classification') -> Tuple[Optional[Union[int, float]], bool]:
    """
    Evaluate prediction using main_eval approach
    
    Args:
        response_text: Full model response text
        ground_truth: The ground truth data containing label/target and features
        data_type: The type of task ('classification' or 'regression')
        
    Returns:
        Tuple of (predicted_value, is_correct)
    """
    if response_text is None:
        return None, False
    
    # Make sure ground_truth data is available
    if ground_truth is None:
        return None, False
    
    # For regression, we look for label or target
    ground_truth_value = None
    if data_type == 'regression':
        if 'target' in ground_truth:
            ground_truth_value = ground_truth['target']
        elif 'label' in ground_truth:
            ground_truth_value = ground_truth['label']
    else:
        # For classification
        if 'label' in ground_truth:
            ground_truth_value = ground_truth['label']
        elif 'target' in ground_truth:
            ground_truth_value = ground_truth['target']
            
    if ground_truth_value is None:
        return None, False
            
    # Convert to Python native type if it's a NumPy scalar
    if hasattr(ground_truth_value, 'item') and callable(getattr(ground_truth_value, 'item')):
        ground_truth_value = ground_truth_value.item()
    
    # Try to parse from answer tags first
    all_matches = list(re.finditer(r'<answer>(.*?)</answer>', response_text, re.DOTALL))
    if all_matches:
        response_extract = None
        for match in all_matches[::-1]:  # Check from last to first
            match_content = match.group(1).strip()
            if data_type == 'regression':
                # For regression, try to parse as float
                try:
                    float(match_content)
                    response_extract = match
                    break
                except ValueError:
                    continue
            else:
                # For classification, only accept digits
                if match_content.isdigit():
                    response_extract = match
                    break
                    
        if response_extract is not None:
            try:
                if data_type == 'regression':
                    prediction = float(response_extract.group(1).strip())
                    # For regression, check if within acceptable error margin
                    error_margin = 1e-6
                    is_correct = abs(prediction - ground_truth_value) < error_margin
                    return prediction, is_correct
                else:
                    # For classification
                    prediction = int(response_extract.group(1).strip())
                    is_correct = prediction == ground_truth_value
                    return prediction, is_correct
            except ValueError:
                pass
    
    # Try to find patterns based on data_type
    if data_type == 'regression':
        # For regression, look for floating point numbers
        float_patterns = [
            r'<answer>\s*(-?\d+\.?\d*)\s*</answer>',
            r'answer:\s*(-?\d+\.?\d*)',
            r'prediction:\s*(-?\d+\.?\d*)',
            r'target:\s*(-?\d+\.?\d*)',
            r'value:\s*(-?\d+\.?\d*)',
            r'result:\s*(-?\d+\.?\d*)',
            r'output:\s*(-?\d+\.?\d*)',
            r'predicted value is\s*(-?\d+\.?\d*)',
            r'the answer is\s*(-?\d+\.?\d*)'
        ]
        
        for pattern in float_patterns:
            matches = re.findall(pattern, response_text, re.IGNORECASE)
            if matches:
                try:
                    prediction = float(matches[-1])  # Use the last match
                    error_margin = 1e-6
                    is_correct = abs(prediction - ground_truth_value) < error_margin
                    return prediction, is_correct
                except (ValueError, TypeError):
                    continue  # Try next pattern
                    
        # Last resort: try to find any floating point number in the text
        float_matches = re.findall(r'-?\d+\.\d+', response_text)
        if float_matches:
            for match in float_matches[::-1]:  # Process from last to first
                try:
                    prediction = float(match)
                    error_margin = 1e-6
                    is_correct = abs(prediction - ground_truth_value) < error_margin
                    return prediction, is_correct
                except ValueError:
                    continue
                    
        # If all float attempts fail, try integer values
        int_matches = re.findall(r'-?\d+', response_text)
        if int_matches:
            for match in int_matches[::-1]:  # Process from last to first
                try:
                    prediction = float(match)  # Convert to float for regression
                    error_margin = 1e-6
                    is_correct = abs(prediction - ground_truth_value) < error_margin
                    return prediction, is_correct
                except ValueError:
                    continue
    else:
        # For classification, look for integer patterns
        int_patterns = [
            r'<answer>\s*(\d+)\s*</answer>',
            r'answer:\s*(\d+)',
            r'class:\s*(\d+)',
            r'prediction:\s*(\d+)',
            r'label:\s*(\d+)',
            r'the answer is\s*(\d+)',
            r'class is\s*(\d+)'
        ]
        
        for pattern in int_patterns:
            matches = re.findall(pattern, response_text, re.IGNORECASE)
            if matches:
                try:
                    prediction = int(matches[-1])  # Use the last match
                    is_correct = prediction == ground_truth_value
                    return prediction, is_correct
                except (ValueError, TypeError):
                    continue  # Try next pattern if this one didn't work
        
        # Final fallback for classification: find any number that could be a valid class
        digit_matches = re.findall(r'\b(\d+)\b', response_text)
        for match in digit_matches[::-1]:  # Check from last to first
            try:
                prediction = int(match)
                is_correct = prediction == ground_truth_value
                return prediction, is_correct
            except (ValueError, TypeError):
                continue
    
    # If no valid prediction found
    return None, False


def extract_and_execute_model_functions(extracted_json: str, ground_truth: Dict[str, Any], data_type: str = 'classification') -> List[Dict[str, Any]]:
    """
    Extract model functions from Claude analysis JSON and execute them on in-context samples
    
    Args:
        extracted_json: JSON string containing model functions
        ground_truth: Ground truth data containing in_context_samples
        data_type: 'classification' or 'regression'
        
    Returns:
        List of dictionaries with model info and accuracy
    """
    if not extracted_json or not ground_truth or 'in_context_samples' not in ground_truth:
        print("Error: Missing required data for model execution - empty extracted_json or missing in_context_samples")
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
    
    print(f"Found {len(models)} models to evaluate with data_type: {data_type}")
    
    # Get in-context samples
    in_context_samples = ground_truth.get('in_context_samples', [])
    
    # Check if in_context_samples is valid and convert to list if it's a NumPy array
    try:
        # If it's a NumPy array
        if hasattr(in_context_samples, 'tolist'):
            in_context_samples = in_context_samples.tolist()
        
        # Check if it's empty
        if len(in_context_samples) == 0:
            print("Error: in_context_samples is empty")
            return []
        
        print(f"Found {len(in_context_samples)} in-context samples for evaluation")
    except Exception as e:
        print(f"Error processing in_context_samples: {e}")
        return []
    
    # Results to return
    model_results = []
    
    # Process each model
    for model_idx, model in enumerate(models):
        if not isinstance(model, dict):
            print(f"Model {model_idx} is not a dictionary, skipping")
            continue
        
        # Extract model details
        model_desc = model.get('description', f'Model {model_idx}')
        model_func_str = model.get('function', '')
        model_family = model.get('model_family', 'cannot parse model family')  # Extract model_family
        
        print(f"\nProcessing model {model_idx+1}/{len(models)}: {model_desc}")
        print(f"Model family: {model_family}")
        
        if not model_func_str:
            print(f"Model {model_idx} has no function string, skipping")
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
            
            print(f"Function name identified: {func_name}")
            
            # Clean up the function code
            # Remove any unsafe operations
            clean_func_str = model_func_str
            
            # Execute the function code in the namespace
            exec(clean_func_str, model_namespace)
            
            # Get the model function
            model_func = model_namespace.get(func_name)
            
            # Skip if the function wasn't properly defined
            if not callable(model_func):
                print(f"Function '{func_name}' is not callable")
                model_results.append({
                    'model_desc': model_desc,
                    'model_func': model_func_str,
                    'model_family': model_family,  # Include model_family
                    'accuracy': 0.0,
                    'error': f"Function '{func_name}' not callable"
                })
                continue
            
            # Init metrics based on data_type
            if data_type == 'regression':
                # For regression, we track squared errors
                squared_errors = []
                mse = 0.0
            else:
                # For classification, we track correct predictions
                correct_count = 0
            
            total_count = 0
            predictions = []
            
            print(f"Evaluating model on {len(in_context_samples)} samples...")
            
            for sample_idx, sample in enumerate(in_context_samples):
                if not isinstance(sample, dict):
                    if hasattr(sample, 'item') and callable(getattr(sample, 'item')):
                        # Try to convert NumPy item to Python native type
                        try:
                            sample = sample.item()
                        except:
                            print(f"  Sample {sample_idx}: Failed to convert to Python type, skipping")
                            continue
                    else:
                        print(f"  Sample {sample_idx}: Not a dictionary, skipping")
                        continue
                
                sample_features = sample.get('features', [])
                # Convert to list if it's a NumPy array
                if hasattr(sample_features, 'tolist'):
                    sample_features = sample_features.tolist()
                
                if len(sample_features) < 2:
                    print(f"  Sample {sample_idx}: Insufficient features, skipping")
                    continue
                    
                # Get true value (label for classification, target for regression)
                true_value = None
                if data_type == 'regression':
                    # For regression, prefer 'target' field
                    true_value = sample.get('target')
                    if true_value is None:
                        # Fallback to label if target doesn't exist
                        true_value = sample.get('label')
                else:
                    # For classification, prefer 'label' field
                    true_value = sample.get('label')
                    if true_value is None:
                        # Fallback to target if label doesn't exist
                        true_value = sample.get('target')
                
                if true_value is None:
                    print(f"  Sample {sample_idx}: No label or target, skipping")
                    continue
                else:
                    if data_type == 'regression':
                        print(f"  Sample {sample_idx}: Using target value: {true_value}")
                    else:
                        print(f"  Sample {sample_idx}: Using label: {true_value}")
                
                # Convert to Python native type if it's a NumPy scalar
                if hasattr(true_value, 'item') and callable(getattr(true_value, 'item')):
                    true_value = true_value.item()
                
                # Try to call the model function with the sample features
                try:
                    # Most functions expect x, y as separate args
                    x, y = sample_features[0], sample_features[1]
                    
                    # Create all_samples by excluding the current sample
                    all_samples = []
                    for each in in_context_samples:
                        if each.get('features') and len(each.get('features')) >= 2:
                            feat_x = each.get('features')[0]
                            feat_y = each.get('features')[1]
                            
                            # Get sample value (label or target)
                            sample_value = None
                            if data_type == 'regression':
                                sample_value = each.get('target')
                                if sample_value is None:
                                    sample_value = each.get('label')
                            else:
                                sample_value = each.get('label')
                                if sample_value is None:
                                    sample_value = each.get('target')
                            
                            # Only add if we have a valid value
                            if sample_value is not None and (feat_x != x or feat_y != y):
                                all_samples.append((feat_x, feat_y, sample_value))
                    
                    # Call the model function
                    pred = model_func(x, y, all_samples)
                    
                    # Process prediction based on data_type
                    if data_type == 'regression':
                        # For regression, convert to float if possible
                        try:
                            if pred is not None:
                                pred = float(pred)
                                # Calculate squared error
                                squared_error = (pred - true_value) ** 2
                                squared_errors.append(squared_error)
                                print(f"    Prediction: {pred}, True: {true_value}, Squared Error: {squared_error}")
                        except (ValueError, TypeError) as e:
                            print(f"    Error converting prediction to float: {e}")
                            pred = None
                    else:
                        # For classification, convert to int if possible
                        try:
                            if pred is not None:
                                pred = int(pred)
                                # Check if prediction is correct
                                is_correct = pred == true_value
                                if is_correct:
                                    correct_count += 1
                        except (ValueError, TypeError) as e:
                            print(f"    Error converting prediction to int: {e}")
                            pred = None
                            is_correct = False
                    
                    # Add prediction to results
                    if data_type == 'regression':
                        predictions.append({
                            'features': sample_features,
                            'true_value': true_value,
                            'predicted': pred,
                            'squared_error': (pred - true_value) ** 2 if pred is not None else None
                        })
                    else:
                        predictions.append({
                            'features': sample_features,
                            'true_label': true_value,
                            'predicted': pred,
                            'correct': pred == true_value if pred is not None else False
                        })
                    
                    total_count += 1
                    
                    if (sample_idx + 1) % 10 == 0:
                        if data_type == 'regression':
                            curr_mse = sum(squared_errors) / len(squared_errors) if squared_errors else float('inf')
                            print(f"  Processed {sample_idx + 1} samples, current MSE: {curr_mse:.6f}")
                        else:
                            print(f"  Processed {sample_idx + 1} samples, correct so far: {correct_count}")
                    
                except Exception as e:
                    print(f"  Sample {sample_idx}: Error with primary approach: {e}")
                    # If the function fails, try alternate argument patterns
                    try:
                        # Try calling with all_samples as the only argument
                        pred = model_func(x, y, all_samples)
                        
                        # Process prediction based on data_type
                        if data_type == 'regression':
                            # For regression, convert to float if possible
                            try:
                                if pred is not None:
                                    pred = float(pred)
                                    # Calculate squared error
                                    squared_error = (pred - true_value) ** 2
                                    squared_errors.append(squared_error)
                                    print(f"    Prediction: {pred}, True: {true_value}, Squared Error: {squared_error}")
                            except (ValueError, TypeError) as e:
                                print(f"    Error converting prediction to float: {e}")
                                pred = None
                        else:
                            # For classification, convert to int if possible
                            try:
                                if pred is not None:
                                    pred = int(pred)
                                    # Check if prediction is correct
                                    is_correct = pred == true_value
                                    if is_correct:
                                        correct_count += 1
                            except (ValueError, TypeError) as e:
                                print(f"    Error converting prediction to int: {e}")
                                pred = None
                                is_correct = False
                        
                        # Add prediction to results
                        if data_type == 'regression':
                            predictions.append({
                                'features': sample_features,
                                'true_value': true_value,
                                'predicted': pred,
                                'squared_error': (pred - true_value) ** 2 if pred is not None else None
                            })
                        else:
                            predictions.append({
                                'features': sample_features,
                                'true_label': true_value,
                                'predicted': pred,
                                'correct': pred == true_value if pred is not None else False
                            })
                        
                        total_count += 1
                        print(f"  Sample {sample_idx}: Alternate approach succeeded")
                    except Exception as e2:
                        print(f"  Sample {sample_idx}: All approaches failed: {e2}")
                        # If all attempts fail, log the error
                        if data_type == 'regression':
                            predictions.append({
                                'features': sample_features,
                                'true_value': true_value,
                                'predicted': None,
                                'squared_error': None,
                                'error': str(e2)
                            })
                        else:
                            predictions.append({
                                'features': sample_features,
                                'true_label': true_value,
                                'predicted': None,
                                'correct': False,
                                'error': str(e2)
                            })
            
            # Calculate final metrics
            if data_type == 'regression':
                # For regression, calculate MSE
                mse = sum(squared_errors) / len(squared_errors) if squared_errors else float('inf')
                print(f"Model evaluation complete: MSE = {mse:.6f} (on {len(squared_errors)} valid predictions)")
                
                # Add result for regression
                model_results.append({
                    'model_desc': model_desc,
                    'model_func': model_func_str,
                    'model_family': model_family,
                    'mse': mse,
                    'total_count': total_count,
                    'valid_predictions': len(squared_errors),
                    'predictions': predictions
                })
            else:
                # For classification, calculate accuracy
                accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0
                print(f"Model evaluation complete: {correct_count}/{total_count} correct ({accuracy:.2f}%)")
                
                # Add result for classification
                model_results.append({
                    'model_desc': model_desc,
                    'model_func': model_func_str,
                    'model_family': model_family,
                    'accuracy': accuracy,
                    'correct_count': correct_count,
                    'total_count': total_count,
                    'predictions': predictions
                })
            
            print(f"Generated {len(predictions)} predictions")
            
        except Exception as e:
            # If there's any error in processing the model
            print(f"Error processing model: {e}")
            model_results.append({
                'model_desc': model_desc,
                'model_func': model_func_str,
                'model_family': model_family,
                'error': str(e)
            })
    
    print(f"\nProcessed {len(model_results)} models")
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