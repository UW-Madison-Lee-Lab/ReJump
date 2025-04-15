#!/usr/bin/env python3
"""
Helper functions for processing data fields in visualization tool
"""

import json
import re
import math
import random
import numpy as np  # 确保导入numpy
from typing import Dict, Any, List, Optional, Tuple, Callable, Union

import pandas as pd
def get_ground_truth(row: pd.Series) -> Optional[Dict]:
    """Extract ground truth from row data."""
    if 'reward_model' not in row or not isinstance(row['reward_model'], dict):
        return None
        
    ground_truth_data = row['reward_model'].get('ground_truth')
    if not isinstance(ground_truth_data, dict):
        return None
        
    if 'label' in ground_truth_data:
        return ground_truth_data
    elif 'target' in ground_truth_data:
        # If there's no label but there's a target, use the target as label
        ground_truth = ground_truth_data.copy()
        ground_truth['label'] = ground_truth_data['target']
        return ground_truth

    return None
            
def get_response_text(row: pd.Series) -> str:
    """Extract response text from row data."""
    responses = row.get('responses', [])
    
    if responses is None:
        return "No response available"
    
    if not isinstance(responses, list):
        try:
            responses = list(responses)
        except (TypeError, ValueError):
            responses = [responses]
    
    # Get first response (assuming single response per example)
    response_text = responses[0] if responses and len(responses) > 0 else "No response generated"
    
    # Ensure response_text is a string
    if not isinstance(response_text, str):
        if isinstance(response_text, bytes):
            response_text = response_text.decode('utf-8')
        else:
            response_text = str(response_text)
    
    # Filter out <|endoftext|> tags
    return response_text.replace("<|endoftext|>", "")

def get_prompt_content(row: pd.Series) -> Optional[str]:
    """Extract prompt content from row data."""
    input_prompt = row.get('prompt')
    if not input_prompt or not isinstance(input_prompt, list) or not input_prompt:
        return None
        
    if isinstance(input_prompt[0], dict) and 'content' in input_prompt[0]:
        return input_prompt[0]['content']
    
    return None

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
    
    return None, False
    
    ## Try to find patterns based on data_type
    # if data_type == 'regression':
    #     # For regression, look for floating point numbers
    #     float_patterns = [
    #         r'<answer>\s*(-?\d+\.?\d*)\s*</answer>',
    #         r'answer:\s*(-?\d+\.?\d*)',
    #         r'prediction:\s*(-?\d+\.?\d*)',
    #         r'target:\s*(-?\d+\.?\d*)',
    #         r'value:\s*(-?\d+\.?\d*)',
    #         r'result:\s*(-?\d+\.?\d*)',
    #         r'output:\s*(-?\d+\.?\d*)',
    #         r'predicted value is\s*(-?\d+\.?\d*)',
    #         r'the answer is\s*(-?\d+\.?\d*)'
    #     ]
        
    #     for pattern in float_patterns:
    #         matches = re.findall(pattern, response_text, re.IGNORECASE)
    #         if matches:
    #             try:
    #                 prediction = float(matches[-1])  # Use the last match
    #                 error_margin = 1e-6
    #                 is_correct = abs(prediction - ground_truth_value) < error_margin
    #                 return prediction, is_correct
    #             except (ValueError, TypeError):
    #                 continue  # Try next pattern
                    
    #     # Last resort: try to find any floating point number in the text
    #     float_matches = re.findall(r'-?\d+\.\d+', response_text)
    #     if float_matches:
    #         for match in float_matches[::-1]:  # Process from last to first
    #             try:
    #                 prediction = float(match)
    #                 error_margin = 1e-6
    #                 is_correct = abs(prediction - ground_truth_value) < error_margin
    #                 return prediction, is_correct
    #             except ValueError:
    #                 continue
                    
    #     # If all float attempts fail, try integer values
    #     int_matches = re.findall(r'-?\d+', response_text)
    #     if int_matches:
    #         for match in int_matches[::-1]:  # Process from last to first
    #             try:
    #                 prediction = float(match)  # Convert to float for regression
    #                 error_margin = 1e-6
    #                 is_correct = abs(prediction - ground_truth_value) < error_margin
    #                 return prediction, is_correct
    #             except ValueError:
    #                 continue
    # else:
    #     # For classification, look for integer patterns
    #     int_patterns = [
    #         r'<answer>\s*(\d+)\s*</answer>',
    #         r'answer:\s*(\d+)',
    #         r'class:\s*(\d+)',
    #         r'prediction:\s*(\d+)',
    #         r'label:\s*(\d+)',
    #         r'the answer is\s*(\d+)',
    #         r'class is\s*(\d+)'
    #     ]
        
    #     for pattern in int_patterns:
    #         matches = re.findall(pattern, response_text, re.IGNORECASE)
    #         if matches:
    #             try:
    #                 prediction = int(matches[-1])  # Use the last match
    #                 is_correct = prediction == ground_truth_value
    #                 return prediction, is_correct
    #             except (ValueError, TypeError):
    #                 continue  # Try next pattern if this one didn't work
        
    #     # Final fallback for classification: find any number that could be a valid class
    #     digit_matches = re.findall(r'\b(\d+)\b', response_text)
    #     for match in digit_matches[::-1]:  # Check from last to first
    #         try:
    #             prediction = int(match)
    #             is_correct = prediction == ground_truth_value
    #             return prediction, is_correct
    #         except (ValueError, TypeError):
    #             continue
    
    # # If no valid prediction found
    # return None, False


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
    
    # Add安全的比较函数
    def safe_equals(a, b):
        """安全地比较两个可能是NumPy数组的值"""
        if hasattr(a, 'shape') or hasattr(b, 'shape'):
            try:
                return np.array_equal(a, b)
            except:
                # 如果不能直接比较，尝试转换为Python原生类型
                try:
                    if hasattr(a, 'item') and callable(getattr(a, 'item')):
                        a = a.item()
                    if hasattr(b, 'item') and callable(getattr(b, 'item')):
                        b = b.item()
                    return a == b
                except:
                    return False
        return a == b
    
    def safe_not_equals(a, b):
        """安全地比较两个可能是NumPy数组的值是否不相等"""
        return not safe_equals(a, b)
    
    def create_all_samples(in_context_samples, current_x, current_y, data_type):
        """
        创建不包含当前样本的样本列表
        
        Args:
            in_context_samples: 所有样本列表
            current_x: 当前样本的x特征
            current_y: 当前样本的y特征
            data_type: 数据类型（'regression'或'classification'）
            
        Returns:
            List of tuples (x, y, value) excluding the current sample
        """
        all_samples = []
        for each in in_context_samples:
            if len(each.get('features')) >= 2:
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
                
                # 使用安全比较函数
                if sample_value is not None:
                    # 检查是否与当前样本不同
                    x_diff = safe_not_equals(feat_x, current_x)
                    y_diff = safe_not_equals(feat_y, current_y)
                    
                    if x_diff or y_diff:
                        all_samples.append((feat_x, feat_y, sample_value))
        
        return all_samples
    
    # Process each model
    for model_idx, model in enumerate(models):
        if not isinstance(model, dict):
            print(f"Model {model_idx} is not a dictionary, skipping")
            continue
        
        # Extract model details
        model_desc = model.get('description', f'Model {model_idx}')
        model_func_str = model.get('function', '')
        model_family = model.get('model_family', 'cannot parse model family')  # Extract model_family
        
        # print(f"\nProcessing model {model_idx+1}/{len(models)}: {model_desc}")
        # print(f"Model family: {model_family}")
        
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
            
            # print(f"Function name identified: {func_name}")
            
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
            
            # print(f"Evaluating model on {len(in_context_samples)} samples...")
            
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
                
                
                # Convert to Python native type if it's a NumPy scalar
                if hasattr(true_value, 'item') and callable(getattr(true_value, 'item')):
                    true_value = true_value.item()
                
                # Try to call the model function with the sample features
                try:
                    # Most functions expect x, y as separate args
                    x, y = sample_features[0], sample_features[1]
                    
                    # Create all_samples by excluding the current sample
                    all_samples = create_all_samples(in_context_samples, x, y, data_type)
                    
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
                                # print(f"    Prediction: {pred}, True: {true_value}, Squared Error: {squared_error}")
                        except (ValueError, TypeError) as e:
                            print(f"    Error converting prediction to float: {e}")
                            pred = None
                    elif data_type == 'classification':
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
                    else:
                        raise NotImplementedError(f"Data type {data_type} not implemented")
                    
                    # Add prediction to results
                    if data_type == 'regression':
                        predictions.append({
                            'features': sample_features,
                            'true_value': true_value,
                            'predicted': pred,
                            'squared_error': (pred - true_value) ** 2 if pred is not None else None
                        })
                    elif data_type == 'classification':
                        predictions.append({
                            'features': sample_features,
                            'true_label': true_value,
                            'predicted': pred,
                            'correct': pred == true_value if pred is not None else False
                        })
                    else:
                        raise NotImplementedError(f"Data type {data_type} not implemented")
                    
                    total_count += 1
                    
                    
                    
                except Exception as e:
                    # print(f"  Sample {sample_idx}: Error with primary approach: {e}")
                    # Print traceback for better debugging
                    # import traceback
                    # traceback.print_exc()
                    # input("Press Enter to continue...")
                    # print(model_func_str)
                    # input("Press Enter to continue2...")
                    # If all attempts fail, log the error
                    if data_type == 'regression':
                        predictions.append({
                            'features': sample_features,
                            'true_value': true_value,
                            'predicted': None,
                            'squared_error': None,
                            'error': str(e)
                        })
                    else:
                        predictions.append({
                            'features': sample_features,
                            'true_label': true_value,
                            'predicted': None,
                            'correct': False,
                            'error': str(e)
                        })
                    
                    # Still count this sample
                    total_count += 1
            
            # Calculate final metrics
            if data_type == 'regression':
                # For regression, calculate MSE
                mse = sum(squared_errors) / len(squared_errors) if squared_errors else float('inf')
                # print(f"Model evaluation complete: MSE = {mse:.6f} (on {len(squared_errors)} valid predictions)")
                
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
            elif data_type == 'classification':
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
            else:
                raise NotImplementedError(f"Data type {data_type} not implemented")
            
            # print(f"Generated {len(predictions)} predictions")
            
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

def compute_model_family_best_mse(model_family, predictions):
    """
    Compute the best MSE using scikit-learn's implementation of the model family.
    
    Args:
        model_family: String indicating scikit-learn model family (e.g., 'sklearn.linear_model.LinearRegression')
        predictions: List of prediction dictionaries with features and true_value
        
    Returns:
        Best MSE achieved by scikit-learn implementation, or None if computation fails
    """
    try:
        # Skip if no predictions or model_family is unknown
        if not predictions or model_family == "unknown" or model_family == "custom":
            return None
            
        # Extract features and targets from predictions
        X = []
        y = []
        for pred in predictions:
            if 'features' in pred and 'true_value' in pred and pred['features'] and len(pred['features']) >= 2:
                # For simplicity, we only use the first two features x and y
                X.append(pred['features'][:2])
                y.append(pred['true_value'])
                
        # Skip if not enough data points
        if len(X) < 2:
            return None
            
        # Import scikit-learn dynamically
        import importlib
        import numpy as np
        from sklearn.metrics import mean_squared_error
        
        # Handle common model families
        best_mse = None
        
        if 'sklearn.linear_model.LinearRegression' in model_family or 'linear' in model_family.lower():
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            model.fit(X, y)
            y_pred = model.predict(X)
            best_mse = mean_squared_error(y, y_pred)
            
        elif 'sklearn.linear_model.Ridge' in model_family or 'ridge' in model_family.lower():
            from sklearn.linear_model import Ridge
            # Try different alpha values and pick the best
            best_mse = float('inf')
            for alpha in [0.01, 0.1, 1.0, 10.0]:
                model = Ridge(alpha=alpha)
                model.fit(X, y)
                y_pred = model.predict(X)
                mse = mean_squared_error(y, y_pred)
                if mse < best_mse:
                    best_mse = mse
                    
        elif 'sklearn.linear_model.Lasso' in model_family or 'lasso' in model_family.lower():
            from sklearn.linear_model import Lasso
            # Try different alpha values and pick the best
            best_mse = float('inf')
            for alpha in [0.01, 0.1, 1.0]:
                model = Lasso(alpha=alpha)
                model.fit(X, y)
                y_pred = model.predict(X)
                mse = mean_squared_error(y, y_pred)
                if mse < best_mse:
                    best_mse = mse
                    
        elif 'sklearn.linear_model.ElasticNet' in model_family or 'elasticnet' in model_family.lower():
            from sklearn.linear_model import ElasticNet
            best_mse = float('inf')
            for alpha in [0.01, 0.1, 1.0]:
                for l1_ratio in [0.1, 0.5, 0.9]:
                    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
                    model.fit(X, y)
                    y_pred = model.predict(X)
                    mse = mean_squared_error(y, y_pred)
                    if mse < best_mse:
                        best_mse = mse
                        
        elif 'sklearn.preprocessing.PolynomialFeatures' in model_family or 'polynomial' in model_family.lower():
            from sklearn.preprocessing import PolynomialFeatures
            from sklearn.linear_model import LinearRegression
            from sklearn.pipeline import Pipeline
            
            best_mse = float('inf')
            for degree in [2]:
                model = Pipeline([
                    ('poly', PolynomialFeatures(degree=degree)),
                    ('linear', LinearRegression())
                ])
                model.fit(X, y)
                y_pred = model.predict(X)
                mse = mean_squared_error(y, y_pred)
                if mse < best_mse:
                    best_mse = mse
                    
        elif 'sklearn.svm' in model_family or 'svm' in model_family.lower():
            from sklearn.svm import SVR
            best_mse = float('inf')
            for C in [0.1, 1.0, 10.0]:
                for kernel in ['linear', 'rbf']:
                    model = SVR(kernel=kernel, C=C)
                    model.fit(X, y)
                    y_pred = model.predict(X)
                    mse = mean_squared_error(y, y_pred)
                    if mse < best_mse:
                        best_mse = mse
                        
        elif 'sklearn.tree' in model_family or 'decision tree' in model_family.lower():
            from sklearn.tree import DecisionTreeRegressor
            best_mse = float('inf')
            for max_depth in [2]:
                model = DecisionTreeRegressor(max_depth=max_depth)
                model.fit(X, y)
                y_pred = model.predict(X)
                mse = mean_squared_error(y, y_pred)
                if mse < best_mse:
                    best_mse = mse
                    
        # If no specific model family was matched but we have "linear" in the name
        elif 'linear' in model_family.lower():
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            model.fit(X, y)
            y_pred = model.predict(X)
            best_mse = mean_squared_error(y, y_pred)
            
        # Fallback to simple linear regression if model family not recognized
        # elif "custom" in model_family.lower():
        #     from sklearn.linear_model import LinearRegression
        #     model = LinearRegression()
        #     model.fit(X, y)
        #     y_pred = model.predict(X)
        #     best_mse = mean_squared_error(y, y_pred)
            
        return best_mse
        
    except Exception as e:
        print(f"Error computing best MSE for model family {model_family}: {e}")
        import traceback
        traceback.print_exc()
        return None

def leave_one_out_cv(model_class, X, y, **model_params):
    """
    使用留一验证(Leave-One-Out Cross Validation)评估模型性能
    
    Args:
        model_class: 模型类(如KNeighborsClassifier)
        X: 特征数据，形状为(n_samples, n_features)
        y: 标签数据，形状为(n_samples,)
        **model_params: 传递给模型的参数
        
    Returns:
        准确率(正确预测的比例)
    """
    import numpy as np
    
    n_samples = len(X)
    correct = 0
    
    # 对每个样本进行留一验证
    for i in range(n_samples):
        # 获取训练数据(排除当前样本)
        X_train = np.delete(X, i, axis=0)
        y_train = np.delete(y, i)
        
        # 获取测试数据(当前样本)
        X_test = X[i:i+1]
        y_test = y[i]
        
        # 训练模型
        model = model_class(**model_params)
        model.fit(X_train, y_train)
        
        # 预测
        y_pred = model.predict(X_test)[0]
        
        # 检查是否正确
        if y_pred == y_test:
            correct += 1
    
    # 计算准确率
    accuracy = correct / n_samples if n_samples > 0 else 0
    return accuracy

def compute_model_family_best_accuracy(model_family, predictions):
    """
    Compute the best accuracy using scikit-learn's implementation of the model family.
    
    Args:
        model_family: String indicating scikit-learn model family (e.g., 'sklearn.linear_model.LogisticRegression')
        predictions: List of prediction dictionaries with features and true_label
        
    Returns:
        Best accuracy achieved by scikit-learn implementation, or None if computation fails
    """
    try:
        # Skip if no predictions or model_family is unknown
        if not predictions or model_family == "unknown" or model_family == "custom":
            return None
            
        # Extract features and targets from predictions
        X = []
        y = []
        for pred in predictions:
            if 'features' in pred and 'true_label' in pred and pred['features'] and len(pred['features']) >= 2:
                # For simplicity, we only use the first two features x and y
                X.append(pred['features'][:2])
                y.append(pred['true_label'])
                
        # Skip if not enough data points
        if len(X) < 2:
            return None
            
        # Import scikit-learn dynamically
        import numpy as np
        from sklearn.metrics import accuracy_score
        
        # Handle common model families
        best_accuracy = None
        
        if 'sklearn.linear_model.LogisticRegression' in model_family or 'logistic' in model_family.lower():
            from sklearn.linear_model import LogisticRegression
            best_accuracy = 0.0
            for C in [0.01, 0.1, 0.5, 1.0, 10.0, 100.0]:
                model = LogisticRegression(C=C, max_iter=1000)
                model.fit(X, y)
                y_pred = model.predict(X)
                acc = accuracy_score(y, y_pred)
                if acc > best_accuracy:
                    best_accuracy = acc
                    
        elif 'sklearn.svm' in model_family or 'svm' in model_family.lower():
            from sklearn.svm import SVC
            best_accuracy = 0.0
            for C in [0.1, 1.0, 10.0]:
                for kernel in ['linear', 'rbf']:
                    model = SVC(kernel=kernel, C=C)
                    model.fit(X, y)
                    y_pred = model.predict(X)
                    acc = accuracy_score(y, y_pred)
                    if acc > best_accuracy:
                        best_accuracy = acc
                        
        elif 'sklearn.tree' in model_family or 'decision tree' in model_family.lower():
            from sklearn.tree import DecisionTreeClassifier
            best_accuracy = 0.0
            for max_depth in [2]:
                model = DecisionTreeClassifier(max_depth=max_depth)
                model.fit(X, y)
                y_pred = model.predict(X)
                acc = accuracy_score(y, y_pred)
                if acc > best_accuracy:
                    best_accuracy = acc
                    
        # elif 'sklearn.ensemble.RandomForest' in model_family or 'random forest' in model_family.lower():
        #     from sklearn.ensemble import RandomForestClassifier
        #     best_accuracy = 0.0
        #     for n_estimators in [10, 50, 100]:
        #         for max_depth in [2, 5, None]:
        #             model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
        #             model.fit(X, y)
        #             y_pred = model.predict(X)
        #             acc = accuracy_score(y, y_pred)
        #             if acc > best_accuracy:
        #                 best_accuracy = acc
                        
        elif 'sklearn.neighbors' in model_family or 'knn' in model_family.lower():
            from sklearn.neighbors import KNeighborsClassifier
            best_accuracy = 0.0
            X_np = np.array(X)
            y_np = np.array(y)
            for n_neighbors in [1,2,3,4,5,6,7]:
                # 使用留一验证评估KNN模型
                acc = leave_one_out_cv(KNeighborsClassifier, X_np, y_np, n_neighbors=n_neighbors)
                if acc > best_accuracy:
                    best_accuracy = acc
                    
        # elif 'sklearn.naive_bayes' in model_family or 'naive bayes' in model_family.lower():
        #     from sklearn.naive_bayes import GaussianNB
        #     model = GaussianNB()
        #     model.fit(X, y)
        #     y_pred = model.predict(X)
        #     best_accuracy = accuracy_score(y, y_pred)
            
        # Fallback to simple logistic regression if model family not recognized
        # elif "custom" in model_family.lower():
        #     from sklearn.linear_model import LogisticRegression
        #     model = LogisticRegression(max_iter=1000)
        #     model.fit(X, y)
        #     y_pred = model.predict(X)
        #     best_accuracy = accuracy_score(y, y_pred)
            
        return best_accuracy * 100  # Convert to percentage to match the accuracy field
        
    except Exception as e:
        print(f"Error computing best accuracy for model family {model_family}: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_model_evaluation_table(model_results, models_data, data_type):
    """Create model evaluation table from results."""
    evaluation_table = []
    
    for idx, result in enumerate(model_results):
        model_desc = result.get('model_desc', f'Model {idx}')
        has_error = 'error' in result
        
        # Get order from original model
        model_order = "ERROR: No order"
        original_model = next((m for m in models_data 
                              if isinstance(m, dict) and m.get('description') == model_desc), None)
        
        if original_model and 'order' in original_model:
            model_order = original_model['order']
        
        # Create row based on data type
        if data_type == 'regression':
            mse = result.get('mse', float('inf'))
            total_count = result.get('total_count', 0)
            valid_predictions = result.get('valid_predictions', 0)
            model_family = result.get("model_family", "unknown")
            predictions = result.get("predictions", [])
            
            # Compute model_family_best_mse
            model_family_best_mse = compute_model_family_best_mse(model_family, predictions)
            
            table_row = {
                "order": model_order,
                "model": model_desc,
                "mse": mse,
                "valid_predictions": valid_predictions,
                "total_count": total_count,
                "details": f"MSE: {mse:.6f} ({valid_predictions} valid)" if not has_error else "Error",
                "has_error": has_error,
                "model_code": result.get("model_func", ""),
                "model_family": model_family,
                "model_family_best_mse": model_family_best_mse,
                "predictions": predictions[:10]  # Include first 10 predictions
            }
        elif data_type == 'classification':
            accuracy = result.get('accuracy', 0.0)
            correct_count = result.get('correct_count', 0)
            total_count = result.get('total_count', 0)
            model_family = result.get("model_family", "unknown")
            predictions = result.get("predictions", [])
            
            # Compute model_family_best_accuracy
            model_family_best_accuracy = compute_model_family_best_accuracy(model_family, predictions)
            
            table_row = {
                "order": model_order,
                "model": model_desc,
                "accuracy": accuracy,
                "correct_count": correct_count,
                "total_count": total_count,
                "details": f"{correct_count} of {total_count} correct" if not has_error else "Error",
                "has_error": has_error,
                "model_code": result.get("model_func", ""),
                "model_family": model_family,
                "model_family_best_accuracy": model_family_best_accuracy,
                "predictions": predictions[:10]  # Include first 10 predictions
            }
        else:
            raise NotImplementedError(f"Data type {data_type} not implemented")
        
        if has_error:
            table_row["error"] = result.get("error", "Unknown error")
        
        # Add debug info if predictions are empty
        if not result.get("predictions"):
            table_row["debug_info"] = {
                "has_predictions": False,
                "reason": "No predictions were generated during model execution",
                "possible_causes": [
                    "Model function execution failed",
                    "In-context samples could not be processed",
                    "Model function raised exceptions for all samples"
                ]
            }
        
        evaluation_table.append(table_row)
    
    return evaluation_table

def select_reward_fn(data_source: str) -> Callable:
    """
    Select the appropriate reward function based on data source
    
    Args:
        data_source: The data source name (e.g., 'blobs', 'circles')
        
    Returns:
        Reward function appropriate for the data source
    """
    from verl.trainer.ppo.helper import _select_rm_score_fn as orig_select_reward_fn
    return orig_select_reward_fn(data_source)