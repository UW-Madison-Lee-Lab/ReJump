#!/usr/bin/env python3
"""
Extract data from parquet files with model responses and save to JSON
Default input file: /staging/szhang967/icl_dataset-output/blobs_50shot_n1.0_f0.0_test10_icl3_seed42.parquet
"""

import pandas as pd
import json
import os
import argparse
import random
from pathlib import Path
from typing import Dict, Any, List, Optional

# Import helper functions from data_processing
from data_processing import (
    process_icl_examples, 
    process_test_examples, 
    add_sample_data_fields,
    get_num_classes,
    get_tokenizer,
    calculate_token_length,
    extract_think_content,
    extract_answer_content,
    clean_response_text,
    get_prediction_result,
    extract_and_execute_model_functions,
    select_reward_fn
)

try:
    from transformers import AutoTokenizer
except ImportError:
    print("Warning: Could not import AutoTokenizer from transformers")
    AutoTokenizer = None

def extract_json_from_parquet(input_file: str, output_dir: Optional[str] = None, max_samples: int = 0, data_type: str = 'regression'):
    """
    Extract data from parquet files with model responses and save to JSON
    
    Args:
        input_file: Path to the input parquet file
        output_dir: Directory to save the output file (default: same as input file's directory)
        max_samples: Maximum number of samples to include (default: 0 = all samples)
        
    Returns:
        Path to the output JSON file
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
        else:
            raise ValueError("'responses' column not found in DataFrame")
        
    except Exception as e:
        print(f"Error reading parquet file: {e}")
        raise
    
    # Calculate accuracy on full dataset first
    total_data_size = len(df)
    full_correct_predictions = 0
    full_refined_correct = 0  # For storing the correct count for refined accuracy
    unparseable_predictions = 0  # Count of unparseable predictions
    parseable_predictions = 0  # Count of parseable predictions
    parseable_correct = 0  # Count of correct predictions among parseable ones
    
    # Variables for MSE calculation in regression mode
    total_squared_error = 0.0
    valid_prediction_count = 0
    
    print(f"Calculating accuracy on all {total_data_size} samples...")
    
    # Process all samples for accuracy calculation
    for idx, row in df.iterrows():
        # For regression samples, output the first few to debug
        if data_type == 'regression' and idx < 5:  # Only show first 5 for debugging
            # Extract the ground truth and examine it
            ground_truth_data = None
            if 'reward_model' in row and isinstance(row['reward_model'], dict):
                ground_truth_data = row['reward_model'].get('ground_truth', None)
                if isinstance(ground_truth_data, dict):
                    print(f"Debug Sample {idx} - Ground truth: {ground_truth_data}")
                    
            # Extract the raw response
            responses = row.get('responses', [])
            response_text = responses[0] if responses and len(responses) > 0 else "No response"
            if isinstance(response_text, str) and len(response_text) > 200:
                short_response = response_text[:200] + "..."
            else:
                short_response = response_text
            print(f"Debug Sample {idx} - Response: {short_response}")
            
            # Extract raw answer
            raw_answer = extract_answer_content(clean_response_text(response_text, None))
            print(f"Debug Sample {idx} - Raw answer: {raw_answer}")

        # Get ground truth label/target
        ground_truth = None
        data_source = row.get('data_source', '')
        
        if 'reward_model' in row and isinstance(row['reward_model'], dict):
            ground_truth_data = row['reward_model'].get('ground_truth', None)
            if isinstance(ground_truth_data, dict):
                if 'label' in ground_truth_data:
                    ground_truth = ground_truth_data
                elif 'target' in ground_truth_data:
                    # If there's no label but there's a target, use the target as label
                    ground_truth = ground_truth_data.copy()
                    ground_truth['label'] = ground_truth_data['target']
                    print(f"Using target as label for ground_truth")

        # Skip further processing if ground_truth is not available
        if not ground_truth:
            continue
            
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
        
        # Calculate token length
        token_length = calculate_token_length(cleaned_response_text, tokenizer)
        
        # Get reward function for the data source
        reward_fn = select_reward_fn(data_source)
        
        # Extract thinking and answer
        raw_thinking = extract_think_content(cleaned_response_text)
        raw_answer = extract_answer_content(cleaned_response_text)
        
        # Initialize variables for sample processing
        is_correct = False
        prediction = None
        squared_error = None
        
        try:
            if data_type == 'regression':
                # For regression, extract prediction and calculate squared error
                # Try to extract from raw_answer first
                if raw_answer:
                    try:
                        prediction = float(raw_answer.strip())
                    except ValueError:
                        prediction = None
                
                # If no prediction from raw_answer, try get_prediction_result
                if prediction is None:
                    prediction, _ = get_prediction_result(cleaned_response_text, ground_truth, data_type)
                
                # Get ground truth value for regression
                ground_truth_value = None
                if ground_truth:
                    if 'target' in ground_truth:
                        ground_truth_value = ground_truth['target']
                    elif 'label' in ground_truth:
                        ground_truth_value = ground_truth['label']
                    
                    # Convert to Python native type if it's a NumPy scalar
                    if ground_truth_value is not None and hasattr(ground_truth_value, 'item') and callable(getattr(ground_truth_value, 'item')):
                        ground_truth_value = ground_truth_value.item()
                
                # Calculate squared error if possible
                if prediction is not None and ground_truth_value is not None:
                    try:
                        # Ensure both values are numeric
                        prediction_float = float(prediction)
                        ground_truth_float = float(ground_truth_value)
                        squared_error = (prediction_float - ground_truth_float) ** 2
                    except (ValueError, TypeError) as e:
                        print(f"Warning: Error calculating squared error for sample {idx}: {e}")
                        squared_error = None
            else:
                # For classification, use traditional correctness evaluation
                is_correct = reward_fn(cleaned_response_text, ground_truth)
                
                # Extract prediction for display with enhanced matching
                prediction = None
                # First check if there's a clean answer tag
                if raw_answer and raw_answer.strip().isdigit():
                    prediction = int(raw_answer.strip())
                
                if prediction is None:
                    # Try to extract prediction using get_prediction_result function
                    parsed_prediction, _ = get_prediction_result(cleaned_response_text, ground_truth, data_type)
                    if parsed_prediction is not None:
                        prediction = parsed_prediction
                    elif ground_truth and 'label' in ground_truth and is_correct:
                        # If correct but can't parse prediction, use ground truth
                        prediction = ground_truth['label']
        
        except Exception as e:
            print(f"Warning: Error processing sample {idx}: {e}")
            # Fallback to simple prediction extraction
            try:
                prediction, is_correct = get_prediction_result(cleaned_response_text, ground_truth, data_type)
                
                # For regression, calculate squared error
                if data_type == 'regression' and prediction is not None:
                    ground_truth_value = None
                    if ground_truth:
                        if 'target' in ground_truth:
                            ground_truth_value = ground_truth['target']
                        elif 'label' in ground_truth:
                            ground_truth_value = ground_truth['label']
                        
                        if ground_truth_value is not None:
                            # Convert to Python native type if it's a NumPy scalar
                            if hasattr(ground_truth_value, 'item') and callable(getattr(ground_truth_value, 'item')):
                                ground_truth_value = ground_truth_value.item()
                            
                            squared_error = (prediction - ground_truth_value) ** 2
            except Exception as e2:
                print(f"Warning: Final fallback error for sample {idx}: {e2}")
        
        # Update metrics based on data type
        if data_type == 'regression':
            # For regression, track predictions that have a valid squared error
            if squared_error is not None:
                parseable_predictions += 1
                # Accumulate squared error for MSE calculation
                total_squared_error += squared_error
                valid_prediction_count += 1
        else:
            # For classification, track correctness
            if is_correct:
                full_correct_predictions += 1
                full_refined_correct += 1  # Correct predictions also count as correct in refined accuracy
                if prediction is not None:
                    parseable_correct += 1
            
            # Track parseable predictions
            if prediction is not None:
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
    
    # Calculate metrics based on data type
    if data_type == 'regression':
        # For regression, we report different metrics
        print(f"Processed {total_data_size} regression samples")
        print(f"Parseable predictions: {parseable_predictions}/{total_data_size} samples ({parseable_predictions/total_data_size*100:.2f}%)")
        print(f"Unparseable predictions: {total_data_size - parseable_predictions} ({(total_data_size - parseable_predictions)/total_data_size*100:.2f}%)")
        print(f"Valid predictions for MSE: {valid_prediction_count}")
        print(f"Total squared error: {total_squared_error}")
        
        # Calculate MSE from the accumulated values
        mse = total_squared_error / valid_prediction_count if valid_prediction_count > 0 else float('inf')
        print(f"Mean Squared Error (MSE): {mse:.6f} (based on {valid_prediction_count} valid predictions)")
        
        # Set these values for metadata
        accuracy = 0.0  # Not applicable for regression
        refined_accuracy = 0.0  # Not applicable for regression
        parseable_accuracy = 0.0  # Not applicable for regression
    else:
        # For classification, calculate accuracy metrics
        accuracy = (full_correct_predictions / total_data_size) * 100 if total_data_size > 0 else 0
        refined_accuracy = (full_refined_correct / total_data_size) * 100 if total_data_size > 0 else 0
        parseable_accuracy = (parseable_correct / parseable_predictions) * 100 if parseable_predictions > 0 else 0
        
        print(f"Overall accuracy from all {total_data_size} samples: {accuracy:.2f}%")
        print(f"Refined accuracy (random guess for unparseable): {refined_accuracy:.2f}%")
        print(f"Parseable accuracy (excluding unparseable): {parseable_accuracy:.2f}% ({parseable_predictions}/{total_data_size} samples)")
        print(f"Unparseable predictions: {unparseable_predictions} ({unparseable_predictions/total_data_size*100:.2f}%)")
    
    # Sample the dataframe if needed for visualization
    display_df = df
    if max_samples > 0 and max_samples < total_data_size:
        display_df = df.sample(n=max_samples, random_state=42)
        print(f"Sampled {len(display_df)} out of {total_data_size} samples for JSON extraction")
    
    # Determine the output file path
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = Path(input_file).parent
    
    json_output_file = output_dir / f"{Path(input_file).stem}_data.json"
    print(f"Output file will be: {json_output_file}")
    
    # Prepare JSON data
    json_data = {
        "metadata": {
            "input_file": str(input_file),
            "timestamp": pd.Timestamp.now().isoformat(),
            "total_samples": total_data_size,
            "displayed_samples": len(display_df),
            "data_type": data_type,
        },
        "samples": []
    }
    
    # Add metrics based on data type
    if data_type == 'regression':
        # For regression, include MSE metrics
        json_data["metadata"].update({
            "mse": mse,
            "valid_predictions": valid_prediction_count,
            "parseable_predictions": parseable_predictions,
            "unparseable_predictions": total_data_size - parseable_predictions
        })
        
        # Add regression metrics documentation
        json_data["metadata"]["model_evaluation_metrics"] = {
            "mse": "Mean Squared Error (lower is better)",
            "valid_predictions": "Number of valid predictions (those that could be calculated)",
            "parseable_predictions": "Number of samples where prediction could be extracted",
            "unparseable_predictions": "Number of samples where prediction could not be extracted"
        }
    else:
        # For classification, include accuracy metrics
        json_data["metadata"].update({
            "accuracy": accuracy,
            "refined_accuracy": refined_accuracy,
            "parseable_accuracy": parseable_accuracy,
            "parseable_predictions": parseable_predictions,
            "unparseable_predictions": unparseable_predictions
        })
        
        # Add classification metrics documentation
        json_data["metadata"]["model_evaluation_metrics"] = {
            "accuracy": "Accuracy percentage",
            "refined_accuracy": "Accuracy with random guesses for unparseable predictions",
            "parseable_accuracy": "Accuracy among only parseable predictions",
            "parseable_predictions": "Number of samples where prediction could be extracted",
            "unparseable_predictions": "Number of samples where prediction could not be extracted"
        }
    
    # Add table info based on data type
    json_data["metadata"]["model_evaluation_table_info"] = {
        "description": "Each sample may contain a 'model_evaluation_table' field with model evaluation results.",
        "structure": {
            "order": "Model order number or identifier",
            "model": "Model description",
            "has_error": "Boolean indicating if the model had errors during execution",
            "error": "Error message (only present if has_error is true)",
            "model_code": "The model function code as a string",
            "model_family": "Type of the model (if available)",
            "predictions": "Array of predictions, each containing features, true value/label, and metrics"
        }
    }
    
    # Add all samples that will be displayed
    for idx, row in display_df.iterrows():
        # Get ground truth label
        ground_truth = None
        if 'reward_model' in row and isinstance(row['reward_model'], dict):
            ground_truth_data = row['reward_model'].get('ground_truth', None)
            if isinstance(ground_truth_data, dict):
                if 'label' in ground_truth_data:
                    ground_truth = ground_truth_data
                elif 'target' in ground_truth_data:
                    # If there's no label but there's a target, use the target as label
                    ground_truth = ground_truth_data.copy()
                    ground_truth['label'] = ground_truth_data['target']
                    print(f"Using target as label for ground_truth")
        
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
        data_source = row.get('data_source', '')
        
        # Get reward function for the data source
        reward_fn = select_reward_fn(data_source)
        
        # Extract thinking and answer
        raw_thinking = extract_think_content(cleaned_response_text)
        raw_answer = extract_answer_content(cleaned_response_text)
        
        # Initialize variables for sample processing
        is_correct = False
        prediction = None
        squared_error = None
        
        try:
            if data_type == 'regression':
                # For regression, extract prediction and calculate squared error
                # Try to extract from raw_answer first
                if raw_answer:
                    try:
                        prediction = float(raw_answer.strip())
                    except ValueError:
                        prediction = None
                
                # If no prediction from raw_answer, try get_prediction_result
                if prediction is None:
                    prediction, _ = get_prediction_result(cleaned_response_text, ground_truth, data_type)
                
                # Get ground truth value for regression
                ground_truth_value = None
                if ground_truth:
                    if 'target' in ground_truth:
                        ground_truth_value = ground_truth['target']
                    elif 'label' in ground_truth:
                        ground_truth_value = ground_truth['label']
                    
                    # Convert to Python native type if it's a NumPy scalar
                    if ground_truth_value is not None and hasattr(ground_truth_value, 'item') and callable(getattr(ground_truth_value, 'item')):
                        ground_truth_value = ground_truth_value.item()
                
                # Calculate squared error if possible
                if prediction is not None and ground_truth_value is not None:
                    try:
                        # Ensure both values are numeric
                        prediction_float = float(prediction)
                        ground_truth_float = float(ground_truth_value)
                        squared_error = (prediction_float - ground_truth_float) ** 2
                    except (ValueError, TypeError) as e:
                        print(f"Warning: Error calculating squared error for sample {idx}: {e}")
                        squared_error = None
            else:
                # For classification, use traditional correctness evaluation
                is_correct = reward_fn(cleaned_response_text, ground_truth)
                
                # Extract prediction for display with enhanced matching
                prediction = None
                # First check if there's a clean answer tag
                if raw_answer and raw_answer.strip().isdigit():
                    prediction = int(raw_answer.strip())
                
                if prediction is None:
                    # Try to extract prediction using get_prediction_result function
                    parsed_prediction, _ = get_prediction_result(cleaned_response_text, ground_truth, data_type)
                    if parsed_prediction is not None:
                        prediction = parsed_prediction
                    elif ground_truth and 'label' in ground_truth and is_correct:
                        # If correct but can't parse prediction, use ground truth
                        prediction = ground_truth['label']
        
        except Exception as e:
            print(f"Warning: Error processing sample {idx}: {e}")
            # Fallback to simple prediction extraction
            try:
                prediction, is_correct = get_prediction_result(cleaned_response_text, ground_truth, data_type)
                
                # For regression, calculate squared error
                if data_type == 'regression' and prediction is not None:
                    ground_truth_value = None
                    if ground_truth:
                        if 'target' in ground_truth:
                            ground_truth_value = ground_truth['target']
                        elif 'label' in ground_truth:
                            ground_truth_value = ground_truth['label']
                        
                        if ground_truth_value is not None:
                            # Convert to Python native type if it's a NumPy scalar
                            if hasattr(ground_truth_value, 'item') and callable(getattr(ground_truth_value, 'item')):
                                ground_truth_value = ground_truth_value.item()
                            
                            squared_error = (prediction - ground_truth_value) ** 2
            except Exception as e2:
                print(f"Warning: Final fallback error for sample {idx}: {e2}")
        
        # Create sample data with fields appropriate for the data type
        if data_type == 'regression':
            # For regression, include squared_error instead of is_correct
            sample_data = {
                "index": int(idx),
                "data_source": data_source,
                "ground_truth": ground_truth,
                "prediction": prediction,
                "squared_error": squared_error,
                "cleaned_response": cleaned_response_text,
                "raw_thinking": raw_thinking,
                "raw_answer": raw_answer
            }
        else:
            # For classification, keep is_correct
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
        
        # Add additional data fields
        sample_data = add_sample_data_fields(sample_data, row)
        
        # Extract model evaluation table data if available
        if 'llm_analysis_extracted_json' in row and row['llm_analysis_extracted_json'] is not None and ground_truth is not None and 'in_context_samples' in ground_truth:
            try:
            #     llm_json = row['llm_analysis_extracted_json']
                llm_json = row['llm_analysis_extracted_json']
                # Extract model functions and evaluate
                model_results = extract_and_execute_model_functions(llm_json, ground_truth, data_type)
                
                # Parse the JSON to get the original models data
                models_data = []
                try:
                    if isinstance(llm_json, str):
                        models_data = json.loads(llm_json)
                    else:
                        models_data = llm_json
                    if not isinstance(models_data, list):
                        models_data = []
                except Exception:
                    models_data = []
                
                # Create model evaluation table data
                evaluation_table = []
                for idx, result in enumerate(model_results):
                    model_desc = result.get('model_desc', f'Model {idx}')
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
                    
                    # Create table row data with different metrics based on data_type
                    if data_type == 'regression':
                        # For regression models
                        mse = result.get('mse', float('inf'))
                        total_count = result.get('total_count', 0)
                        valid_predictions = result.get('valid_predictions', 0)
                        
                        table_row = {
                            "order": model_order,
                            "model": model_desc,
                            "mse": mse,
                            "valid_predictions": valid_predictions,
                            "total_count": total_count,
                            "details": f"MSE: {mse:.6f} ({valid_predictions} valid)" if not has_error else "Error",
                            "has_error": has_error,
                            "model_code": result.get("model_func", ""),
                            "model_family": result.get("model_family", "unknown"),
                            "predictions": result.get("predictions", [])[:10]  # Include first 10 predictions
                        }
                    else:
                        # For classification models
                        accuracy = result.get('accuracy', 0.0)
                        correct_count = result.get('correct_count', 0)
                        total_count = result.get('total_count', 0)
                        
                        table_row = {
                            "order": model_order,
                            "model": model_desc,
                            "accuracy": accuracy,
                            "correct_count": correct_count,
                            "total_count": total_count,
                            "details": f"{correct_count} of {total_count} correct" if not has_error else "Error",
                            "has_error": has_error,
                            "model_code": result.get("model_func", ""),
                            "model_family": result.get("model_family", "unknown"),
                            "predictions": result.get("predictions", [])[:10]  # Include first 10 predictions
                        }
                    
                    if has_error:
                        table_row["error"] = result.get("error", "Unknown error")
                    
                    # Add additional debug info if predictions are empty
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
                
                # Add to sample data
                sample_data["model_evaluation_table"] = evaluation_table
            except Exception as e:
                print(f"Error creating model evaluation table data: {e}")
        
        # Add to samples list
        json_data["samples"].append(sample_data)
    
    # Save to JSON file
    try:
        # Add some statistics about model evaluation table data
        empty_predictions_count = 0
        non_empty_predictions_count = 0
        error_count = 0
        
        # Count statistics across all samples
        for sample in json_data["samples"]:
            if "model_evaluation_table" in sample:
                for model_row in sample["model_evaluation_table"]:
                    if not model_row.get("predictions"):
                        empty_predictions_count += 1
                    else:
                        non_empty_predictions_count += 1
                    
                    if model_row.get("has_error", False):
                        error_count += 1
        
        print(f"Model evaluation statistics:")
        print(f"  - Total models with empty predictions: {empty_predictions_count}")
        print(f"  - Total models with non-empty predictions: {non_empty_predictions_count}")
        print(f"  - Total models with errors: {error_count}")
        
        with open(json_output_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, default=str)
        print(f"JSON data saved to: {json_output_file}")
        print(f"Model evaluation table data has been included for each applicable sample")
    except Exception as e:
        print(f"Error saving JSON data: {e}")
    
    return str(json_output_file)


def main():
    parser = argparse.ArgumentParser(description='Extract data from parquet files and save to JSON')
    parser.add_argument('--input', type=str, 
                        required=True,
                        help='Path to the input parquet file with model responses')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save the output file (default: same as input file)')
    parser.add_argument('--max-samples', type=int, default=0,
                        help='Maximum number of samples to include (default: 0 = all samples)')
    parser.add_argument('--data_type', type=str, default='regression',
                        help='Data type to extract')
    
    args = parser.parse_args()      
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: File '{args.input}' not found")
        return
    
    # Extract and save JSON
    output_file = extract_json_from_parquet(args.input, args.output_dir, args.max_samples, args.data_type)
    print(f"JSON extraction complete! Saved to: {output_file}")


if __name__ == "__main__":
    main() 