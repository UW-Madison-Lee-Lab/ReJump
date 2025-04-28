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
from typing import Dict, Any, List, Optional, Tuple

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
    select_reward_fn
)

# Import LLM analysis processor
from llm_analysis import process_llm_analysis

try:
    from transformers import AutoTokenizer
except ImportError:
    print("Warning: Could not import AutoTokenizer from transformers")
    AutoTokenizer = None

def create_json_metadata(input_file: str, total_data_size: int, display_df_size: int, data_type: str,
                        mse: float = None, valid_prediction_count: int = None,
                        parseable_predictions: int = None, accuracy: float = None,
                        refined_accuracy: float = None, parseable_accuracy: float = None,
                        unparseable_predictions: int = None) -> dict:
    """
    Create metadata dictionary for JSON output.
    
    Args:
        input_file: Path to the input file
        total_data_size: Total number of samples in the dataset
        display_df_size: Number of samples being displayed
        data_type: Type of data ('regression' or 'classification')
        mse: Mean Squared Error (for regression)
        valid_prediction_count: Number of valid predictions (for regression)
        parseable_predictions: Number of parseable predictions
        accuracy: Overall accuracy (for classification)
        refined_accuracy: Refined accuracy (for classification)
        parseable_accuracy: Accuracy among parseable predictions (for classification)
        unparseable_predictions: Number of unparseable predictions
        
    Returns:
        dict: Metadata dictionary
    """
    metadata = {
        "input_file": str(input_file),
        "timestamp": pd.Timestamp.now().isoformat(),
        "total_samples": total_data_size,
        "displayed_samples": display_df_size,
        "data_type": data_type,
    }
    
    # Add metrics based on data type
    if data_type == 'regression':
        metadata.update({
            "mse": mse,
            "valid_predictions": valid_prediction_count,
            "parseable_predictions": parseable_predictions,
            "unparseable_predictions": total_data_size - parseable_predictions
        })
        
        # Add regression metrics documentation
        metadata["model_evaluation_metrics"] = {
            "mse": "Mean Squared Error (lower is better)",
            "valid_predictions": "Number of valid predictions (those that could be calculated)",
            "parseable_predictions": "Number of samples where prediction could be extracted",
            "unparseable_predictions": "Number of samples where prediction could not be extracted"
        }
    elif data_type == 'classification':
        metadata.update({
            "accuracy": accuracy,
            "refined_accuracy": refined_accuracy,
            "parseable_accuracy": parseable_accuracy,
            "parseable_predictions": parseable_predictions,
            "unparseable_predictions": unparseable_predictions
        })
        
        # Add classification metrics documentation
        metadata["model_evaluation_metrics"] = {
            "accuracy": "Accuracy percentage",
            "refined_accuracy": "Accuracy with random guesses for unparseable predictions",
            "parseable_accuracy": "Accuracy among only parseable predictions",
            "parseable_predictions": "Number of samples where prediction could be extracted",
            "unparseable_predictions": "Number of samples where prediction could not be extracted"
        }
    else:
        raise NotImplementedError(f"Data type {data_type} not implemented")
    
    # Add table info
    metadata["model_evaluation_table_info"] = {
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
    
    return metadata

from data_processing import get_ground_truth, get_response_text, get_prompt_content

def process_regression_sample(raw_answer: str, cleaned_response: str, ground_truth: Dict) -> Tuple[Optional[float], Optional[float]]:
    """Process regression sample to extract prediction and squared error."""
    # Extract prediction from raw answer if possible
    prediction = None
    if raw_answer:
        try:
            prediction = float(raw_answer.strip())
        except ValueError:
            prediction = None
            
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
    squared_error = None
    if prediction is not None and ground_truth_value is not None:
        try:
            prediction_float = float(prediction)
            ground_truth_float = float(ground_truth_value)
            squared_error = (prediction_float - ground_truth_float) ** 2
        except (ValueError, TypeError):
            pass
    
    return prediction, squared_error

def process_classification_sample(raw_answer: str, cleaned_response: str, ground_truth: Dict, reward_fn) -> Tuple[Optional[int], bool]:
    """Process classification sample to extract prediction and correctness."""
    # Check if response is correct
    is_correct = reward_fn(cleaned_response, ground_truth)
    
    # Extract prediction
    prediction = None
    # First check if there's a clean answer tag
    if raw_answer and raw_answer.strip().isdigit():
        prediction = int(raw_answer.strip())
    
    return prediction, is_correct

def create_sample_data(idx, data_source, ground_truth, prediction, cleaned_response_text, 
                      raw_thinking, raw_answer, data_type, squared_error=None, is_correct=False):
    """Create sample data dictionary based on data type."""
    if data_type == 'regression':
        return {
            "index": int(idx),
            "data_source": data_source,
            "ground_truth": ground_truth,
            "prediction": prediction,
            "squared_error": squared_error,
            "cleaned_response": cleaned_response_text,
            "raw_thinking": raw_thinking,
            "raw_answer": raw_answer,
            "model_evaluation_table": None  # Initialize as None
        }
    elif data_type == 'classification':
        return {
            "index": int(idx),
            "data_source": data_source,
            "ground_truth": ground_truth,
            "prediction": prediction,
            "is_correct": is_correct,
            "cleaned_response": cleaned_response_text,
            "raw_thinking": raw_thinking,
            "raw_answer": raw_answer,
            "model_evaluation_table": None  # Initialize as None
        }
    else:
        raise NotImplementedError(f"Data type {data_type} not implemented")

def extract_json_from_parquet(input_file: str, output_dir: Optional[str] = None, 
                             max_samples: int = 0, data_type: str = 'regression',
                             display_llm_analysis: bool = False):
    """Extract data from parquet files with model responses and save to JSON."""
    # Set random seed for consistency
    random.seed(42)
    
    # Initialize tokenizer
    tokenizer = get_tokenizer()
    
    # Read the parquet file
    print(f"Reading parquet file: {input_file}")
    df = pd.read_parquet(input_file)
    print(f"DataFrame loaded with {len(df)} rows and columns: {df.columns.tolist()}")
    assert 'responses' in df.columns, "'responses' column not found in DataFrame"
    
    # Initialize metrics tracking
    total_data_size = len(df)
    samples_data = []
    
    # Classification-specific metrics
    full_correct_predictions = 0
    full_refined_correct = 0
    parseable_predictions = 0
    parseable_correct = 0
    unparseable_predictions = 0
    
    # Regression-specific metrics
    total_squared_error = 0.0
    valid_prediction_count = 0
    
    # Sample the dataframe if needed
    display_df = df
    if max_samples > 0 and max_samples < total_data_size:
        display_df = df.sample(n=max_samples, random_state=42)
        print(f"Sampled {len(display_df)} out of {total_data_size} samples for JSON extraction")
    
    print(f"Processing {len(display_df)} samples...")
    
    # Single pass through the data to both collect samples and calculate metrics
    for idx, row in display_df.iterrows():
        # Get ground truth and data source
        ground_truth = get_ground_truth(row)
        assert ground_truth is not None, f"No ground truth found for sample {idx}"
        
        data_source = row['data_source']
        
        # Get response text
        response_text = get_response_text(row)
        
        # Get input prompt and clean response
        input_prompt_content = get_prompt_content(row)
        cleaned_response_text = clean_response_text(response_text, input_prompt_content)
        
        # Remove prompt content from response if present
        if input_prompt_content:
            cleaned_response_text = cleaned_response_text.replace(input_prompt_content, "")
        
        # Extract thinking and answer
        raw_thinking = extract_think_content(cleaned_response_text)
        raw_answer = extract_answer_content(cleaned_response_text)
        
        # Get reward function for the data source
        reward_fn = select_reward_fn(data_source)
        
        # Process sample based on data type
        try:
            if data_type == 'regression':
                prediction, squared_error = process_regression_sample(raw_answer, cleaned_response_text, ground_truth)
                is_correct = False  # Not applicable for regression
                
                # Update regression metrics
                if squared_error is not None:
                    parseable_predictions += 1
                    total_squared_error += squared_error
                    valid_prediction_count += 1
                    
            elif data_type == 'classification':  # classification
                prediction, is_correct = process_classification_sample(raw_answer, cleaned_response_text, ground_truth, reward_fn)
                squared_error = None  # Not applicable for classification
                
                # Update classification metrics
                if is_correct:
                    full_correct_predictions += 1
                    full_refined_correct += 1
                    parseable_correct += 1
                
                if prediction is not None:
                    parseable_predictions += 1
                elif 'label' in ground_truth:
                    # For unparseable predictions, calculate refined accuracy with random guessing
                    unparseable_predictions += 1
                    num_classes = get_num_classes(data_source)
                    random_label = random.randint(0, num_classes - 1)
                    if random_label == ground_truth['label']:
                        full_refined_correct += 1
            else:
                raise NotImplementedError(f"Data type '{data_type}' not implemented")
        except Exception as e:
            print(f"Warning: Error processing sample {idx}: {e}")
            prediction = None
            is_correct = False
            squared_error = None
        
        # Create sample data
        sample_data = create_sample_data(
            idx, data_source, ground_truth, prediction, cleaned_response_text,
            raw_thinking, raw_answer, data_type, squared_error, is_correct
        )
        
        # Add additional data fields
        sample_data = add_sample_data_fields(sample_data, row)
        
        # Process LLM analysis if requested
        if display_llm_analysis and 'llm_analysis_extracted_json' in row and row['llm_analysis_extracted_json'] is not None and 'in_context_samples' in ground_truth:
            try:
                llm_json = row['llm_analysis_extracted_json']
                evaluation_table = process_llm_analysis(llm_json, ground_truth, data_type)
                if evaluation_table:
                    sample_data["model_evaluation_table"] = evaluation_table
            except Exception as e:
                print(f"Error processing LLM analysis: {e}")
        
        # Add to samples list
        samples_data.append(sample_data)
    
    # Calculate final metrics based on data type
    if data_type == 'regression':
        print(f"Processed {len(samples_data)} regression samples")
        print(f"Parseable predictions: {parseable_predictions}/{len(samples_data)} ({parseable_predictions/len(samples_data)*100 if samples_data else 0:.2f}%)")
        print(f"Valid predictions for MSE: {valid_prediction_count}")
        
        mse = total_squared_error / valid_prediction_count if valid_prediction_count > 0 else float('inf')
        print(f"Mean Squared Error (MSE): {mse:.6f}")
        
        # Set classification metrics to defaults for metadata
        accuracy = 0.0  # Not applicable for regression
        refined_accuracy = 0.0
        parseable_accuracy = 0.0
    elif data_type == 'classification':
        total_processed = len(samples_data)
        accuracy = (full_correct_predictions / total_processed) * 100 if total_processed > 0 else 0
        refined_accuracy = (full_refined_correct / total_processed) * 100 if total_processed > 0 else 0
        parseable_accuracy = (parseable_correct / parseable_predictions) * 100 if parseable_predictions > 0 else 0
        
        print(f"Overall accuracy: {accuracy:.2f}%")
        print(f"Refined accuracy: {refined_accuracy:.2f}%")
        print(f"Parseable accuracy: {parseable_accuracy:.2f}% ({parseable_predictions}/{total_processed} samples)")
        print(f"Unparseable predictions: {unparseable_predictions} ({unparseable_predictions/total_processed*100 if total_processed else 0:.2f}%)")
        
        # Set regression metrics to defaults for metadata
        mse = None
    else:
        raise NotImplementedError(f"Data type '{data_type}' not implemented")
    
    # Determine output file path
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = Path(input_file).parent
    
    json_output_file = output_dir / f"{Path(input_file).stem}_data.json"
    print(f"Output file will be: {json_output_file}")
    
    # Prepare JSON data with metadata and samples
    if data_type == 'regression' or data_type == 'classification':
        json_data = {
            "metadata": create_json_metadata(
                input_file=input_file,
                total_data_size=total_data_size,
                display_df_size=len(samples_data),
                data_type=data_type,
                mse=mse if data_type == 'regression' else None,
                valid_prediction_count=valid_prediction_count if data_type == 'regression' else None,
                parseable_predictions=parseable_predictions,
                accuracy=accuracy if data_type == 'classification' else None,
                refined_accuracy=refined_accuracy if data_type == 'classification' else None,
                parseable_accuracy=parseable_accuracy if data_type == 'classification' else None,
                unparseable_predictions=unparseable_predictions if data_type == 'classification' else None
            ),
            "samples": samples_data
        }
    else:
        # This should never happen as we've already validated the data_type above,
        # but included for completeness
        raise NotImplementedError(f"Data type '{data_type}' not implemented")
    
    with open(json_output_file, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, default=str)
    
    print(f"JSON data saved to: {json_output_file}")
    return str(json_output_file)

def main():
    parser = argparse.ArgumentParser(description='Extract data from parquet files and save to JSON')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to the input parquet file with model responses')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save the output file (default: same as input file)')
    parser.add_argument('--max-samples', type=int, default=0,
                        help='Maximum number of samples to include (default: 0 = all samples)')
    parser.add_argument('--data_type', type=str, default='regression',
                        choices=['regression', 'classification'],
                        help='Data type to extract')
    parser.add_argument('--display_llm_analysis', action='store_true', default=False,
                        help='Process and include LLM analysis in output (default: False)')
    
    args = parser.parse_args()      
    
    # Check if input file exists
    assert os.path.exists(args.input), f"File '{args.input}' not found"
    
    # Extract and save JSON
    output_file = extract_json_from_parquet(
        args.input, 
        args.output_dir, 
        args.max_samples, 
        args.data_type,
        args.display_llm_analysis
    )
    print(f"JSON extraction complete! Saved to: {output_file}")

if __name__ == "__main__":
    main() 