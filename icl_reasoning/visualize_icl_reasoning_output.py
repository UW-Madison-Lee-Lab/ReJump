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
from typing import Dict, Any, List, Optional, Tuple


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
    直接返回原始响应文本，不进行任何清理
    
    Args:
        response_text: 模型响应文本
        prompt_content: 输入提示内容 (未使用)
        
    Returns:
        原始响应文本
    """
    # 简单地过滤掉<|endoftext|>标记
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


def visualize_icl_reasoning_output(input_file: str, output_format: str = "txt", save_dir: Optional[str] = None):
    """
    Visualize ICL reasoning output from parquet files with model responses
    
    Args:
        input_file: Path to the input parquet file
        output_format: Output format, supports txt and html
        save_dir: Directory to save the output file (default: same as input file's directory)
        
    Returns:
        Path to the output file
    """
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
    
    # Statistics
    total_samples = len(df)
    correct_predictions = 0
    
    if output_format == "html":
        # HTML output
        html_content = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            "    <title>ICL Reasoning Visualization</title>",
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
            "    </style>",
            "</head>",
            "<body>",
            f"<h1>ICL Reasoning Results: {Path(input_file).name}</h1>",
        ]
        
        # We'll add the summary after processing all samples
        
        # Process each sample
        for idx, row in df.iterrows():
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
            
            # 过滤掉<|endoftext|>标记
            response_text = response_text.replace("<|endoftext|>", "")
            
            # 直接使用原始响应，不尝试移除prompt
            cleaned_response_text = clean_response_text(response_text, None)
            
            # 不再检查是否成功移除了prompt
            # prompt_removed = len(cleaned_response_text) < len(response_text)
            
            # 保留原始内容，不提取thinking和answer
            # We still extract for display purposes
            input_prompt_content = row.get('prompt')[0]['content']
            cleaned_response_text = cleaned_response_text.replace(input_prompt_content, "")
            raw_thinking = extract_think_content(cleaned_response_text)
            raw_answer = extract_answer_content(cleaned_response_text)
            
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
                correct_predictions += 1
            
            # Start building the sample HTML
            html_content.append(f'<div class="sample">')
            html_content.append(f'<h2>Sample {idx+1}</h2>')
            
            # Data source
            if 'data_source' in row:
                html_content.append(f'<div class="section">')
                html_content.append(f'<div class="section-title">Data Source</div>')
                html_content.append(f'<div>{row["data_source"]}</div>')
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
                                    # 转义HTML内容
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
                    # 转义HTML内容
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
            
            # Prediction result (仍然显示，因为这对用户有用)
            html_content.append(f'<div class="section">')
            html_content.append(f'<div class="section-title">Prediction Result</div>')
            if prediction is not None:
                result_class = "correct" if is_correct else "incorrect"
                html_content.append(f'<div class="{result_class}">Predicted: {prediction} ({("CORRECT" if is_correct else "INCORRECT")})</div>')
            else:
                html_content.append(f'<div class="incorrect">Unable to parse prediction</div>')
            html_content.append(f'</div>')
            
            # 恢复折叠式完整响应，但不提取标签内容
            html_content.append(f'<details open>')
            html_content.append(f'<summary>Model Response (Cleaned)</summary>')
            html_content.append(f'<div class="section">')
            # 使用html.escape确保标签正确显示，不被浏览器解释为HTML标签
            escaped_response = html.escape(cleaned_response_text)
            html_content.append(f'<div class="response" style="white-space: pre-wrap; font-family: monospace;">{escaped_response}</div>')
            html_content.append(f'</div>')
            html_content.append(f'</details>')
            
            # End sample div
            html_content.append(f'</div>')
        
        # Add summary before all samples
        accuracy = (correct_predictions / total_samples) * 100 if total_samples > 0 else 0
        summary_html = [
            f'<div class="summary">',
            f'<h2>Results Summary</h2>',
            f'<table>',
            f'<tr><th>Metric</th><th>Value</th></tr>',
            f'<tr><td>Total Samples</td><td>{total_samples}</td></tr>',
            f'<tr><td>Correct Predictions</td><td>{correct_predictions}</td></tr>',
            f'<tr><td>Accuracy</td><td>{accuracy:.2f}%</td></tr>',
            f'</table>',
            f'</div>'
        ]
        
        # Insert summary after the title
        html_content = html_content[:8] + summary_html + html_content[8:]
        
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
            for idx, row in df.iterrows():
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
                
                # Get ground truth label
                ground_truth = None
                if 'reward_model' in row and isinstance(row['reward_model'], dict):
                    ground_truth_data = row['reward_model'].get('ground_truth', None)
                    if isinstance(ground_truth_data, dict) and 'label' in ground_truth_data:
                        ground_truth = ground_truth_data
                
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
                
                # 过滤掉<|endoftext|>标记
                response_text = response_text.replace("<|endoftext|>", "")
                
                # 直接使用原始响应，不尝试移除prompt
                cleaned_response_text = clean_response_text(response_text, None)
                
                # 不再检查是否成功移除了prompt
                # prompt_removed = len(cleaned_response_text) < len(response_text)
                
                # 保留原始内容，不提取thinking和answer
                # We still extract for display purposes
                raw_thinking = extract_think_content(cleaned_response_text)
                raw_answer = extract_answer_content(cleaned_response_text)
                
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
                    correct_predictions += 1
                
                # Write prediction result
                f.write("--- Prediction Result ---\n")
                if prediction is not None:
                    result_str = "CORRECT" if is_correct else "INCORRECT"
                    f.write(f"Predicted: {prediction} ({result_str})\n")
                else:
                    f.write("Unable to parse prediction\n")
                f.write("\n")
                
                # Write full response (不提取标签内容)
                f.write(f"--- Model Response (Cleaned) ---\n")
                f.write(f"{cleaned_response_text}\n")
                f.write("\n")
                
                # Write separator
                f.write("="*80 + "\n\n")
            
            # Write summary at the end
            accuracy = (correct_predictions / total_samples) * 100 if total_samples > 0 else 0
            f.write("="*80 + "\n")
            f.write("SUMMARY\n")
            f.write("="*80 + "\n")
            f.write(f"Total samples: {total_samples}\n")
            f.write(f"Correct predictions: {correct_predictions}\n")
            f.write(f"Accuracy: {accuracy:.2f}%\n")
            f.write("="*80 + "\n")
    
    print(f"Visualization saved to: {output_file}")
    print(f"Total samples: {total_samples}")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Accuracy: {accuracy:.2f}%")
    
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
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: File '{args.input}' not found")
        return
    
    # Visualize the output
    output_file = visualize_icl_reasoning_output(args.input, args.format, args.output_dir)
    print(f"Visualization complete! Saved to: {output_file}")


if __name__ == "__main__":
    main() 