#!/usr/bin/env python3
import os
import json
import argparse
import re
from typing import Dict, Any, List, Optional, Tuple
import logging
from pathlib import Path
import concurrent.futures
from tqdm import tqdm
import time

from llm_apis import ResponseAnalyzer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def read_prompt(prompt_path: str) -> str:
    """
    Read the classification prompt from file
    
    Args:
        prompt_path: Path to the prompt file
        
    Returns:
        The prompt content as a string
    """
    with open(prompt_path, 'r') as f:
        return f.read()

def read_json_file(file_path: str) -> Dict[str, Any]:
    """
    Read a JSON file
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Parsed JSON content as dictionary
    """
    with open(file_path, 'r') as f:
        return json.load(f)

def save_json_file(data: Dict[str, Any], file_path: str) -> None:
    """
    Save dictionary as JSON file
    
    Args:
        data: Dictionary to save
        file_path: Path where to save the JSON file
    """
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

def extract_json_from_text(text: str) -> Dict[str, Any]:
    """
    Extract JSON from text response, handling various formats:
    1. JSON within ```json code blocks
    2. JSON within curly braces {}
    
    Args:
        text: Text containing JSON
        
    Returns:
        Extracted JSON dictionary or empty dict if not found
    """
    if text is None:
        return {}
    
    try:
        # Try to find JSON in code blocks with ```json
        code_block_pattern = r'```(?:json)?\s*\n?([\s\S]*?)\n?```'
        code_blocks = re.findall(code_block_pattern, text, re.DOTALL)
        
        # Check each code block for valid JSON
        for block in code_blocks:
            block = block.strip()
            try:
                # Try parsing the block directly
                return json.loads(block)
            except:
                # If it fails, try to find JSON object inside the block
                obj_match = re.search(r'(\{[\s\S]*\})', block, re.DOTALL)
                if obj_match:
                    try:
                        return json.loads(obj_match.group(1))
                    except:
                        continue  # Try next block
        
        # If no valid JSON in code blocks, search directly in the text
        # Look for JSON objects
        obj_pattern = r'(\{[\s\S]*?\})'
        obj_matches = re.finditer(obj_pattern, text, re.DOTALL)
        
        # Try each match from longest to shortest
        matches = sorted([(m.group(1), len(m.group(1))) for m in obj_matches], 
                        key=lambda x: x[1], reverse=True)
        
        for match_text, _ in matches:
            try:
                return json.loads(match_text)
            except:
                continue
        
        # Last resort: try to construct JSON from the text
        # Look for patterns like "category": "linear"
        category_match = re.search(r'"category"\s*:\s*"([^"]+)"', text)
        if category_match:
            category = category_match.group(1)
            # Construct a simple valid JSON with just the category
            return {"category": category, "detail": {}}
        
        # If everything fails, return empty structure
        logger.warning("No valid JSON found in the text")
        return {"category": "error", "detail": {}, "error": "Failed to extract JSON"}
        
    except Exception as e:
        logger.error(f"Error extracting JSON: {e}")
        return {"category": "error", "detail": {}, "error": str(e)}

def classify_model_function(model_code: str, prompt_template: str, analyzer: ResponseAnalyzer) -> Tuple[Dict[str, Any], str]:
    """
    Classify a model function using LLM API
    
    Args:
        model_code: The Python function code to classify
        prompt_template: The prompt template to use for classification
        analyzer: LLM API analyzer instance
        
    Returns:
        Tuple of (classification result as dictionary, raw LLM response)
    """
    # Format the prompt with the model code
    full_prompt = f"{prompt_template}\n\nClassify this function:\n```python\n{model_code}\n```"
    
    # Get classification from LLM
    try:
        llm_response = analyzer.analyze_response(full_prompt)
        
        # Extract JSON from the response
        classification = extract_json_from_text(llm_response)
        
        # If we got valid classification, return it along with raw response
        if classification and "category" in classification:
            return classification, llm_response
        
        # If the extraction failed, return an error along with raw response
        logger.error(f"Failed to extract valid classification from response: {llm_response[:300]}...")
        return {"category": "error", "detail": {}, "error": "Failed to extract valid classification"}, llm_response
    except Exception as e:
        logger.error(f"Error calling LLM API: {str(e)}")
        return {"category": "error", "detail": {}, "error": str(e)}, ""

def process_model(args: Tuple) -> Tuple:
    """
    Process a single model and classify its function
    
    Args:
        args: Tuple containing (model_info, sample_index, model_index, prompt, analyzer)
        
    Returns:
        Tuple of (sample_index, model_index, classification, raw_response)
    """
    model_info, sample_index, model_index, prompt, analyzer = args
    
    try:
        if "model_code" in model_info and not "py_function_type" in model_info:
            model_code = model_info["model_code"]
            classification, raw_response = classify_model_function(model_code, prompt, analyzer)
            return sample_index, model_index, classification, raw_response
        return sample_index, model_index, None, None
    except Exception as e:
        logger.error(f"Error processing model at sample {sample_index}, model {model_index}: {str(e)}")
        return sample_index, model_index, {"category": "error", "detail": {}, "error": str(e)}, ""

def process_json_file(input_file: str, output_file: str, prompt_path: str, llm_type: str = "claude", 
                      max_workers: int = 5, delay: float = 0.5) -> None:
    """
    Process a JSON file containing model codes, classify them, and save the results
    
    Args:
        input_file: Path to the input JSON file
        output_file: Path to save the processed JSON file
        prompt_path: Path to the classification prompt file
        llm_type: Type of LLM to use (default: "claude")
        max_workers: Maximum number of worker threads (default: 5)
        delay: Delay between API calls in seconds (default: 0.5)
    """
    # Read the input JSON file
    logger.info(f"Reading input file: {input_file}")
    data = read_json_file(input_file)
    
    # Read the classification prompt
    logger.info(f"Reading prompt from: {prompt_path}")
    prompt = read_prompt(prompt_path)
    
    # Check if the JSON file has the expected structure
    if "samples" not in data:
        logger.error("Input JSON does not contain 'samples' field")
        return
    
    # Initialize the LLM client
    analyzer = ResponseAnalyzer(llm_type=llm_type, temperature=0.1, max_tokens=1000)
    analyzer.set_rate_limit(delay)  # Set rate limit to avoid API rate limiting
    
    # Collect all models that need classification
    tasks = []
    for sample_idx, sample in enumerate(data["samples"]):
        if "model_evaluation_table" in sample:
            for model_idx, model in enumerate(sample["model_evaluation_table"]):
                if "model_code" in model and not "py_function_type" in model:
                    tasks.append((model, sample_idx, model_idx, prompt, analyzer))
    
    logger.info(f"Found {len(tasks)} models to classify using {max_workers} workers")
    
    # No models to process
    if not tasks:
        logger.info("No models found for classification. Saving unchanged data.")
        save_json_file(data, output_file)
        return
    
    # Add metadata to track progress
    if "metadata" not in data:
        data["metadata"] = {}
    if "classification_progress" not in data["metadata"]:
        data["metadata"]["classification_progress"] = {
            "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_models": len(tasks),
            "processed_models": 0,
            "current_batch": 0
        }
    
    # Process models in parallel, batch by batch
    total_processed = data["metadata"]["classification_progress"]["processed_models"]
    batch_number = data["metadata"]["classification_progress"]["current_batch"] + 1
    
    # Break tasks into batches of max_workers size
    for batch_start in range(0, len(tasks), max_workers):
        # Get batch of tasks
        batch_end = min(batch_start + max_workers, len(tasks))
        batch_tasks = tasks[batch_start:batch_end]
        logger.info(f"Processing batch {batch_number}: tasks {batch_start+1} to {batch_end} (total: {len(batch_tasks)})")
        
        # Process this batch
        batch_results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Create a progress bar for this batch
            with tqdm(total=len(batch_tasks), desc=f"Batch {batch_number}") as pbar:
                # Submit all tasks for this batch
                futures = [executor.submit(process_model, task) for task in batch_tasks]
                
                # Process completed tasks as they finish
                for future in concurrent.futures.as_completed(futures):
                    sample_idx, model_idx, classification, raw_response = future.result()
                    if classification:
                        batch_results.append((sample_idx, model_idx, classification, raw_response))
                    pbar.update(1)
        
        # Update the data with classifications from this batch
        for sample_idx, model_idx, classification, raw_response in batch_results:
            data["samples"][sample_idx]["model_evaluation_table"][model_idx]["py_function_type"] = classification
            data["samples"][sample_idx]["model_evaluation_table"][model_idx]["llm_raw_classification_response"] = raw_response
        
        # Update progress metadata
        total_processed += len(batch_results)
        data["metadata"]["classification_progress"]["processed_models"] = total_processed
        data["metadata"]["classification_progress"]["current_batch"] = batch_number
        data["metadata"]["classification_progress"]["last_update"] = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Save to the output file after each batch
        logger.info(f"Saving results after batch {batch_number} (processed {len(batch_results)} models in this batch, {total_processed} total) to: {output_file}")
        save_json_file(data, output_file)
        
        # Increment batch number
        batch_number += 1
    
    # Update completion metadata
    data["metadata"]["classification_progress"]["completion_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    data["metadata"]["classification_progress"]["status"] = "completed"
    
    # Save the final processed data
    logger.info(f"All batches completed. Processed {total_processed} models in total. Final results saved to: {output_file}")
    save_json_file(data, output_file)

def main():
    """Main function to handle command-line arguments and run the classification"""
    parser = argparse.ArgumentParser(description="Classify model functions in a JSON file using LLM API")
    parser.add_argument("--input", required=True, help="Path to the input JSON file")
    parser.add_argument("--output", help="Path to save the processed JSON file (default: input file with '_classified' suffix)")
    parser.add_argument("--prompt", default="/home/szhang967/liftr/reasoning_analysis/classifier_prompt_classfication.txt", 
                        help="Path to the classification prompt file")
    parser.add_argument("--llm", default="claude", choices=["openai", "claude", "gemini", "deepseek"],
                        help="LLM API to use for classification (default: claude)")
    parser.add_argument("--workers", type=int, default=500, 
                        help="Maximum number of parallel workers (default: 1000)")
    parser.add_argument("--delay", type=float, default=0.5, 
                        help="Delay between API calls in seconds (default: 0.5)")
    
    args = parser.parse_args()
    
    # Set default output file if not provided
    if not args.output:
        input_path = Path(args.input)
        # Create default output filename with _classified suffix
        output_filename = f"{input_path.stem}_classified{input_path.suffix}"
        args.output = str(input_path.parent / output_filename)
    
    # Process the JSON file
    process_json_file(
        input_file=args.input,
        output_file=args.output,
        prompt_path=args.prompt,
        llm_type=args.llm,
        max_workers=args.workers,
        delay=args.delay
    )
    logger.info("Classification completed successfully")

if __name__ == "__main__":
    main() 