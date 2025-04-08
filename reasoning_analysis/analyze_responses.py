#!/usr/bin/env python3
import pandas as pd
import os
import json
import re
import time
import argparse
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Any, Optional, Union, List

# Import from llm_apis module
from llm_apis import ResponseAnalyzer, get_available_llm_types

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_json(text: str) -> Any:
    """
    Extract JSON from text response, handling various formats:
    1. JSON within ```json code blocks
    2. JSON within square brackets []
    3. JSON within curly braces {}
    
    Args:
        text: Text containing JSON
        
    Returns:
        Extracted JSON (list, dict, etc.) or empty list/dict if not found
    """
    if text is None:
        return []
    
    try:
        # Try to find JSON array or object in code blocks with ```json
        code_block_pattern = r'```(?:json)?\s*\n?([\s\S]*?)\n?```'
        code_blocks = re.findall(code_block_pattern, text, re.DOTALL)
        
        # Check each code block for valid JSON
        for block in code_blocks:
            block = block.strip()
            try:
                # Try parsing the block directly
                return json.loads(block)
            except:
                # If it fails, try to find JSON array inside the block
                array_match = re.search(r'(\[[\s\S]*\])', block, re.DOTALL)
                if array_match:
                    try:
                        return json.loads(array_match.group(1))
                    except:
                        continue  # Try next block
                
                # Try to find JSON object inside the block
                obj_match = re.search(r'(\{[\s\S]*\})', block, re.DOTALL)
                if obj_match:
                    try:
                        return json.loads(obj_match.group(1))
                    except:
                        continue  # Try next block
        
        # If no valid JSON in code blocks, search directly in the text
        # Look for JSON arrays
        array_pattern = r'(\[[\s\S]*?\])'
        array_matches = re.finditer(array_pattern, text, re.DOTALL)
        
        # Try each match from longest to shortest (assuming the longest is the complete JSON)
        matches = sorted([(m.group(1), len(m.group(1))) for m in array_matches], 
                         key=lambda x: x[1], reverse=True)
        
        for match_text, _ in matches:
            try:
                return json.loads(match_text)
            except:
                continue
        
        # Look for JSON objects if no arrays found
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
        
        # If everything fails, return empty structure
        logger.warning("No valid JSON found in the text")
        return []
        
    except Exception as e:
        logger.error(f"Error extracting JSON: {e}")
        return []


def read_instruction_file(file_path: str) -> str:
    """
    Read the instruction from a file
    
    Args:
        file_path: Path to the instruction file
        
    Returns:
        Content of the instruction file
    """
    try:
        with open(file_path, 'r') as f:
            return f.read()
    except Exception as e:
        logger.error(f"Error reading instruction file: {e}")
        raise ValueError(f"Could not read instruction file: {file_path}")


def process_file(
    input_file: str,
    output_file: str,
    instruction_file: str,
    llm_type: str = "openai",
    temperature: float = 0.8,
    max_tokens: int = 40000,
    max_retries: int = 5,
    delay: int = 1,
    continue_on_error: bool = False
) -> None:
    """
    Process a parquet file and analyze responses
    
    Args:
        input_file: Path to input parquet file
        output_file: Path to output parquet file
        instruction_file: Path to instruction file
        llm_type: Type of LLM API to use
        temperature: Temperature for generation
        max_tokens: Maximum tokens for response
        max_retries: Maximum number of retry attempts for API calls
        delay: Delay between API calls in seconds
        continue_on_error: Whether to continue processing after an error
    """
    logger.info(f"Reading input file: {input_file}")
    df = pd.read_parquet(input_file)
    
    # Read the instruction file
    logger.info(f"Reading instruction file: {instruction_file}")
    instruction = read_instruction_file(instruction_file)
    
    # Create analyzer
    analyzer = ResponseAnalyzer(llm_type, temperature, max_tokens, max_retries)
    
    # Create new columns for analysis results
    column_prefix = f"{llm_type}_analysis"
    df[f'{column_prefix}_raw_output'] = None
    df[f'{column_prefix}_extracted_json'] = None
    
    logger.info(f"Processing {len(df)} rows using {llm_type} API")
    
    # Process each row
    for i, row in tqdm(df.iterrows(), total=len(df)):
        try:
            # Handle responses based on its type (could be string or list)
            response = row['responses'][0]
            # Create the prompt by appending the response to the instruction
            # Replace the placeholder with the actual response
            prompt = instruction.replace("<INSERT MODEL OUTPUT TRANSCRIPT HERE>", response)
            
            # Call LLM API with instruction + response
            raw_output = analyzer.analyze_response(prompt)
            df.at[i, f'{column_prefix}_raw_output'] = raw_output
            
            # Extract JSON
            extracted_json = extract_json(raw_output)
            df.at[i, f'{column_prefix}_extracted_json'] = json.dumps(extracted_json)
            
            # Save intermediate results after each successful processing
            if i > 0 and i % 5 == 0:
                logger.info(f"Saving intermediate results after processing {i} rows...")
                df.to_parquet(output_file)
                
        except Exception as e:
            error_msg = f"Error processing row {i}: {str(e)}"
            logger.error(error_msg)
            
            if not continue_on_error:
                logger.error("Stopping due to error. Use --continue-on-error flag to continue despite errors.")
                raise RuntimeError(error_msg)
            else:
                logger.warning("Continuing to next row due to --continue-on-error flag...")
                df.at[i, f'{column_prefix}_raw_output'] = f"ERROR: {str(e)}"
                df.at[i, f'{column_prefix}_extracted_json'] = "{}"

        
    

    
    logger.info(f"Analysis complete using {llm_type} API")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Analyze model responses using LLM APIs')
    
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Path to input parquet file')
    
    parser.add_argument('--output', '-o', type=str,
                        help='Path to output parquet file (default: auto-generated)')
    
    parser.add_argument('--instruction', '-p', type=str, 
                        default="/home/szhang967/liftr/reasoning_analysis/fitting_model_extraction_prompt.txt",
                        help='Path to instruction file')
    
    parser.add_argument('--llm', '-l', type=str, default='openai',
                        choices=get_available_llm_types(),
                        help=f'LLM API to use for analysis (default: openai)')
    
    parser.add_argument('--temperature', '-t', type=float, default=0.3,
                        help='Temperature for LLM generation (default: 0.3)')
    
    parser.add_argument('--max_tokens', '-m', type=int, default=20000,
                        help='Maximum tokens for LLM response (default: 20000)')
    
    parser.add_argument('--max_retries', '-r', type=int, default=10,
                        help='Maximum number of retry attempts for API calls (default: 10)')
    
    parser.add_argument('--delay', '-d', type=int, default=1,
                        help='Delay between API calls in seconds (default: 1)')
    
    parser.add_argument('--continue_on_error', '-c', action='store_true',
                        help='Continue processing after errors (default: False)')
    
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose logging')
    
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_arguments()
    
    # Set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Create output file path if not provided
    if args.output is None:
        input_path = Path(args.input)
        output_path = input_path.parent / f"{input_path.stem}_{args.llm}_analysis.parquet"
        args.output = str(output_path)
    
    # Process file
    process_file(
        input_file=args.input,
        output_file=args.output,
        instruction_file=args.instruction,
        llm_type=args.llm,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_retries=args.max_retries,
        delay=args.delay,
        continue_on_error=args.continue_on_error
    )


if __name__ == "__main__":
    main() 