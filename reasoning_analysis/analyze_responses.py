#!/usr/bin/env python3
import pandas as pd
import os
import json
import re
import time
import argparse
import logging
import numpy as np
import tiktoken
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Any, Optional, Union, List
import concurrent.futures

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

# def process_reasoning(input: str) -> str:
#     #split the input by '\n' and concate them with their index
#     lines = input.split('\n')
#     #remove empty lines
#     lines = [line for line in lines if line.strip()]
#     processed_lines = []
#     reasoning_dict = {}
#     for i, line in enumerate(lines):
#         step_key = i # Use integer index as key
#         reasoning_dict[step_key] = line
#         processed_lines.append(f"[Step {i}]: {line}")
#     return '\n'.join(processed_lines), reasoning_dict

# def process_reasoning(input_str: str) -> tuple[str, dict]:

#     if not input_str:
#         return "", {}
        
#     parts = re.split(r'([.?!])', input_str)
    
#     sentences = []
#     current_sentence = ""
#     for part in parts:
#         # Skip empty parts that can result from the split
#         if not part: 
#             continue
        
#         # Append the current part (either text or punctuation)
#         current_sentence += part
        
#         # If the part is a punctuation mark, the sentence is complete
#         if part in ['.', '?', '!']:
#             sentence_strip = current_sentence.strip()
#             if sentence_strip: # Avoid adding empty strings if there's only whitespace
#                  sentences.append(sentence_strip)
#             # Reset for the next sentence
#             current_sentence = ""

#     # Add any remaining part if the input doesn't end with punctuation
#     remaining_strip = current_sentence.strip()
#     if remaining_strip:
#         sentences.append(remaining_strip)

#     # Filter out potentially empty strings again (e.g., from consecutive delimiters)
#     sentences = [s for s in sentences if s]

#     processed_lines = []
#     reasoning_dict = {}
#     for i, sentence in enumerate(sentences):
#         step_key = i # Use integer index as key
#         reasoning_dict[step_key] = sentence
#         processed_lines.append(f"[Step {i}]: {sentence}")
        
#     # Return the formatted string and the dictionary
#     return '\n'.join(processed_lines), reasoning_dict

def process_reasoning_with_probs(probs_dict: Dict[str, Any], reasoning_only: bool = True) -> tuple[str, Dict[int, Dict[str, Any]]]:
    """
    Process reasoning with probability information, splitting by punctuation
    
    Args:
        probs_dict: Dictionary containing logprobs and tokens
        reasoning_only: Whether to only include reasoning (exclude answer)
        
    Returns:
        Tuple of (formatted_text, reasoning_dict) where:
            - formatted_text is a string with line breaks between sentences
            - reasoning_dict has sentence-level statistics including logprobs
    """
    logprobs_list = probs_dict['logprobs']
    tokens_list = probs_dict['tokens']
    assert len(logprobs_list) == len(tokens_list)
    logprob_token_pairs = list(zip(logprobs_list, tokens_list))

    if reasoning_only:
        # Remove the tokens that are after "<","answer",">", these are 3 consecutive tokens
        # When this pattern is found, remove the 3 tokens and the following tokens
        for i, (logprob, token) in enumerate(logprob_token_pairs):
            if i+2 < len(logprob_token_pairs) and token == "<" and logprob_token_pairs[i+1][1] == "answer" and logprob_token_pairs[i+2][1] == ">":
                logprob_token_pairs = logprob_token_pairs[:i]
                break
    
    # Remove "<th","ink",">" the 3 consecutive tokens, also remove ":<\/","think",">\n\n" the 3 consecutive tokens
    # In other words, remove only the 6 tokens and the corresponding logprobs. Keep the rest of the tokens and logprobs.
    # i = 0
    # while i < len(logprob_token_pairs) - 2:
    #     if (logprob_token_pairs[i][1] == "<th" and 
    #         i+2 < len(logprob_token_pairs) and
    #         logprob_token_pairs[i+1][1] == "ink" and 
    #         logprob_token_pairs[i+2][1] == ">"):
    #         logprob_token_pairs = logprob_token_pairs[:i] + logprob_token_pairs[i+3:]
    #         continue
    #     elif ( (logprob_token_pairs[i][1] == ":<\/" or logprob_token_pairs[i][1] == "<\/") and 
    #          i+2 < len(logprob_token_pairs) and
    #          logprob_token_pairs[i+1][1] == "think" and 
    #          logprob_token_pairs[i+2][1] == ">\n\n"):
    #         logprob_token_pairs = logprob_token_pairs[:i] + logprob_token_pairs[i+3:]
    #         continue
    #     i += 1
    
    
    # Split text into sentences based on punctuation
    # Use regex to keep punctuation with the sentence
    sentences = []
    current_sentence_tokens = []
    current_sentence_logprobs = []
    current_sentence_text = ""
    
    for i, (logprob, token) in enumerate(logprob_token_pairs):
        current_sentence_tokens.append(token)
        current_sentence_logprobs.append(logprob)
        current_sentence_text += token
        
        # Check if the token ends with punctuation (.!?)
        if token in ['.\n', ".\n\n", "\n\n", " \n\n",'!', '?']:
            if current_sentence_text:
                sentences.append({
                    'text': current_sentence_text,
                    'tokens': current_sentence_tokens,
                    'logprobs': current_sentence_logprobs
                })
            # Reset for next sentence
            current_sentence_tokens = []
            current_sentence_logprobs = []
            current_sentence_text = ""
    
    # Add the last sentence if it's not empty (might not end with punctuation)
    if current_sentence_text:
        sentences.append({
            'text': current_sentence_text,
            'tokens': current_sentence_tokens,
            'logprobs': current_sentence_logprobs
        })
    
    # Format the output and create reasoning dictionary
    processed_lines = []
    reasoning_dict = {}
    
    for i, sentence_info in enumerate(sentences):
        step_key = i
        sentence_text = sentence_info['text']
        sentence_tokens = sentence_info['tokens']
        sentence_logprobs = sentence_info['logprobs']
        
        # Calculate exact token count using tiktoken
        n_tokens = len(sentence_tokens)
        
        # Create the dictionary entry for this sentence
        reasoning_dict[step_key] = {
            'sentence': sentence_text,
            'sentence_tokens': sentence_tokens,
            'logprobs_list': sentence_logprobs,
            'avg_logprob': sum(sentence_logprobs) / n_tokens,
            'avg_prob': np.exp(sum(sentence_logprobs) / n_tokens),
            'n_tokens': n_tokens
        }
        
        # Format the line for the string output
        processed_lines.append(f"[Step {i}]: {sentence_text}")
    
    # Return the formatted string and the dictionary
    return '\n'.join(processed_lines), reasoning_dict

def process_row(args):
    """
    Process a single row from the dataframe using LLM API
    
    Args:
        args: Tuple containing (row_idx, row_data, instruction, analyzer, column_prefix, continue_on_error, field_of_interests)
        
    Returns:
        Tuple of (row_idx, raw_output, extracted_json, total_input, reasoning_dict, error)
    """
    row_idx, row, instruction, analyzer, column_prefix, continue_on_error, field_of_interests = args
    
    try:
        # Handle responses based on its type (could be string or list)
        # reasonings_str, reasoning_dict = process_reasoning(row['reasonings'][0])
        reasonings_str, reasoning_dict = process_reasoning_with_probs(row['probs'][0])
        input_prompt = row['prompt'][0]['content']
        responses = row['responses'][0]

        if field_of_interests == 'input+reasonings':
            # Concatenate 'prompt' and 'reasonings' fields
            total_input = 'Input: ' + input_prompt + '\n' + 'Reasoning: ' + reasonings_str
        elif field_of_interests == 'reasonings':
            total_input = reasonings_str
        elif field_of_interests == 'responses':
            total_input = responses
        else:
            raise NotImplementedError(f"Field of interests {field_of_interests} not implemented")
        
        if "<INSERT MODEL OUTPUT TRANSCRIPT HERE>" in instruction:
            prompt = instruction.replace("<INSERT MODEL OUTPUT TRANSCRIPT HERE>", total_input)
        else:
            prompt = instruction + total_input
        
        # Call LLM API with instruction + response
        raw_output = analyzer.analyze_response(prompt)
        
        # Extract JSON
        extracted_json = extract_json(raw_output)
        
        return row_idx, raw_output, json.dumps(extracted_json), total_input, reasoning_dict, None
    except Exception as e:
        error_msg = f"Error processing row {row_idx}: {str(e)}"
        logger.error(error_msg)
        
        if not continue_on_error:
            return row_idx, None, None, None, None, error_msg
        else:
            logger.warning(f"Continuing due to --continue-on-error flag for row {row_idx}...")
            return row_idx, f"ERROR: {str(e)}", "{}", None, {}, None


def process_file(
    input_file: str,
    output_file: str,
    instruction_file: str,
    llm_type: str = "gemini",
    temperature: float = 0.8,
    max_tokens: int = 40000,
    max_retries: int = 5,
    delay: int = 1,
    continue_on_error: bool = False,
    batch_size: int = 10,
    field_of_interests: str = "responses",
    debug: bool = False
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
        batch_size: Size of batches for parallel processing and saving
        field_of_interests: The field name in the dataframe to use as the response (default: 'responses')
        debug: Whether to run in debug mode (limits to 5 rows)
    """
    logger.info(f"Reading input file: {input_file}")
    df = pd.read_parquet(input_file)
    
    if debug:
        logger.info("Running in debug mode - using only first 5 rows")
        df = df.head(5)
        batch_size = min(batch_size, 5)
    
    # Read the instruction file
    logger.info(f"Reading instruction file: {instruction_file}")
    instruction = read_instruction_file(instruction_file)
    
    # Create analyzer
    analyzer = ResponseAnalyzer(llm_type, temperature, max_tokens, max_retries)
    analyzer.set_rate_limit(delay)  # Set delay between API calls
    
    # Create new columns for analysis results
    column_prefix = "llm_analysis"
    df[f'{column_prefix}_llm_type'] = llm_type
    df[f'{column_prefix}_raw_output'] = None
    df[f'{column_prefix}_extracted_json'] = None
    df[f'{column_prefix}_total_input'] = None  # Add new column for total_input
    df[f'{column_prefix}_splitted_reasoning_dict'] = None # Add new column for reasoning dict
    
    logger.info(f"Processing {len(df)} rows using {llm_type} API with batch size {batch_size}")
    
    # Create progress bar for overall progress
    pbar = tqdm(total=len(df), desc="Overall Progress")
    processed_count = 0
    
    # Process in batches using ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
        # Process all rows in batches
        for batch_start in range(0, len(df), batch_size):
            batch_end = min(batch_start + batch_size, len(df))
            logger.info(f"Processing batch: rows {batch_start} to {batch_end-1}")
            
            # Create list of tasks for this batch
            tasks = [
                (i, df.iloc[i], instruction, analyzer, column_prefix, continue_on_error, field_of_interests)
                for i in range(batch_start, batch_end)
            ]
            
            # Submit all tasks for this batch
            future_to_idx = {executor.submit(process_row, task): task[0] for task in tasks}
            
            # Process completed tasks as they finish
            batch_errors = []
            for future in concurrent.futures.as_completed(future_to_idx):
                row_idx, raw_output, extracted_json, total_input, reasoning_dict, error = future.result()
                
                if error and not continue_on_error:
                    batch_errors.append(error)
                    continue
                    
                df.at[row_idx, f'{column_prefix}_raw_output'] = raw_output
                df.at[row_idx, f'{column_prefix}_extracted_json'] = extracted_json
                df.at[row_idx, f'{column_prefix}_total_input'] = total_input  # Save total_input
                # Store the reasoning dictionary as a JSON string
                df.at[row_idx, f'{column_prefix}_splitted_reasoning_dict'] = json.dumps(reasoning_dict) 
                
                # Update progress bar
                processed_count += 1
                pbar.update(1)
            
            # Check if we had any errors and should stop
            if batch_errors and not continue_on_error:
                pbar.close()
                logger.error("Stopping due to errors. Use --continue-on-error flag to continue despite errors.")
                raise RuntimeError(batch_errors[0])
                
            # Save after each batch
            logger.info(f"Saving results after processing batch up to row {batch_end-1}...")
            df.to_parquet(output_file)
            
            # In debug mode, also save as JSON
            if debug:
                json_output_file = str(Path(output_file).with_suffix('.json'))
                logger.info(f"Debug mode: Also saving results to JSON file: {json_output_file}")
                # Convert DataFrame to JSON with records orientation and indent for readability
                df.to_json(json_output_file, orient='records', indent=2, force_ascii=False)
    
    # Close progress bar
    pbar.close()
    logger.info(f"Analysis complete using {llm_type} API")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Analyze model responses using LLM APIs')
    
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Path to input parquet file')
    
    parser.add_argument('--output', '-o', type=str,
                        help='Path to output parquet file (default: auto-generated)')
    
    parser.add_argument('--instruction', '-p', type=str, 
                        help='Path to instruction file')
    
    parser.add_argument('--llm', '-l', type=str, default='gemini',
                        choices=get_available_llm_types(),
                        help=f'LLM API to use for analysis')
    
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
    
    parser.add_argument('--batch_size', '-b', type=int, default=10,
                        help='Batch size for parallel processing and saving (default: 10)')
    
    parser.add_argument('--output_suffix', '-s', type=str, default='',
                        help='Suffix to append to the output filename (default: empty)')
    
    parser.add_argument('--field_of_interests', '-f', type=str, default='responses',
                        help="Field name in the dataframe to use as the response (default: 'responses')")
    
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose logging')
    
    parser.add_argument('--debug', '-D', action='store_true',
                        help='Run in debug mode (limits to 5 rows)')
    
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
        suffix = args.output_suffix
        if args.debug:
            suffix = '_debug' + suffix
        output_path = input_path.parent / f"{input_path.stem}_{args.llm}_analysis{suffix}.parquet"
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
        continue_on_error=args.continue_on_error,
        batch_size=args.batch_size,
        field_of_interests=args.field_of_interests,
        debug=args.debug
    )


if __name__ == "__main__":
    main() 