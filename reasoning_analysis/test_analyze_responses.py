#!/usr/bin/env python3
import pandas as pd
import os
import argparse
import logging
from pathlib import Path

# Import from llm_apis module
from llm_apis import test_functions, get_available_llm_types

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_read_parquet(input_file):
    """Test if the parquet file can be read and verify it has 'responses' field"""
    try:
        df = pd.read_parquet(input_file)
        logger.info(f"Successfully read parquet file with {len(df)} rows")
        
        # Check if 'responses' field exists
        if 'responses' not in df.columns:
            logger.error("ERROR: 'responses' field not found in the parquet file")
            return False
        
        # Print first row's response to verify structure
        logger.info("\nSample data:")
        
        response = df['responses'].iloc[0]
        if isinstance(response, list) and len(response) > 0:
            response = response[0]
            logger.info(f"Response (first item in list): {response[:100]}...")
        else:
            logger.info(f"Response: {response[:100]}...")
        
        return True
    except Exception as e:
        logger.error(f"ERROR reading parquet file: {e}")
        return False

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Test environment for analyzing model responses')
    
    parser.add_argument('--input', '-i', type=str, 
                        default="/home/szhang967/liftr/reasoning_analysis/deepseek-zeroshot.parquet",
                        help='Path to input parquet file')
    
    parser.add_argument('--llm', '-l', type=str, default='all',
                        choices=['all'] + get_available_llm_types(),
                        help='LLM API to test (default: all)')
    
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose logging')
    
    return parser.parse_args()

def main():
    """Main function"""
    args = parse_arguments()
    
    # Set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    logger.info("Running tests for analyze_responses.py")
    
    # Test reading the parquet file
    parquet_ok = test_read_parquet(args.input)
    
    # Test LLM clients
    llm_ok = True
    if args.llm == 'all':
        # Test all LLM clients
        for llm_name, test_func in test_functions.items():
            logger.info(f"\nTesting {llm_name} client...")
            if not test_func():
                llm_ok = False
                logger.warning(f"{llm_name} client test failed")
    else:
        # Test only the specified LLM client
        logger.info(f"\nTesting {args.llm} client...")
        if not test_functions[args.llm]():
            llm_ok = False
            logger.warning(f"{args.llm} client test failed")
    
    # Summarize test results
    if parquet_ok and llm_ok:
        logger.info("\nAll tests PASSED. You can now run analyze_responses.py")
    else:
        logger.error("\nSome tests FAILED. Please fix the issues before running analyze_responses.py")
        if not parquet_ok:
            logger.error("- Parquet file test failed")
        if not llm_ok:
            logger.error("- LLM client test(s) failed")

if __name__ == "__main__":
    main() 