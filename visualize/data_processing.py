#!/usr/bin/env python3
"""
Helper functions for processing data fields in visualization tool
"""

import json
from typing import Dict, Any, List, Optional, Tuple


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