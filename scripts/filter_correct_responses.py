import pandas as pd
import argparse
import numpy as np
from verl.utils.reward_score import math, gsm8k

def select_reward_fn(data_source):
    if data_source == 'lighteval/MATH':
        return math.compute_score
    elif data_source == "blobs":
        from examples.data_preprocess.blobs import blobs_reward_fn
        return blobs_reward_fn
    elif data_source == "moons":
        from examples.data_preprocess.moons import moons_reward_fn
        return moons_reward_fn
    elif data_source == "linear":
        from examples.data_preprocess.linear import linear_reward_fn
        return linear_reward_fn
    else:
        raise NotImplementedError

def evaluate_responses(row):
    response_lst = row['responses']
    data_source = row['data_source']
    reward_data = row['reward_model']
    reward_fn = select_reward_fn(data_source)
    ground_truth = reward_data['ground_truth']
    
    score_lst = []
    correct_responses = []
    for r in response_lst:
        score = reward_fn(r, ground_truth)
        score_lst.append(score)
        if score == 1:
            correct_responses.append(r)
    
    return correct_responses

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--previous_correct_path", type=str, default=None, 
                       help="Path to previously used correct responses")
    args = parser.parse_args()
    
    # Read the generated responses
    df = pd.read_parquet(args.input_path)
    
    # Read previously used correct responses if provided
    previous_correct = None
    if args.previous_correct_path:
        previous_correct = pd.read_parquet(args.previous_correct_path)
        # Create a set of previously used prompt-response pairs for quick lookup
        previous_pairs = set(zip(previous_correct['prompt'], previous_correct['answer']))
    
    # Extract correct responses and create new dataset
    new_data = []
    total_responses = 0
    correct_responses = 0
    new_correct_responses = 0
    
    for _, row in df.iterrows():
        correct_responses_lst = evaluate_responses(row)
        total_responses += len(row['responses'])
        correct_responses += len(correct_responses_lst)
        
        # Create a copy of the row without the responses list
        row_data = row.drop('responses').to_dict()
        
        for response in correct_responses_lst:
            # Check if this prompt-response pair was previously used
            if previous_correct is None or (row_data['prompt'], response) not in previous_pairs:
                # Add the correct response to the row data
                row_data['answer'] = response
                new_data.append(row_data)
                new_correct_responses += 1
    
    # Create new dataframe with correct responses
    correct_df = pd.DataFrame(new_data)
    
    # Save filtered dataset
    correct_df.to_parquet(args.output_path)
    
    # Print statistics
    print(f'Total responses: {total_responses}')
    print(f'Correct responses: {correct_responses}')
    print(f'Accuracy: {correct_responses/total_responses:.4f}')
    print(f'New correct responses (not used before): {new_correct_responses}')
    print(f'Number of prompt-response pairs in new dataset: {len(correct_df)}')
    print('\nColumns in output dataset:')
    for col in correct_df.columns:
        print(f'- {col}')

if __name__ == "__main__":
    main() 