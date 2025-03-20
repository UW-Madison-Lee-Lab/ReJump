import pandas as pd
import argparse, os
import numpy as np
from verl.utils.reward_score import math, gsm8k
import sys
import wandb

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
    parser.add_argument("--already_trained_correct_path", type=str, required=True,
                       help="Path to already trained correct responses")
    parser.add_argument("--min_correct_count", type=int, default=10,
                       help="Minimum number of correct responses required to continue")
    parser.add_argument("--wandb_project", type=str, default=None,
                       help="Weights & Biases project name for logging (optional)")
    parser.add_argument("--wandb_entity", type=str, default=None,
                       help="Weights & Biases entity name for logging (optional)")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                       help="Weights & Biases run name (optional)")
    args = parser.parse_args()
    
    # Initialize wandb if project name is provided
    if args.wandb_project:
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            config={
                "input_path": args.input_path,
                "output_path": args.output_path,
                "already_trained_path": args.already_trained_correct_path,
                "min_correct_count": args.min_correct_count
            }
        )
    
    # Read the generated responses
    df = pd.read_parquet(args.input_path)
    
    # Read already trained correct responses if provided
    already_trained = None
    if os.path.exists(args.already_trained_correct_path):
        already_trained = pd.read_parquet(args.already_trained_correct_path)
        # Create a set of previously used prompt-response pairs for quick lookup
        previous_pairs = set((str(p), str(a)) for p, a in zip(already_trained['prompt'], already_trained['answer']))
    
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
            if already_trained is None or (str(row_data['prompt']), str(response)) not in previous_pairs:
                # Add the correct response to the row data
                row_data['answer'] = response
                new_data.append(row_data)
                new_correct_responses += 1
    
    # Create new dataframe with correct responses
    correct_df = pd.DataFrame(new_data)
    
    # Print statistics
    accuracy = correct_responses/total_responses if total_responses > 0 else 0
    print(f'Total responses: {total_responses}')
    print(f'Correct responses: {correct_responses}')
    print(f'Accuracy: {accuracy:.4f}')
    print(f'New correct responses (not used before): {new_correct_responses}')
    print(f'Number of prompt-response pairs in new dataset: {len(correct_df)}')
    
    # Log metrics to wandb if enabled
    if args.wandb_project:
        wandb.log({
            "total_responses": total_responses,
            "correct_responses": correct_responses,
            "accuracy": accuracy,
            "new_correct_responses": new_correct_responses,
            "dataset_size": len(correct_df)
        })
        
        # Create a table for column information
        columns_table = wandb.Table(columns=["column_name"])
        for col in correct_df.columns:
            columns_table.add_data(col)
        wandb.log({"dataset_columns": columns_table})
    
    # Check if we have enough new correct responses
    if new_correct_responses < args.min_correct_count:
        error_msg = f"ERROR: Only found {new_correct_responses} new correct responses, which is less than the minimum required ({args.min_correct_count})."
        print(error_msg)
        if args.wandb_project:
            wandb.log({"error": error_msg})
            wandb.finish()
        sys.exit(1)  
    
    # Update the already_trained dataset only if we have new correct responses
    if already_trained is None:
        already_trained = correct_df
    else:
        # Concatenate and drop duplicates based on prompt and answer
        already_trained = pd.concat([already_trained, correct_df], ignore_index=True)
        already_trained = already_trained.drop_duplicates(subset=['prompt', 'answer'])
    
    already_trained.to_parquet(args.already_trained_correct_path)
    
    # Save filtered dataset
    correct_df.to_parquet(args.output_path)
    
    print('\nColumns in output dataset:')
    for col in correct_df.columns:
        print(f'- {col}')
    
    # Finish wandb run if enabled
    if args.wandb_project:
        wandb.finish()
    
    # Return success (0)
    return 0

if __name__ == "__main__":
    sys.exit(main())