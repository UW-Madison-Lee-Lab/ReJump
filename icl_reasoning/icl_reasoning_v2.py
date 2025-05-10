import pandas as pd
import random
import pdb, argparse
import numpy as np
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--result_path", type=str, nargs="+")
    parser.add_argument("--num_shot", type=int, default=3)
    parser.add_argument("--output_path", type=str)
    args = parser.parse_args()



    df = pd.read_parquet(args.dataset_path)
    result_dfs = [pd.read_parquet(path) for path in args.result_path]

    new_df = []

    for i in range(len(df)):
        row = df.iloc[i]
        prompt = row["prompt"][0]["content"]
        
        replace = len(args.result_path) < args.num_shot
        result_idices = random.choices(np.arange(len(args.result_path)), k=args.num_shot, replace=replace)
        indices = [i]
        
        for result_idx in result_idices:
            result_df = result_dfs[result_idx]
            other_indices = list(range(len(result_df)))
            for j in indices:
                other_indices.remove(j)
            random_index = random.choice(other_indices)
            indices.append(random_index)
        new_prompt = f"""
        You will be provided with examples of how a skilled reasoner solves a problem.
        Study the examples carefully to understand the reasoning process.
        """
        for j in random_index:
            new_prompt += f"""
            |--- Example {j} ---|
            Q: {result_df.iloc[j]['prompt'][0]['content']}
            A: {result_df.iloc[j]['responses'][0]}
            |--- End of Example {j} ---|
            """
        new_prompt += f"""
        Now, solve the problem following a similar reasoning approach.
        |--- Problem ---|
        Q: {prompt}
        |--- End of Problem ---|
        """
        
        new_row = row.copy()
        new_row["prompt"] = [{"content": new_prompt}]
        new_df.append(new_row)
        
    new_df = pd.DataFrame(new_df)
    new_df.to_parquet(args.output_path)







