from verl.utils.reward_score import gsm8k, math, multiply, countdown
from verl import DataProto
import torch

def _select_rm_score_fn(data_source):
    if data_source == 'openai/gsm8k':
        return gsm8k.compute_score
    elif data_source == 'lighteval/MATH':
        return math.compute_score
    elif "multiply" in data_source or "arithmetic" in data_source:
        return multiply.compute_score
    elif "countdown" in data_source:
        return countdown.compute_score
    elif "linear" in data_source:
        from examples.data_preprocess.linear import linear_reward_fn
        return linear_reward_fn
    elif "blobs" in data_source:
        from examples.data_preprocess.blobs import blobs_reward_fn
        return blobs_reward_fn
    elif "moons" in data_source:
        from examples.data_preprocess.moons import moons_reward_fn
        return moons_reward_fn
    else:
        raise NotImplementedError



class RewardManager():
    """The reward manager.
    """

    def __init__(
        self, 
        tokenizer, 
        num_examine, 
        return_dict=False,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.return_dict = return_dict
    
    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        sequences_lst = []
        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)
            sequences_lst.append(sequences_str)
            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
            # print(f"ground_truth: {ground_truth}")
            # print("type of ground_truth: ", type(ground_truth))
            # print(f"sequences_str: {sequences_str}")
            # print("type of sequences_str: ", type(sequences_str))
            # input("Press Enter to continue...")
            # print(f"sequences_str: {sequences_str}")


            # select rm_score
            data_source = data_item.non_tensor_batch['data_source']
            print(f"data_source: {data_source}")
            compute_score_fn = _select_rm_score_fn(data_source)

            score = compute_score_fn(solution_str=sequences_str, ground_truth=ground_truth)
            reward_tensor[i, valid_response_length - 1] = score

            print(f"ground_truth: {ground_truth}")
            print("type of ground_truth: ", type(ground_truth))
            print(f"sequences_str: {sequences_str}")
            print("type of sequences_str: ", type(sequences_str))
            print(f"sequences_str: {sequences_str}")
            print(f"score: {score}")
            print("shape of reward_tensor: ", reward_tensor.shape)
            print(f"reward_tensor: {reward_tensor}")
            input("Press Enter to continue...")

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1

        if self.return_dict:
            return {
                'reward_tensor': reward_tensor, 
                'sequences_lst': sequences_lst,
            }
        else:
            return reward_tensor
