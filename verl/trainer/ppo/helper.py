

from verl import DataProto
import torch, re
from examples.data_preprocess.helper import _select_rm_score_fn
import pdb

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
        answer_lst = []
        reasoning_lst = []
        already_print_data_sources = {}
        probs_token_list = []
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            probs = data_item.batch['log_probs']
            
            if len(probs)>=valid_response_length:
                item_logprob_dict = {'tokens': [], 'logprobs': []}
                for j in range(valid_response_length):
                    item_logprob_dict['tokens'].append(self.tokenizer.decode([int(valid_response_ids[j])]))
                    item_logprob_dict['logprobs'].append(float(probs[j]))  # Assuming probs[i] is a tuple of (token_id, logprob)
            else:
                item_logprob_dict = {None}
                    # --- Store the dictionary for this batch item ---
                    # Note: No padding applied to lists inside the dict
            probs_token_list.append(item_logprob_dict)
            # else: logprobs not requested or not returned, batch_logprob_dicts[i] remains None
            # decode
            # sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences = valid_response_ids
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
            # print(f"data_source: {data_source}")
            compute_score_fn = _select_rm_score_fn(data_source)

            score = compute_score_fn(solution_str=sequences_str, ground_truth=ground_truth)
            reward_tensor[i, valid_response_length - 1] = score
            
            answer_match = re.search(r'<answer>(.*?)</answer>', sequences_str, re.DOTALL)
            answer_content = answer_match.group(1).strip() if answer_match else ""
            reasoning_match = re.search(r'<think>(.*?)</think>', sequences_str, re.DOTALL)
            reasoning_content = reasoning_match.group(1).strip() if reasoning_match else ""
            
            answer_lst.append(answer_content)
            reasoning_lst.append(reasoning_content)

            # print(f"ground_truth: {ground_truth}")
            # print("type of ground_truth: ", type(ground_truth))
            # print(f"sequences_str: {sequences_str}")
            # print("type of sequences_str: ", type(sequences_str))
            # print(f"score: {score}")
            # print("shape of reward_tensor: ", reward_tensor.shape)
            # print(f"reward_tensor: {reward_tensor}")
            # input("Press Enter to continue...")

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(sequences_str)

        if self.return_dict:
            return {
                'reward_tensor': reward_tensor, 
                'sequences_lst': sequences_lst,
                'answer_lst': answer_lst,
                'reasoning_lst': reasoning_lst,
                'probs_token_list': probs_token_list,
            }
        else:
            return reward_tensor