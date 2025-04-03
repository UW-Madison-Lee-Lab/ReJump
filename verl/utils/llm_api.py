from typing import List, Dict, Any
from openai import OpenAI
import anthropic
import concurrent.futures
import time
import numpy as np
import torch
import pdb
from verl.utils.reward_score import gsm8k, math, multiply, countdown

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
    elif "circles" in data_source:
        from examples.data_preprocess.circles import circles_reward_fn
        return circles_reward_fn
    else:
        raise NotImplementedError

class APIRewardManager:
    """The reward manager for API outputs.
    """
    def __init__(
        self, 
        num_examine, 
        return_dict=False,
    ) -> None:
        self.num_examine = num_examine  # the number of batches of responses to print to the console
        self.return_dict = return_dict
    
    def __call__(self, sequences_lst: List[str], ground_truths: List[str], data_sources: List[str]):
        """Compute rewards for API-generated sequences.
        
        Args:
            sequences_lst: List of generated sequences (already decoded strings)
            ground_truths: List of ground truth answers
            data_sources: List of data source identifiers
        """
        reward_tensor = torch.zeros(len(sequences_lst), dtype=torch.float32)
        already_print_data_sources = {}

        for i in range(len(sequences_lst)):
            sequences_str = sequences_lst[i]
            ground_truth = ground_truths[i]
            data_source = data_sources[i]

            # select rm_score
            compute_score_fn = _select_rm_score_fn(data_source)

            score = compute_score_fn(solution_str=sequences_str, ground_truth=ground_truth)
            reward_tensor[i] = score

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(sequences_str)

        if self.return_dict:
            return {
                'reward_tensor': reward_tensor, 
                'sequences_lst': sequences_lst,
            }
        else:
            return reward_tensor

class LLMAPI:
    def __init__(self, api_key: str, model_name: str, template_type: str = None):
        """
        Initialize the LLM API client.
        
        Args:
            api_key: API key for the service
            model_name: Name of the model to use (e.g., "deepseek-ai/deepseek-chat" or "openai/gpt-4")
        """
        self.model_name = model_name
        if model_name.startswith("deepseek-ai/"):
            self.client = OpenAI(
                api_key=api_key,
                base_url="https://api.deepseek.com/v1"
            )
            self.model = model_name.replace("deepseek-ai/", "")  # Remove the prefix to get the actual model name
        elif model_name.startswith("openai/"):
            self.client = OpenAI(
                api_key=api_key
            )
            self.model = model_name.replace("openai/", "")  # Remove the prefix to get the actual model name
        elif model_name.startswith("openrouter-"):
            self.client = OpenAI(
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1"
            )
            self.model = model_name.replace("openrouter-", "")  # Remove the prefix to get the actual model name
        elif model_name.startswith("claude/"):
            self.client = anthropic.Anthropic(api_key=api_key)
            self.model = model_name.replace("claude/", "")  # Remove the prefix to get the actual model name
            self.client_type = "anthropic"
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        if "no_reasoning" in template_type:
            self.thinking = False
        else:
            self.thinking = True

    def generate(self, messages: List[Dict[str, str]], max_tokens: int = 8000, temperature: float = 0.7) -> str:
        max_retries = 1000  # Very large number of retries
        timeout = 60  # 60 seconds timeout
        
        # Ensure messages is a list
        if not isinstance(messages, list):
            messages = [messages]
        
        for attempt in range(max_retries):
            try:
                if self.client_type is not None and self.client_type == "anthropic":
                    if self.thinking:
                        thinking={
                            "type": "enabled",
                            "budget_tokens": min(16000, max_tokens // 2)
                        }
                    else:
                        thinking = {
                            "type": "disabled",
                        }
                    response = self.client.messages.create(
                        model=self.model,
                        max_tokens=max_tokens,
                        messages=messages,
                        thinking=thinking
                    )
                    content = response.content[1].text
                    reasoning_content = response.content[0].thinking
                else:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        #max_tokens=max_tokens,######
                        #temperature=temperature,
                        #top_p=0.9,
                        #frequency_penalty=0.0,
                        #presence_penalty=0.0,
                        #stream=False,
                        #timeout=timeout
                    )
                
                    content = response.choices[0].message.content
                    reasoning_content = getattr(response.choices[0].message, 'reasoning_content', None)
                    if reasoning_content is None:
                        reasoning_content = getattr(response.choices[0].message, 'reasoning', None)
                """
                response = self.client.responses.create(
                    model=self.model,
                    input=messages, 
                    #generate_summary="detailed"
                )


                print(response.output) 
                #print(response.summary)
                
                breakpoint()                
                
                """

                # Check if response is complete (ends with proper tags or punctuation)
                if '</answer>' not in content:
                    if attempt < max_retries - 1:
                        time.sleep(5)
                        continue

                return content, reasoning_content

            except KeyboardInterrupt:
                raise KeyboardInterrupt
            except pdb.bdb.BdbQuit:
                raise pdb.bdb.BdbQuit
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(5)
                    continue
                    
                if "maximum context length" in str(e).lower():
                    return (response.choices[0].message.content, getattr(response.choices[0].message, 'reasoning_content', None)) if 'response' in locals() else ("", None)
                raise

    def process_batch(self, batch_chat_lst: List[Dict[str, str]], n_samples: int = 1, config=None, batch_idx: int = 0, wandb=None, ground_truths=None, data_sources=None) -> List[List[str]]:
        """Process a batch of chat messages in parallel using threading."""
        batch_output_lst = [[] for _ in range(n_samples)]
        
        def process_single_chat(chat, sample_idx):
            try:
                gen_start_time = time.time()
                
                response, reasoning_content = self.generate(
                    messages=chat,
                    max_tokens=config.rollout.response_length if config else 8000,
                    temperature=config.rollout.temperature if config else 0.7
                )
                response_length = len(response.split())
                generation_time = time.time() - gen_start_time
                
                metrics = {
                    f'sample_{sample_idx}/generation_time': generation_time,
                    f'sample_{sample_idx}/tokens_per_second': response_length / max(generation_time, 1e-6),
                    f'sample_{sample_idx}/response_length': response_length,
                    'batch': batch_idx,
                    'sample_idx': sample_idx
                }
                return response, reasoning_content, metrics, None
            except Exception as e:
                print(f"Error processing chat {sample_idx}: {str(e)}")
                return None, None, None, e

        # Create a thread pool for parallel processing
        max_workers = min(len(batch_chat_lst), 16)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for i in range(n_samples):
                for j, chat in enumerate(batch_chat_lst):
                    future = executor.submit(process_single_chat, chat, i)
                    futures.append((future, i, j))

            batch_reasoning_lst = [[] for _ in range(n_samples)]
            for future, sample_idx, chat_idx in futures:
                response, reasoning_content, metrics, error = future.result()
                if error is None:
                    batch_output_lst[sample_idx].append(response)
                    batch_reasoning_lst[sample_idx].append(reasoning_content)
                    if wandb is not None:
                        wandb.log(metrics)
                else:
                    print(f"Error in batch {batch_idx}, sample {sample_idx}, chat {chat_idx}: {str(error)}")
                    batch_output_lst[sample_idx].append("")  # Append empty string for failed generations
                    batch_reasoning_lst[sample_idx].append(None)

        # If ground truths and data sources are provided, compute rewards
        if ground_truths is not None and data_sources is not None:
            reward_manager = APIRewardManager(num_examine=0, return_dict=True)
            reward_dict = reward_manager(
                sequences_lst=[seq for samples in batch_output_lst for seq in samples],
                ground_truths=ground_truths,
                data_sources=data_sources
            )
            return batch_output_lst, reward_dict, batch_reasoning_lst

        return batch_output_lst, None, batch_reasoning_lst

    @staticmethod
    def convert_chat_list(chat_lst: List[Any]) -> List[Dict[str, str]]:
        """Convert a list of chats to the format expected by the API."""
        converted_chats = []
        for chat in chat_lst:
            if isinstance(chat, np.ndarray):
                chat = chat.tolist()
            
            if isinstance(chat, list):
                # If it's a list of dictionaries, take the first one
                if len(chat) > 0 and isinstance(chat[0], dict):
                    chat = chat[0]
                else:
                    chat = " ".join(str(x) for x in chat)
            
            if isinstance(chat, dict):
                if 'content' in chat:
                    content = chat['content']
                    # If content is a string that looks like a dictionary, extract the inner content
                    if isinstance(content, str) and content.startswith('{') and content.endswith('}'):
                        try:
                            import ast
                            inner_dict = ast.literal_eval(content)
                            if isinstance(inner_dict, dict) and 'content' in inner_dict:
                                content = inner_dict['content']
                        except:
                            pass
                    converted_chats.append({
                        "role": "user",
                        "content": content
                    })
                else:
                    # If it's a dict but doesn't have content, convert the whole dict to string
                    converted_chats.append({
                        "role": "user",
                        "content": str(chat)
                    })
            else:
                # If it's not a dict, convert to string
                converted_chats.append({
                    "role": "user",
                    "content": str(chat)
                })
            
        return converted_chats 