from typing import List, Dict, Any
from openai import OpenAI
import anthropic
import concurrent.futures
import time
import numpy as np
import torch
import pdb
import re, openai
import json

from examples.data_preprocess.helper import _select_rm_score_fn, get_answer_format
from google import genai
import httpx
from constants import supported_datasets
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
    def __init__(
        self, 
        api_key: str, 
        model_name: str, 
        template_type: str = None,
    ):
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
            self.client_type = "deepseek"
        elif model_name.startswith("openai/"):
            self.client = OpenAI(
                api_key=api_key
            )
            self.model = model_name.replace("openai/", "")  # Remove the prefix to get the actual model name
            self.client_type = "openai"
        elif model_name.startswith("openrouter-"):
            self.client = OpenAI(
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1"
            )
            self.model = model_name.replace("openrouter-", "")  # Remove the prefix to get the actual model name
            self.client_type = "openrouter"
        elif model_name.startswith("claude/"):
            print(f"Initializing Claude client with model: {model_name}")
            try:
                self.client = anthropic.Anthropic(
                    api_key=api_key,
                    timeout=120.0  # Set timeout to 120 seconds
                )
                print(f"Claude client initialized successfully")
            except Exception as e:
                print(f"Error initializing Claude client: {str(e)}")
                raise
            self.model = model_name.replace("claude/", "")  # Remove the prefix to get the actual model name
            print(f"Using Claude model: {self.model}")
            self.client_type = "anthropic"
            if "thinking" in self.model: 
                self.model = self.model.replace("-thinking", "")
        elif model_name.startswith("xai/"):
            self.client = OpenAI(
                api_key=api_key,
                base_url="https://api.x.ai/v1"
            )
            self.model = model_name.replace("xai/", "")
            self.client_type = "xai"
        elif model_name.startswith("google/"):
            self.client = genai.Client(api_key=api_key)
            self.model = model_name.replace("google/", "")
            self.client_type = "google"
        elif model_name.startswith("alibaba/"):
            self.client = OpenAI(
                api_key=api_key,
                base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
            )
            self.model = model_name.replace("alibaba/", "")
            self.client_type = "alibaba"
            if "thinking" in self.model: 
                self.model = self.model.replace("-thinking", "")
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        if "reasoning_api" in template_type:
            self.thinking = "enabled"
        elif "no_reasoning" in template_type:
            self.thinking = "disabled"
        else:
            self.thinking = "cot"
        

    def generate(
        self, 
        messages: List[Dict[str, str]], 
        max_tokens: int = 8000, 
        temperature: float = 0.1,
        data_source: str = None,
    ) -> str:
        max_retries = 1000  # Increased retry count
        if max_retries <= 0: raise ValueError("max_retries must be greater than 0")
        timeout = 120  # Increased timeout to 120 seconds
        
        # Ensure messages is a list
        if not isinstance(messages, list):
            messages = [messages]
        
        for attempt in range(max_retries):
            try:
                if self.client_type == "anthropic":
                    if self.thinking == "enabled":
                        thinking={
                            "type": "enabled",
                            "budget_tokens": max(1048, min(30000, max_tokens - 10))
                        }
                    else:
                        thinking = {
                            "type": "disabled",
                        }
                    response = self.client.messages.create(
                        model=self.model,
                        system = "You are a helpful data analysis assistant.",
                        max_tokens=max(1064, max_tokens),
                        messages=messages,
                        thinking=thinking
                    )
                    
                    if self.thinking == "enabled":
                        reasoning = response.content[0].thinking
                        output = response.content[1].text
                    else:
                        output = response.content[0].text
                        reasoning = ""
                        
                elif self.client_type == "google":
                    response = self.client.models.generate_content(
                        model=self.model,
                        contents = [messages[0]['content']],
                    )
                    if response is None or response.candidates is None or len(response.candidates) == 0: 
                        reasoning, output = "", ""
                    else:
                        output = response.candidates[0].content.parts[0].text
                        reasoning = ""
                elif self.client_type in ["openai", "deepseek", "openrouter", "xai"]:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=temperature,
                    )
                
                    output = response.choices[0].message.content
                    reasoning = getattr(response.choices[0].message, 'reasoning_content', None)
                    if reasoning is None:
                        reasoning = getattr(response.choices[0].message, 'reasoning', None)
                elif self.client_type == "alibaba":
                    if self.thinking == "enabled":
                        extra_body = {
                            "enable_thinking": True,
                            "thinking_budget": max(1048, min(30000, max_tokens - 10)),
                            "result_format": "message",
                        }
                        stream = True
                    else:
                        extra_body = None
                        stream = False
                    completion = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=temperature,
                        extra_body=extra_body,
                        stream=stream,
                    )
                    
                    if self.thinking == "enabled":
                        reasoning_content = ""  # Complete reasoning process
                        answer_content = ""  # Complete response
                        is_answering = False  # Whether entering the response phase
                        
                        for chunk in completion:
                            if not chunk.choices:
                                continue 
                            
                            delta = chunk.choices[0].delta
                            if hasattr(delta, "reasoning_content") and delta.reasoning_content is not None:
                                reasoning_content += delta.reasoning_content
                            if hasattr(delta, "content") and delta.content:
                                if not is_answering:
                                    is_answering = True
                                answer_content += delta.content
                        
                        reasoning = reasoning_content
                        output = answer_content
                    else:
                        reasoning = ""
                        output = completion.choices[0].message.content

                answer_format = get_answer_format(supported_datasets[data_source]['answer_format'], "")
                if self.thinking == "enabled":
                    reasoning_content = reasoning
                    answer_content = output
                    if answer_format["left"] in answer_content:
                        response_content = f"<think>{reasoning_content}</think>\n{answer_content}"
                    else:
                        response_content = f"<think>{reasoning_content}</think>\n{get_answer_format(supported_datasets[data_source]['answer_format'], answer_content)['example']}"
                elif self.thinking == "cot":
                    reasoning_match = re.search(r'<think>(.*?)</think>', output, re.DOTALL)
                    reasoning_content = reasoning_match.group(1).strip() if reasoning_match else ""
                    
                    answer_match = re.search(rf"{get_answer_format(supported_datasets[data_source]['answer_format'], '.*?')['example']}", output, re.DOTALL)
                    answer_content = answer_match.group(1).strip() if answer_match else ""
                    response_content = output
                else:
                    reasoning_content = ""
                    answer_content = output
                    if answer_format["left"] in answer_content:
                        response_content = answer_content
                    else:
                        response_content = f"{get_answer_format(supported_datasets[data_source]['answer_format'], answer_content)['example']}"
                    
                # Check if response is complete (ends with proper tags or punctuation)
                if answer_format["right"] not in response_content:
                    print(f"output: {output}")
                    print(f"reasoning: {reasoning}")

                return response_content, reasoning_content, answer_content
            
            except anthropic.RateLimitError as e:
                print(f"Rate limit error: {e}")
                time.sleep(timeout)
                
            except json.decoder.JSONDecodeError as e:
                print(f"JSONDecodeError: {e}")
                time.sleep(timeout)
                
            except IndexError as e:
                print(f"IndexError: {e}")
                return "", "", ""
            
            except httpx.ReadError as e:
                print(f"ReadError: {e}")
                time.sleep(timeout)
                
            except genai.errors.ServerError as e:
                print(f"ServerError: {e}")
                time.sleep(timeout)
                
            except genai.errors.ClientError as e:
                print(f"ClientError: {e}")
                time.sleep(timeout)
                
                
            except Exception as e:
                print(type(e), e)
                pdb.set_trace()
                
            print(f"Failed to generate response after {attempt} attempts, max_retries: {max_retries}")
            
        raise Exception("Failed to generate response")


    def process_batch(self, batch_chat_lst: List[Dict[str, str]], n_samples: int = 1, config=None, batch_idx: int = 0, wandb=None, ground_truths=None, data_sources=None) -> List[List[str]]:
        """Process a batch of chat messages in parallel using threading."""
        batch_output_lst = [[] for _ in range(n_samples)]
        
        def process_single_chat(chat, sample_idx):
            # try:
            gen_start_time = time.time()
            
            response_content, reasoning_content, answer_content = self.generate(
                messages=chat,
                max_tokens=config.rollout.response_length,
                temperature=config.rollout.temperature if config else 0.7,
                data_source=data_sources[sample_idx],
            )
            response_length = len(response_content.split())
            generation_time = time.time() - gen_start_time
            
            metrics = {
                f'sample_{sample_idx}/generation_time': generation_time,
                f'sample_{sample_idx}/tokens_per_second': response_length / max(generation_time, 1e-6),
                f'sample_{sample_idx}/response_length': response_length,
                'batch': batch_idx,
                'sample_idx': sample_idx
            }
            return response_content, reasoning_content, answer_content, metrics, None
            # except Exception as e:
            #     print(f"Error processing chat {sample_idx}: {str(e)}")
            #     return None, None, None, None, e

        # Create a thread pool for parallel processing
        max_workers = min(len(batch_chat_lst), config.rollout.api_workers)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for i in range(n_samples):
                for j, chat in enumerate(batch_chat_lst):
                    future = executor.submit(process_single_chat, chat, i)
                    futures.append((future, i, j))

            batch_reasoning_lst = [[] for _ in range(n_samples)]
            batch_answer_lst = [[] for _ in range(n_samples)]
            for future, sample_idx, chat_idx in futures:
                response_content, reasoning_content, answer_content, metrics, error = future.result()
                if error is None:
                    batch_output_lst[sample_idx].append(response_content)
                    batch_reasoning_lst[sample_idx].append(reasoning_content)
                    batch_answer_lst[sample_idx].append(answer_content)
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
            return batch_output_lst, reward_dict, batch_reasoning_lst, batch_answer_lst

        return batch_output_lst, None, batch_reasoning_lst, batch_answer_lst

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