import os
import json
import random
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, DictConfig, OmegaConf


@dataclass
class ICLExampleConfig:
    task_type: str = MISSING  # blobs, circles, linear, moons
    flip_rate: float = MISSING  # 0.0, 0.1, 0.2
    noise_type: float = MISSING  # noise level like 0.1, 1.0
    shot_type: int = MISSING  # 50, 100
    reslen_type: int = MISSING  # 3046, 5686
    nsamples_type: int = MISSING  # 500
    num_examples: int = MISSING  # Number of examples to use


@dataclass
class TestDataConfig:
    task_type: str = MISSING  # blobs, circles, linear, moons
    flip_rate: float = MISSING  # 0.0, 0.1, 0.2
    noise_type: float = MISSING  # noise level like 0.1, 1.0
    shot_type: int = MISSING  # 50, 100
    nsamples_type: int = MISSING  # 500
    num_samples: int = MISSING  # Number of test samples to use


@dataclass
class ICLReasoningConfig:
    icl_examples: List[ICLExampleConfig] = field(default_factory=list)
    test_data: TestDataConfig = MISSING
    icl_example_seed: int = 42
    test_data_seed: int = 42
    output_path: str = "/staging/szhang967/icl_datasets"


cs = ConfigStore.instance()
cs.store(name="config", node=ICLReasoningConfig)
cs.store(group="icl_examples", name="example_config", node=ICLExampleConfig)
cs.store(group="test_data", name="test_config", node=TestDataConfig)


def load_deepseek_results(base_path: str, task_type: str, shot_type: int, 
                         reslen_type: int, nsamples_type: int, 
                         noise_type: float, flip_rate: float):
    """
    Load DeepSeek results based on the configuration
    """
    result_dir = f"{task_type}_{shot_type}_shot_base_reslen_{reslen_type}_nsamples_{nsamples_type}_noise_{noise_type}_flip_rate_{flip_rate}"
    result_path = os.path.join(base_path, result_dir, "global_step_0", "test.json")
    
    # Read the JSON file
    with open(result_path, 'r') as f:
        content = f.read()
        # Parse each JSON line separately
        results = []
        for line in content.strip().split('\n'):
            if line.strip():
                results.append(json.loads(line))
                
    return results


def load_test_data(base_path: str, task_type: str, shot_type: int, 
                  nsamples_type: int, noise_type: float, flip_rate: float):
    """
    Load test data based on the configuration
    """
    test_path = os.path.join(
        base_path, 
        "datasets", 
        task_type,
        f"{shot_type}_shot",
        "base",
        f"{nsamples_type}_samples_{noise_type}_noise_{flip_rate}_flip_rate",
        "test.parquet"
    )
    
    return pd.read_parquet(test_path)


def create_instruction(task_type: str) -> str:
    """
    Create a task instruction based on the task type.
    This instruction should be generic enough to not reveal specific task settings.
    """
    instructions = {
        "blobs": (
            "This is a classification task. You will be provided with examples of how "
            "a skilled reasoner classifies data points based on their features. "
            "Study the examples carefully to understand the reasoning process. "
            "Then, classify the new data point following a similar reasoning approach. "
            "First work through your reasoning step by step in <think></think> tags, "
            "then provide your final answer in <answer></answer> tags."
        ),
        "circles": (
            "In this classification task, you will see examples of how to classify points "
            "based on their coordinates. The examples show a skilled reasoner's thought process "
            "and final classification. Learn from these examples, then classify the new data point "
            "using a similar approach. Show your reasoning in <think></think> tags and "
            "your final answer in <answer></answer> tags."
        ),
        "linear": (
            "You are presented with a classification problem where each data point has two features. "
            "Examples show how an expert reasoner classifies these points and their thought process. "
            "Study the examples to understand the classification pattern, then apply similar reasoning "
            "to classify the new data point. Document your thinking in <think></think> tags and "
            "provide your classification in <answer></answer> tags."
        ),
        "moons": (
            "In this task, you'll classify data points into categories based on their features. "
            "The examples show how a skilled reasoner approaches this problem, including their "
            "step-by-step thought process and final classification. Learn from these examples, "
            "then apply similar reasoning to classify the new data point. Record your thinking "
            "in <think></think> tags and your answer in <answer></answer> tags."
        )
    }
    
    return instructions.get(task_type, instructions["blobs"])


def extract_test_prompt_content(result: Dict[str, Any]) -> str:
    """
    Extract the prompt content from a DeepSeek result
    """
    if "prompt" in result and isinstance(result["prompt"], list) and len(result["prompt"]) > 0:
        return result["prompt"][0]["content"]
    return ""


def extract_icl_responses(result: Dict[str, Any]) -> str:
    """
    Extract the model's responses from a DeepSeek result
    """
    if "responses" in result and isinstance(result["responses"], list) and len(result["responses"]) > 0:
        return result["responses"][0]
    return ""


def sample_icl_examples(config: ICLExampleConfig, base_path: str, rng: np.random.RandomState) -> List[Dict[str, Any]]:
    """
    Sample ICL examples based on the configuration
    """
    # Load all available examples from the specified configuration
    results = load_deepseek_results(
        base_path,
        config.task_type,
        config.shot_type,
        config.reslen_type,
        config.nsamples_type,
        config.noise_type,
        config.flip_rate
    )
    
    # Sample the required number of examples
    if len(results) <= config.num_examples:
        return results
    
    return rng.choice(results, config.num_examples, replace=False).tolist()


def create_prompt(instruction: str, icl_examples: List[Dict[str, Any]], test_prompt_content: str) -> str:
    """
    Create a prompt combining instruction, ICL examples, and test data
    """
    prompt = instruction + "\n\n"
    
    # Add ICL examples
    for i, example in enumerate(icl_examples):
        example_prompt = extract_test_prompt_content(example)
        example_response = extract_icl_responses(example)
        
        # Extract the task description and example data points from the prompt
        # Split by "User:" to get the part after it
        if "User:" in example_prompt:
            example_content = example_prompt.split("User:", 1)[1].strip()
        else:
            example_content = example_prompt
            
        prompt += f"Example {i+1}:\n"
        prompt += f"Problem: {example_content}\n"
        prompt += f"Reasoning: {example_response}\n\n"
    
    # Add test data (only the part after "User:")
    if "User:" in test_prompt_content:
        test_content = test_prompt_content.split("User:", 1)[1].strip()
    else:
        test_content = test_prompt_content
        
    prompt += f"Now, solve this problem:\n{test_content}"
    
    return prompt


@hydra.main(config_path=None, config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    
    # Create output directory if it doesn't exist
    output_path = Path(cfg.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Set random seeds for reproducibility
    random.seed(cfg.icl_example_seed)
    icl_rng = np.random.RandomState(cfg.icl_example_seed)
    test_rng = np.random.RandomState(cfg.test_data_seed)
    
    # Base paths
    deepseek_results_base = "/home/szhang967/liftr/analyze_deepseek/deepseek-resutls/deepseek-ai-deepseek-reasoner"
    test_data_base = "/home/szhang967/liftr/analyze_deepseek/deepseek-used_datasets"
    
    # Initialize dataset list
    dataset = []
    
    # Load test data
    test_data_df = load_test_data(
        test_data_base,
        cfg.test_data.task_type,
        cfg.test_data.shot_type,
        cfg.test_data.nsamples_type,
        cfg.test_data.noise_type,
        cfg.test_data.flip_rate
    )
    
    # Create base instruction
    instruction = create_instruction(cfg.test_data.task_type)
    
    # Sample test data
    if len(test_data_df) <= cfg.test_data.num_samples:
        test_indices = list(range(len(test_data_df)))
    else:
        test_indices = test_rng.choice(len(test_data_df), cfg.test_data.num_samples, replace=False)
    
    # For each group of ICL examples, create a dataset
    for icl_config in cfg.icl_examples:
        # Sample ICL examples
        icl_examples = sample_icl_examples(icl_config, deepseek_results_base, icl_rng)
        
        # For each test data point, create a prompt
        for test_idx in test_indices:
            test_row = test_data_df.iloc[test_idx]
            
            # Load the corresponding DeepSeek result to get the prompt content
            deepseek_results = load_deepseek_results(
                deepseek_results_base,
                cfg.test_data.task_type,
                cfg.test_data.shot_type,
                icl_config.reslen_type,  # Using the ICL's reslen type for consistency
                cfg.test_data.nsamples_type,
                cfg.test_data.noise_type,
                cfg.test_data.flip_rate
            )
            
            # Find the matching test data point in DeepSeek results
            matched_result = None
            for result in deepseek_results:
                if ("features" in result and "features" in test_row and 
                    np.array_equal(np.array(result["features"]), np.array(test_row["features"]))):
                    matched_result = result
                    break
            
            if matched_result is None:
                print(f"Warning: Could not find matching DeepSeek result for test data point {test_idx}")
                continue
            
            # Extract test prompt content
            test_prompt_content = extract_test_prompt_content(matched_result)
            
            # Create prompt
            prompt = create_prompt(instruction, icl_examples, test_prompt_content)
            
            # Create dataset entry
            entry = {
                "prompt": prompt,
                "features": test_row["features"],
                "label": test_row["label"] if "label" in test_row else None,
                "icl_examples": [
                    {
                        "task_type": icl_config.task_type,
                        "shot_type": icl_config.shot_type,
                        "reslen_type": icl_config.reslen_type,
                        "nsamples_type": icl_config.nsamples_type,
                        "noise_type": icl_config.noise_type,
                        "flip_rate": icl_config.flip_rate,
                        "features": example["features"],
                        "label": example["label"],
                        "responses": example["responses"],
                        "reasoning": example.get("reasonings", [])
                    }
                    for example in icl_examples
                ],
                "test_data": {
                    "task_type": cfg.test_data.task_type,
                    "shot_type": cfg.test_data.shot_type,
                    "nsamples_type": cfg.test_data.nsamples_type,
                    "noise_type": cfg.test_data.noise_type,
                    "flip_rate": cfg.test_data.flip_rate,
                }
            }
            
            dataset.append(entry)
    
    # Create a descriptive filename
    filename = f"{cfg.test_data.task_type}_{cfg.test_data.shot_type}shot_n{cfg.test_data.noise_type}_f{cfg.test_data.flip_rate}_test{cfg.test_data.num_samples}_icl{len(cfg.icl_examples)}_seed{cfg.icl_example_seed}.parquet"
    output_file = output_path / filename
    
    # Convert to DataFrame and save as parquet
    df = pd.DataFrame(dataset)
    df.to_parquet(str(output_file))
    
    print(f"Created dataset with {len(dataset)} samples, saved to {output_file}")


if __name__ == "__main__":
    main()
