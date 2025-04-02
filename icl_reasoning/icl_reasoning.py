import os
import json
import random
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, DictConfig, OmegaConf
from transformers import AutoTokenizer

# Import data generation functions from data_preprocess
from examples.data_preprocess.moons import gen_dataset as gen_moons_dataset
from examples.data_preprocess.circles import gen_dataset as gen_circles_dataset
from examples.data_preprocess.blobs import gen_dataset as gen_blobs_dataset
from examples.data_preprocess.linear import gen_dataset as gen_linear_dataset


@dataclass
class ICLExampleConfig:
    task_type: str = MISSING  # blobs, circles, linear, moons
    flip_rate: float = MISSING  # 0.0, 0.1, 0.2
    noise_type: float = MISSING  # noise level like 0.1, 1.0
    shot_type: int = MISSING  # 50, 100
    reslen_type: int = MISSING  # 3046, 5686
    nsamples_type: int = MISSING  # 500
    num_examples: int = MISSING  # Number of examples to use
    icl_example_maxlength: int = 5000  # Maximum token length for example


@dataclass
class TestDataConfig:
    task_type: str = MISSING  # blobs, circles, linear, moons
    flip_rate: float = MISSING  # 0.0, 0.1, 0.2
    noise_type: float = MISSING  # noise level like 0.1, 1.0
    num_samples: int = MISSING  # Number of test samples to use


@dataclass
class TestDataExampleConfig:
    task_type: str = MISSING  # blobs, circles, linear, moons
    flip_rate: float = MISSING  # 0.0, 0.1, 0.2
    noise_type: float = MISSING  # noise level like 0.1, 1.0
    shot_type: int = MISSING  # Number of examples to use


@dataclass
class ICLReasoningConfig:
    icl_examples: List[ICLExampleConfig] = field(default_factory=list)
    test_data: TestDataConfig = MISSING
    test_data_examples: TestDataExampleConfig = MISSING
    icl_example_seed: int = 42
    test_data_seed: int = 42
    tokenizer_name: str = "Qwen/Qwen2.5-3B-Instruct"  # Tokenizer used for length calculation
    output_path: str = "/staging/szhang967/icl_datasets"


cs = ConfigStore.instance()
cs.store(name="config", node=ICLReasoningConfig)
cs.store(group="icl_examples", name="example_config", node=ICLExampleConfig)
cs.store(group="test_data", name="test_config", node=TestDataConfig)
cs.store(group="test_data_examples", name="test_data_example_config", node=TestDataExampleConfig)


def get_tokenizer(tokenizer_name: str):
    """
    Get the tokenizer for token length calculation
    """
    try:
        return AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    except Exception as e:
        print(f"Warning: Failed to load tokenizer {tokenizer_name}: {e}")
        print("Falling back to default tokenizer (gpt2)")
        return AutoTokenizer.from_pretrained("gpt2")


def calculate_token_length(text: str, tokenizer) -> int:
    """
    Calculate the token length of a text string
    
    Args:
        text: Input text
        tokenizer: Tokenizer object
        
    Returns:
        Number of tokens
    """
    tokens = tokenizer.encode(text)
    return len(tokens)


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


def generate_test_data(task_type: str, num_samples: int, noise_level: float, flip_rate: float, seed_value: int) -> pd.DataFrame:
    """
    Generate test data using the appropriate data generator based on task type.
    
    Args:
        task_type: Type of task (moons, circles, blobs, linear)
        num_samples: Number of samples to generate
        noise_level: Noise level for the generator
        flip_rate: Label flip rate
        seed_value: Random seed for reproducibility
        
    Returns:
        DataFrame containing generated test data
    """
    if task_type == "moons":
        samples = gen_moons_dataset(
            num_samples=num_samples,
            noise=noise_level,
            label_flip_rate=flip_rate,
            seed_value=seed_value
        )
    elif task_type == "circles":
        samples = gen_circles_dataset(
            num_samples=num_samples,
            noise=noise_level,
            label_flip_rate=flip_rate,
            seed_value=seed_value
        )
    elif task_type == "blobs":
        samples = gen_blobs_dataset(
            num_samples=num_samples,
            cluster_std=noise_level,
            label_flip_rate=flip_rate,
            seed_value=seed_value
        )
    elif task_type == "linear":
        samples = gen_linear_dataset(
            num_samples=num_samples,
            noise=noise_level,
            label_flip_rate=flip_rate,
            seed_value=seed_value
        )
    else:
        raise ValueError(f"Unknown task type: {task_type}")
    
    # Convert to DataFrame
    df = pd.DataFrame({
        'features': [sample[0] for sample in samples],
        'label': [sample[1] for sample in samples]
    })
    
    return df


def create_instruction(task_type: str) -> str:
    """
    Create a task instruction based on the task type.
    This instruction should be generic enough to not reveal specific task settings.
    """
    valid_task_types = ["blobs", "circles", "linear", "moons"]
    if task_type not in valid_task_types:
        raise ValueError(f"Invalid task type: {task_type}. Must be one of {valid_task_types}")
    
    # Use the same instruction for all task types
    instruction = (
        "This is a classification task. You will be provided with examples of how "
        "a skilled reasoner classifies data points based on their features. "
        "Study the examples carefully to understand the reasoning process. "
        "Then, classify the new data point following a similar reasoning approach. "
        "First work through your reasoning step by step in <think></think> tags, "
        "then provide your final answer in <answer></answer> tags."
    )
    
    return instruction


def extract_test_prompt_content(result: Dict[str, Any]) -> str:
    """
    Extract the prompt content from a DeepSeek result
    """
    if "prompt" in result and isinstance(result["prompt"], list) and len(result["prompt"]) > 0:
        return result["prompt"][0]["content"]
    raise ValueError("Failed to extract prompt content from result")


def extract_icl_reasonings(result: Dict[str, Any]) -> str:
    """
    Extract the model's reasonings from a DeepSeek result
    """
    if "reasonings" in result and isinstance(result["reasonings"], list) and len(result["reasonings"]) > 0:
        return result["reasonings"][0]
    raise ValueError("Failed to extract reasonings from result")

def extract_icl_responses(result: Dict[str, Any]) -> str:
    """
    Extract the model's responses from a DeepSeek result
    """
    if "responses" in result and isinstance(result["responses"], list) and len(result["responses"]) > 0:
        return result["responses"][0]
    raise ValueError("Failed to extract responses from result")


def extract_features_from_prompt(prompt_content):
    """
    Extract all possible feature representations from a prompt content string
    
    Args:
        prompt_content: String containing the prompt content
        
    Returns:
        Set of extracted feature strings in various formats
    """
    extracted_features = set()
    
    # Method 1: Extract features from [brackets] after "features" (case insensitive)
    for keyword in ["features", "Features"]:
        prompt_parts = prompt_content.split(keyword)
        for part in prompt_parts[1:]:  # Skip the first part before "features"
            # Try to extract feature values using a simple approach
            if "[" in part and "]" in part:
                feature_str = part[part.find("["):part.find("]")+1]
                extracted_features.add(feature_str)
    
    # Method 2: Extract features from format "Features: x.xxx, y.yyy, Label: z"
    import re
    # Pattern to match "Features: X.XXX, Y.YYY" - both with and without brackets
    patterns = [
        r'Features:\s*([-\d\.]+),\s*([-\d\.]+)',
        r'features:\s*([-\d\.]+),\s*([-\d\.]+)',
        r'Features:\s*\[([-\d\.]+),\s*([-\d\.]+)\]',
        r'features:\s*\[([-\d\.]+),\s*([-\d\.]+)\]'
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, prompt_content)
        for match in matches:
            if len(match) == 2:  # Should have two coordinates
                try:
                    x, y = float(match[0]), float(match[1])
                    feature_str = f"[{x}, {y}]"
                    extracted_features.add(feature_str)
                    # Also add other formats of the same value
                    extracted_features.add(f"[{x:.6f}, {y:.6f}]")
                    extracted_features.add(str([x, y]))
                except ValueError:
                    # Skip if conversion to float fails
                    pass
    
    # Add the entire prompt for more thorough comparison later
    extracted_features.add(prompt_content)
    
    return extracted_features


def sample_icl_examples(config: ICLExampleConfig, base_path: str, rng: np.random.RandomState, tokenizer) -> List[Dict[str, Any]]:
    """
    Sample ICL examples based on the configuration, respecting the max token length constraint
    
    Args:
        config: ICL example configuration
        base_path: Base path for DeepSeek results
        rng: Random state for reproducibility
        tokenizer: Tokenizer for length calculation
        
    Returns:
        List of sampled examples
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
    
    if len(results) == 0:
        raise ValueError(f"No examples found for configuration: {config}")
    
    # Shuffle the results to get different samples each time
    results_copy = results.copy()
    rng.shuffle(results_copy)
    
    # Sample examples that respect the maximum token length
    sampled_examples = []
    for example in results_copy:
        if len(sampled_examples) >= config.num_examples:
            break
            
        # Check if example is under the maximum token length
        try:
            prompt_content = extract_test_prompt_content(example)
            reasoning = extract_icl_reasonings(example)
            response = extract_icl_responses(example)
            
            # Combine all text that would be included in the ICL example
            full_example_text = f"Problem: {prompt_content}\nReasoning: {reasoning}\n\n{response}"
            token_length = calculate_token_length(full_example_text, tokenizer)
            
            if token_length <= config.icl_example_maxlength:
                # Add metadata about this example
                example['config_info'] = {
                    "task_type": config.task_type,
                    "shot_type": config.shot_type,
                    "reslen_type": config.reslen_type,
                    "nsamples_type": config.nsamples_type,
                    "noise_type": config.noise_type,
                    "flip_rate": config.flip_rate,
                    "token_length": token_length
                }
                sampled_examples.append(example)
            else:
                print(f"Skipping example with token length {token_length} > {config.icl_example_maxlength}")
        except Exception as e:
            print(f"Error processing example: {e}")
    
    # If we couldn't find enough examples under the token limit
    if len(sampled_examples) < config.num_examples:
        # print(f"Warning: Requested {config.num_examples} examples but only {len(sampled_examples)} are under the token limit of {config.icl_example_maxlength}.")
        raise ValueError(f"Requested {config.num_examples} examples but only {len(sampled_examples)} are under the token limit of {config.icl_example_maxlength}.")
    return sampled_examples


def create_prompt(instruction: str, icl_examples: List[Dict[str, Any]], test_examples: List[Tuple], test_features: List[float], num_classes: int) -> str:
    """
    Create a prompt combining instruction, ICL examples, test examples, and test data
    
    Args:
        instruction: Task instruction
        icl_examples: List of ICL examples from DeepSeek
        test_examples: List of test examples (features, label) tuples
        test_features: Features of the target test data point
        num_classes: Number of classes for the task
    
    Returns:
        Complete prompt with all components
    """
    if len(icl_examples) == 0:
        raise ValueError("No ICL examples provided. At least one example is required.")
    
    prompt = instruction + "\n\n"
    
    # Add ICL examples
    for i, example in enumerate(icl_examples):
        example_prompt = extract_test_prompt_content(example)
        if not example_prompt:
            raise ValueError(f"Failed to extract prompt content from example {i+1}")
            
        example_response = extract_icl_reasonings(example) + "\n\n" + extract_icl_responses(example)
        if not example_response:
            raise ValueError(f"Failed to extract response from example {i+1}")
        
        # Extract the task description and example data points from the prompt
        # Split by "User:" to get the part after it
        if "User:" in example_prompt:
            example_content = example_prompt.split("User:", 1)[1].strip()
        else:
            example_content = example_prompt
            
        prompt += f"Example {i+1}:\n"
        prompt += f"Problem: {example_content}\n"
        prompt += f"Reasoning: {example_response}\n\n"
    
    # Add generated test examples in the format matching user's example
    if not test_examples:
        raise ValueError("No test examples provided. Test examples are required for the prompt.")
        
    prompt += f"The dataset has {num_classes} classes: {list(range(num_classes))}. We first provide you with some examples of how to classify data points.\n"
    
    for features, label in test_examples:
        # Format with 3 decimal places without brackets to match example format
        formatted_features = f"{features[0]:.3f}, {features[1]:.3f}"
        prompt += f"Features: {formatted_features}, Label: {label}\n"
    
    prompt += "\n"
    
    # Add the test problem
    test_prompt = f"Given the data point with features {test_features[0]:.3f}, {test_features[1]:.3f}, classify it into one of the possible classes. Show your work in <think></think> tags. And return the final answer in <answer></answer> tags, for example <answer>1</answer>."
    
    prompt += f"Now, solve this problem:\n{test_prompt}"
    
    return prompt


def get_num_classes(task_type: str) -> int:
    """
    Return the number of classes for each task type
    """
    num_classes = {
        "blobs": 3,
        "circles": 2,
        "linear": 2,
        "moons": 2
    }
    if task_type not in num_classes:
        raise ValueError(f"Unknown task type: {task_type}")
    
    return num_classes[task_type]


def is_duplicate(features, existing_features_list, tolerance=1e-5):
    """
    Check if features are duplicates of any in the existing list
    
    Args:
        features: Feature vector to check
        existing_features_list: List of existing feature vectors
        tolerance: Tolerance for floating point comparison
        
    Returns:
        True if duplicate, False otherwise
    """
    features_arr = np.array(features)
    for existing_features in existing_features_list:
        existing_arr = np.array(existing_features)
        if np.allclose(features_arr, existing_arr, rtol=tolerance, atol=tolerance):
            return True
    return False


def is_feature_duplicate(features, label, all_features_str_set, all_feature_vectors, icl_prompt_contents):
    """
    Check if a feature is a duplicate in various ways
    
    Args:
        features: Feature vector to check
        label: Label of the sample
        all_features_str_set: Set of string representations of features
        all_feature_vectors: List of feature vectors for numerical comparison
        icl_prompt_contents: Set of prompt contents for string matching
        
    Returns:
        True if duplicate, False otherwise
    """
    features_str = str(features)
    
    # Format feature string in various ways that could appear in prompts
    formatted_features = [
        f"[{features[0]:.6f}, {features[1]:.6f}]",  # Formatted with brackets
        f"[{features[0]}, {features[1]}]",  # With brackets, no formatting
        features_str,  # Raw string representation
        f"{features[0]:.3f}, {features[1]:.3f}",  # Just numbers with commas, 3 decimals
        f"{features[0]}, {features[1]}",  # Just numbers with commas, no formatting
        f"{features[0]:.6f}, {features[1]:.6f}"  # Just numbers with commas, 6 decimals
    ]
    
    # Check if this feature appears in any ICL prompt content
    is_duplicate_in_prompt = False
    
    # First check feature representation as string
    for feature_format in formatted_features:
        for prompt_content in icl_prompt_contents:
            if feature_format in prompt_content:
                is_duplicate_in_prompt = True
                return True
    
    # Also check for feature format that appears in the sample prompt
    feature_line = f"Features: {features[0]:.3f}, {features[1]:.3f}, Label: {label}"
    for prompt_content in icl_prompt_contents:
        if feature_line in prompt_content:
            return True
    
    # Check if feature string representation is in set or if numerically duplicate
    if features_str in all_features_str_set or is_duplicate(features, all_feature_vectors):
        return True
    
    return False


def sample_non_duplicate_examples(
    task_type: str, 
    num_samples: int, 
    noise_level: float, 
    flip_rate: float, 
    seed_value: int,
    all_features_str_set: set,
    all_feature_vectors: list,
    icl_prompt_contents: set,
    rng: np.random.RandomState,
    max_attempts: int = 10
) -> Tuple[List[Tuple], List, set]:
    """
    Sample a specified number of non-duplicate examples
    
    Args:
        task_type: Type of task (blobs, circles, linear, moons)
        num_samples: Number of samples to find
        noise_level: Noise level for data generation
        flip_rate: Label flip rate
        seed_value: Base seed value for reproducibility
        all_features_str_set: Set of existing feature string representations
        all_feature_vectors: List of existing feature vectors
        icl_prompt_contents: Set of existing prompt contents
        rng: Random state for shuffling
        max_attempts: Maximum number of attempts with different seeds
        
    Returns:
        Tuple containing:
            - List of tuples (features, label)
            - Updated list of all feature vectors
            - Updated set of all feature string representations
    """
    examples = []
    seed_offset = 0
    attempts = 0
    
    # Create local copies to update
    local_feature_vectors = list(all_feature_vectors)
    local_features_str_set = set(all_features_str_set)
    
    while len(examples) < num_samples and attempts < max_attempts:
        # Generate more samples than needed
        current_seed = seed_value + seed_offset
        examples_df = generate_test_data(
            task_type=task_type,
            num_samples=num_samples * 10,
            noise_level=noise_level,
            flip_rate=flip_rate,
            seed_value=current_seed
        )
        
        # Shuffle to get random examples
        examples_df = examples_df.sample(frac=1, random_state=rng)
        
        for _, row in examples_df.iterrows():
            if len(examples) >= num_samples:
                break
                
            features = row["features"]
            label = row["label"]
            
            if not is_feature_duplicate(features, label, local_features_str_set, local_feature_vectors, icl_prompt_contents):
                examples.append((features, label))
                local_feature_vectors.append(features)
                local_features_str_set.add(str(features))
        
        seed_offset += 1
        attempts += 1
    
    if len(examples) < num_samples:
        raise ValueError(f"Could only find {len(examples)} non-duplicate examples out of requested {num_samples} after {attempts} attempts")
    
    return examples, local_feature_vectors, local_features_str_set


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
    test_example_rng = np.random.RandomState(cfg.test_data_seed + 1) # different seed
    
    # Load tokenizer for token length calculation
    tokenizer = get_tokenizer(cfg.tokenizer_name)
    print(f"Loaded tokenizer: {cfg.tokenizer_name}")
    
    # Base paths
    deepseek_results_base = "/home/szhang967/liftr/analyze_deepseek/deepseek-resutls/deepseek-ai-deepseek-reasoner"
    
    # Initialize dataset list
    dataset = []
    
    # Check if all test_data_examples configs match the test_data task_type
    if cfg.test_data_examples.task_type != cfg.test_data.task_type:
        raise ValueError(f"All test_data_examples must have the same task_type as test_data. "
                        f"Found {cfg.test_data_examples.task_type} instead of {cfg.test_data.task_type}")
    
    # First, collect all ICL examples to check for duplicates later
    all_icl_examples = []
    icl_config_info = []
    icl_features_set = set()  # To store string representation of features for quick lookup
    icl_prompt_contents = set()  # To store prompt contents for string matching
    
    for icl_config in cfg.icl_examples:
        # Sample ICL examples for this configuration with token length check
        examples = sample_icl_examples(icl_config, deepseek_results_base, icl_rng, tokenizer)
        
        # Store feature representation for duplicate checking
        for example in examples:
            if "features" in example:
                features_str = str(example["features"])
                icl_features_set.add(features_str)
            
            # Extract and store prompt content for string matching
            try:
                prompt_content = extract_test_prompt_content(example)
                
                # Extract features from prompt content
                features_from_prompt = extract_features_from_prompt(prompt_content)
                icl_prompt_contents.update(features_from_prompt)
                
            except Exception as e:
                raise e
        
        # Add to our collection
        all_icl_examples.extend(examples)
        
        # Store config info for metadata
        if examples:  # Only add config info if we have examples
            icl_config_info.append({
                "task_type": icl_config.task_type,
                "shot_type": icl_config.shot_type,
                "reslen_type": icl_config.reslen_type,
                "nsamples_type": icl_config.nsamples_type,
                "noise_type": icl_config.noise_type,
                "flip_rate": icl_config.flip_rate,
                "num_examples": len(examples),
                "icl_example_maxlength": icl_config.icl_example_maxlength
            })
    
    # Shuffle all examples using the random state
    icl_rng.shuffle(all_icl_examples)
    
    # Track all generated features to avoid duplicates
    all_feature_vectors = []
    all_features_str_set = set()  # String representations for quick lookup
    
    # Extract and add actual feature vectors from ICL examples
    for example in all_icl_examples:
        features = example["features"]
        all_feature_vectors.append(features)
    
    # Add ICL features string representations to avoid overlap
    all_features_str_set.update(icl_features_set)
    
    # Sample test data points using the helper function
    print(f"Sampling {cfg.test_data.num_samples} test data points...")
    test_data, all_feature_vectors, all_features_str_set = sample_non_duplicate_examples(
        task_type=cfg.test_data.task_type,
        num_samples=cfg.test_data.num_samples,
        noise_level=cfg.test_data.noise_type,
        flip_rate=cfg.test_data.flip_rate,
        seed_value=cfg.test_data_seed,
        all_features_str_set=all_features_str_set,
        all_feature_vectors=all_feature_vectors,
        icl_prompt_contents=icl_prompt_contents,
        rng=test_rng
    )
    
    # Extract test features and labels
    test_data_features = [features for features, _ in test_data]
    test_data_labels = [label for _, label in test_data]
    
    # Sample additional examples for the instructions
    print(f"Sampling {cfg.test_data_examples.shot_type} examples for instructions...")
    test_examples, all_feature_vectors, all_features_str_set = sample_non_duplicate_examples(
        task_type=cfg.test_data_examples.task_type,
        num_samples=cfg.test_data_examples.shot_type,
        noise_level=cfg.test_data_examples.noise_type,
        flip_rate=cfg.test_data_examples.flip_rate,
        seed_value=cfg.test_data_seed + 100,  # Different seed base
        all_features_str_set=all_features_str_set,
        all_feature_vectors=all_feature_vectors,
        icl_prompt_contents=icl_prompt_contents,
        rng=test_example_rng
    )
    
    # Create base instruction
    instruction = create_instruction(cfg.test_data.task_type)
    
    # Get number of classes for the test task
    num_classes = get_num_classes(cfg.test_data.task_type)
    
    # For each test data point, create a prompt using the same shuffled ICL examples and test examples
    for idx, (test_features, test_label) in enumerate(zip(test_data_features, test_data_labels)):
        # Create prompt using the ICL examples, test examples, and test features
        prompt_text = create_prompt(instruction, all_icl_examples, test_examples, test_features, num_classes)
        
        # Calculate final prompt length in tokens
        prompt_token_length = calculate_token_length(prompt_text, tokenizer)
        print(f"Test example {idx+1}: Prompt length = {prompt_token_length} tokens")
        
        # Create dataset entry with the new format
        entry = {
            "data_source": cfg.test_data.task_type,
            "prompt": [{
                "role": "user",
                "content": prompt_text,
            }],
            "ability": "classification",
            "reward_model": {
                "style": "rule",
                "ground_truth": {
                    "features": test_features,
                    "label": int(test_label)
                }
            },
            "icl_example_meta_info": icl_config_info,
            "icl_examples": [json.dumps(ex) if isinstance(ex, dict) else str(ex) for ex in all_icl_examples],  # Convert to string to avoid parquet issues
            "test_examples": json.dumps(test_examples),  # Convert to JSON string for consistent serialization
            "test_data": {
                "task_type": cfg.test_data.task_type,
                "shot_type": cfg.test_data_examples.shot_type,
                "noise_type": cfg.test_data.noise_type,
                "flip_rate": cfg.test_data.flip_rate,
            },
            "extra_info": {
                'split': 'test',
                'index': idx,
                'prompt_token_length': prompt_token_length
            }
        }
        
        dataset.append(entry)
    
    # Create a descriptive filename
    filename = f"{cfg.test_data.task_type}_shot{cfg.test_data_examples.shot_type}_n{cfg.test_data.noise_type}_f{cfg.test_data.flip_rate}_test{cfg.test_data.num_samples}_icl{len(cfg.icl_examples)}_seed{cfg.icl_example_seed}.parquet"
    output_file = output_path / filename
    
    # Convert to DataFrame and save as parquet
    df = pd.DataFrame(dataset)
    
    # Ensure all object columns are serialized consistently for parquet
    try:
        df.to_parquet(str(output_file))
    except Exception as e:
        print(f"Failed to save directly to parquet: {e}")
        print("Attempting to save with more stringent serialization...")
        
        # Alternative approach: serialize all complex objects to ensure consistency
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].apply(lambda x: json.dumps(x) if not isinstance(x, (str, int, float)) else x)
        
        # Try saving again
        df.to_parquet(str(output_file))
    
    print(f"Created dataset with {len(dataset)} samples, saved to {output_file}")
    print(f"Max prompt token length: {max([entry['extra_info']['prompt_token_length'] for entry in dataset])}")


if __name__ == "__main__":
    main()
    