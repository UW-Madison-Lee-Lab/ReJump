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
    shot_type: int = MISSING  # 50, 100
    nsamples_type: int = MISSING  # 500
    num_examples: int = MISSING  # Number of examples to use


@dataclass
class ICLReasoningConfig:
    icl_examples: List[ICLExampleConfig] = field(default_factory=list)
    test_data: TestDataConfig = MISSING
    test_data_examples: TestDataExampleConfig = MISSING
    icl_example_seed: int = 42
    test_data_seed: int = 42
    output_path: str = "/staging/szhang967/icl_datasets"


cs = ConfigStore.instance()
cs.store(name="config", node=ICLReasoningConfig)
cs.store(group="icl_examples", name="example_config", node=ICLExampleConfig)
cs.store(group="test_data", name="test_config", node=TestDataConfig)
cs.store(group="test_data_examples", name="test_data_example_config", node=TestDataExampleConfig)


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
        if len(results) == 0:
            raise ValueError(f"No examples found for configuration: {config}")
        print(f"Warning: Requested {config.num_examples} examples but only {len(results)} available. Using all available examples.")
        return results
    
    return rng.choice(results, config.num_examples, replace=False).tolist()


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
        # Sample ICL examples for this configuration
        examples = sample_icl_examples(icl_config, deepseek_results_base, icl_rng)
        
        # Add metadata about these examples
        for example in examples:
            example['config_info'] = {
                "task_type": icl_config.task_type,
                "shot_type": icl_config.shot_type,
                "reslen_type": icl_config.reslen_type,
                "nsamples_type": icl_config.nsamples_type,
                "noise_type": icl_config.noise_type,
                "flip_rate": icl_config.flip_rate
            }
            
            # Store feature representation for duplicate checking
            if "features" in example:
                features_str = str(example["features"])
                icl_features_set.add(features_str)
            
            # Extract and store prompt content for string matching
            try:
                prompt_content = extract_test_prompt_content(example)
                
                # Method 1: Extract features from [brackets] after "features" (case insensitive)
                for keyword in ["features", "Features"]:
                    prompt_parts = prompt_content.split(keyword)
                    for part in prompt_parts[1:]:  # Skip the first part before "features"
                        # Try to extract feature values using a simple approach
                        if "[" in part and "]" in part:
                            feature_str = part[part.find("["):part.find("]")+1]
                            icl_prompt_contents.add(feature_str)
                
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
                                icl_prompt_contents.add(feature_str)
                                # Also add other formats of the same value
                                icl_prompt_contents.add(f"[{x:.6f}, {y:.6f}]")
                                icl_prompt_contents.add(str([x, y]))
                            except ValueError:
                                # Skip if conversion to float fails
                                pass
                
                # Store the entire prompt for more thorough comparison later
                icl_prompt_contents.add(prompt_content)
                
            except Exception as e:
                print(f"Warning: Could not extract prompt content for duplicate checking: {e}")
        
        # Add to our collection
        all_icl_examples.extend(examples)
        
        # Store config info for metadata
        icl_config_info.append({
            "task_type": icl_config.task_type,
            "shot_type": icl_config.shot_type,
            "reslen_type": icl_config.reslen_type,
            "nsamples_type": icl_config.nsamples_type,
            "noise_type": icl_config.noise_type,
            "flip_rate": icl_config.flip_rate,
            "num_examples": len(examples)
        })
    
    # Shuffle all examples using the random state
    icl_rng.shuffle(all_icl_examples)
    
    # Generate test data using data generation functions
    test_data_df = generate_test_data(
        task_type=cfg.test_data.task_type,
        num_samples=1000,  # Generate a larger pool of samples to choose from
        noise_level=cfg.test_data.noise_type,
        flip_rate=cfg.test_data.flip_rate,
        seed_value=cfg.test_data_seed
    )
    
    if len(test_data_df) < cfg.test_data.num_samples:
        raise ValueError(f"Not enough test data samples: {len(test_data_df)} < {cfg.test_data.num_samples}")
    
    # Track all generated features to avoid duplicates
    all_feature_vectors = []
    all_features_str_set = set()  # String representations for quick lookup
    
    # Sample test data points and keep track of their features
    if len(test_data_df) <= cfg.test_data.num_samples:
        test_indices = list(range(len(test_data_df)))
    else:
        test_indices = test_rng.choice(len(test_data_df), cfg.test_data.num_samples, replace=False)
    
    test_features_list = [test_data_df.iloc[idx]["features"] for idx in test_indices]
    all_feature_vectors.extend(test_features_list)
    
    # Add string representations of test features for quick lookup
    for features in test_features_list:
        all_features_str_set.add(str(features))
    
    # Also add ICL features to avoid overlap
    all_features_str_set.update(icl_features_set)
    
    # Generate test examples that don't overlap with test data or ICL examples
    seed_offset = 100  # Start with a different seed for test examples
    test_examples = []
    
    while len(test_examples) < cfg.test_data_examples.num_examples:
        # Try with different seeds until we find enough non-duplicate examples
        test_examples_df = generate_test_data(
            task_type=cfg.test_data_examples.task_type,
            num_samples=cfg.test_data_examples.num_examples * 10,  # Generate more than we need
            noise_level=cfg.test_data_examples.noise_type,
            flip_rate=cfg.test_data_examples.flip_rate,
            seed_value=cfg.test_data_seed + seed_offset
        )
        
        # Shuffle to get random examples
        test_examples_df = test_examples_df.sample(frac=1, random_state=test_example_rng)
        
        for _, row in test_examples_df.iterrows():
            if len(test_examples) >= cfg.test_data_examples.num_examples:
                break
            
            features = row["features"]
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
                        break
                if is_duplicate_in_prompt:
                    break
            
            # Also check for feature format that appears in the sample prompt
            feature_line = f"Features: {features[0]:.3f}, {features[1]:.3f}, Label: {row['label']}"
            for prompt_content in icl_prompt_contents:
                if feature_line in prompt_content:
                    is_duplicate_in_prompt = True
                    break
            
            if not is_duplicate_in_prompt and features_str not in all_features_str_set and not is_duplicate(features, all_feature_vectors):
                test_examples.append((features, row["label"]))
                all_feature_vectors.append(features)
                all_features_str_set.add(features_str)
        
        # If we still need more examples, try with a different seed
        seed_offset += 1
        
        if seed_offset > 200:  # Avoid infinite loop
            raise ValueError(f"Could not generate enough non-duplicate test examples for {cfg.test_data_examples.task_type}")
    
    # Create base instruction
    instruction = create_instruction(cfg.test_data.task_type)
    
    # Get number of classes for the test task
    num_classes = get_num_classes(cfg.test_data.task_type)
    
    # For each test data point, create a prompt using the same shuffled ICL examples and test examples
    for idx, test_idx in enumerate(test_indices):
        test_row = test_data_df.iloc[test_idx]
        test_features = test_row["features"]
        
        # Create prompt using the ICL examples, test examples, and test features
        prompt = create_prompt(instruction, all_icl_examples, test_examples, test_features, num_classes)
        
        # Create dataset entry
        entry = {
            "prompt": prompt,
            "features": test_row["features"],
            "label": test_row["label"] if "label" in test_row else None,
            "icl_example_meta_info": icl_config_info,
            "icl_examples": [json.dumps(ex) if isinstance(ex, dict) else str(ex) for ex in all_icl_examples],  # Convert to string to avoid parquet issues
            "test_examples": json.dumps(test_examples),  # Convert to JSON string for consistent serialization
            "test_data": {
                "task_type": cfg.test_data.task_type,
                "shot_type": cfg.test_data_examples.shot_type,
                "nsamples_type": cfg.test_data_examples.nsamples_type,
                "noise_type": cfg.test_data.noise_type,
                "flip_rate": cfg.test_data.flip_rate,
            }
        }
        
        dataset.append(entry)
    
    # Create a descriptive filename
    filename = f"{cfg.test_data.task_type}_{cfg.test_data_examples.shot_type}shot_n{cfg.test_data.noise_type}_f{cfg.test_data.flip_rate}_test{cfg.test_data.num_samples}_icl{len(cfg.icl_examples)}_seed{cfg.icl_example_seed}.parquet"
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


if __name__ == "__main__":
    main()
    