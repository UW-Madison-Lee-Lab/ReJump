import os, re
import pdb
from environment import root_dir, DEEPSEEK_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY, OPENROUTER_API_KEY, GEMINI_API_KEY, ALIBABA_API_KEY, XAI_API_KEY


data_dir = os.path.join(root_dir, 'datasets')

def get_model_name(
    dataset_name, 
    model_name, 
    shot,
    template_type, 
    response_length,
    num_samples,
    feature_noise,
    label_noise,
    data_mode,
    n_query=1,
):
    return f"{model_name.replace('/', '-')}/{dataset_name}_{shot}_shot_{n_query}_query_{template_type}_reslen_{response_length}_nsamples_{num_samples}_noise_{feature_noise}_flip_rate_{label_noise}_mode_{data_mode}"
def get_configs_via_model_name(model_name):
    pattern = r"(.+?)/(.+?)_(.+?)_shot_(.+?)_query_(.+)_reslen_(.+)_nsamples_(.+)_noise_(.+)_flip_rate_(.+)_mode_(.+)"
    match = re.match(pattern, model_name)
    
    if match:
        model_name = match.group(1)
        dataset_name = match.group(2)
        shot = match.group(3)
        n_query          = match.group(4)  #new
        template_type  = match.group(5)
        response_length= match.group(6)
        num_samples    = match.group(7)
        feature_noise  = match.group(8)
        label_noise    = match.group(9)
        data_mode      = match.group(10)

        return {
            "dataset_name": dataset_name,
            "model_name": model_name,
            "shot": shot,
            "n_query": n_query,
            "template_type": template_type,
            "response_length": response_length,
            "num_samples": num_samples,
            "feature_noise": feature_noise,
            "label_noise": label_noise,
            "data_mode": data_mode
        }
    else:
        raise ValueError(f"Invalid model name: {model_name}")
    
def get_model_dir(
    dataset_name, 
    model_name, 
    shot, 
    template_type, 
    response_length, 
    num_samples, 
    feature_noise,
    label_noise,
    data_mode,
    train_step = 0,
    n_query=1,
):
    return os.path.join(root_dir, 'checkpoints', 'TinyZero', get_model_name(dataset_name, model_name, shot, template_type, response_length, num_samples, feature_noise, label_noise, data_mode, n_query=n_query), "actor", f"global_step_{train_step}")
def get_configs_via_model_dir(model_dir):
    # Extract model name and train step from the model directory path using regex
    pattern = r".*TinyZero[/\\](.+)[/\\]actor[/\\]global_step_(\d+)$"
    match = re.match(pattern, model_dir)
    if match:
        model_name = match.group(1)  # The full model name with all parameters
        train_step = match.group(2)    # The training step number
    else:
        raise ValueError(f"Invalid model directory structure: {model_dir}")
    configs = get_configs_via_model_name(model_name)
    configs["train_step"] = int(train_step)
    return configs


def get_result_dir(
    dataset_name, 
    model_name, 
    shot, 
    template_type, 
    response_length,
    num_samples, 
    feature_noise,
    label_noise,
    train_step = 0,
    data_mode = "default",
    n_query=1,
    temperature =0,
    replicate_id = 0,
):
    return os.path.join(root_dir, 'results', get_model_name(dataset_name, model_name, shot, template_type, response_length, num_samples, feature_noise, label_noise, data_mode, n_query=n_query), f"temperature_{temperature:.2f}", f"replicate_{replicate_id}", f"global_step_{train_step}")
def get_configs_via_result_dir(result_dir):
    # Extract model name from the result directory path using regex
    pattern = r".*results[/\\](.+)[/\\]temperature_(.+)[/\\]replicate_(.+)[/\\]global_step_(\d+)$"
    match = re.match(pattern, result_dir)
    if match:
        model_name = match.group(1)  # The full model name with all parameters
        temperature = match.group(2)
        replicate_id = match.group(3)
        steps = match.group(4)      # The training step number
    else:
        raise ValueError(f"Invalid result directory structure: {result_dir}")
    configs = get_configs_via_model_name(model_name)
    configs["train_step"] = int(steps)
    configs["temperature"] = float(temperature)
    configs["replicate_id"] = int(replicate_id)
    return configs

    
    
def get_dataset_dir(
    dataset_name, 
    shot, 
    template_type, 
    num_samples, 
    feature_noise = 0,
    label_noise = 0,
    data_mode = "default",
    n_query=1,
):
    if "ricl" in template_type:
        ricl_shot = int(re.match(r".*?ricl_(\d+)", template_type).group(1))
        shot = f"{ricl_shot}*{shot}"
    return os.path.join(root_dir, 'datasets', dataset_name, f"{shot}_shot_{n_query}_query", template_type, f"{num_samples}_samples_{feature_noise}_noise_{label_noise}_flip_rate_{data_mode}_mode")
def get_configs_via_dataset_dir(dataset_dir):

    basename = dataset_dir.replace(f"{root_dir}/datasets/", "")
    pattern = r"(.+)/(.+)_shot_(\d+)_query/(.+)/(.+)_samples_(.+)_noise_(.+)_flip_rate_(.+)_mode"
    match = re.match(pattern, basename)
    if match:
        dataset_name   = match.group(1)
        shot           = match.group(2)
        n_query          = match.group(3)  #new
        template_type  = match.group(4)
        num_samples    = match.group(5)
        feature_noise  = match.group(6)
        label_noise    = match.group(7)
        data_mode      = match.group(8)
    else:
        return {}
    
    return {
        "dataset_name": dataset_name,
        "shot": shot,
        "n_query": n_query,
        "template_type": template_type,
        "num_samples": num_samples,
        "feature_noise": feature_noise,
        "label_noise": label_noise,
        "data_mode": data_mode
    }
    
def get_dataset_filename(
    split,
    data_mode,
):
    if "-" in data_mode:
        data_mode = "mixed"
    if data_mode == "grid":
        return "grid.parquet"
    else:
        return f"{split}_{data_mode}.parquet"
    

def get_mixed_configs(
    dataset_paths,
    dataset_ratios,
    num_samples,
):

    configs = dict()
    for dataset_path in dataset_paths:
        data_configs = get_configs_via_dataset_dir(dataset_path)
        for key in data_configs:
            if key in configs:
                configs[key] = f"{configs[key]}-{data_configs[key]}"
            else:
                configs[key] = data_configs[key]
           
    mix_str = ""
    for dataset_ratio in dataset_ratios:
        mix_str += f"{dataset_ratio}-"
    mix_str = mix_str[:-1]
            
    configs['dataset_name'] = configs['dataset_name']+"_mix_"+mix_str
    configs['num_samples'] = num_samples

    return configs
    
supported_llms = {
    # Qwen
    "Qwen/Qwen2.5-0.5B": {
        "template_type": "qwen-instruct",
        "model_size": 500_000_000,
        "type": "huggingface",
    },
    "Qwen/Qwen2.5-1.5B-Instruct": {
        "template_type": "qwen-instruct",
        "model_size": 1_500_000_000,
        "type": "huggingface",
    },
    "Qwen/Qwen2.5-3B-Instruct": {
        "template_type": "qwen-instruct",
        "model_size": 3_000_000_000,
        "type": "huggingface",
    },
    "Qwen/Qwen2.5-7B-Instruct": {
        "template_type": "qwen-instruct",
        "model_size": 7_000_000_000,
        "type": "huggingface",
    },
    "Qwen/Qwen2.5-14B-Instruct": {
        "template_type": "qwen-instruct",
        "model_size": 14_000_000_000,
        "type": "huggingface",
    },
    # QwQ
    "Qwen/QwQ-32B": {
        "template_type": "qwen-instruct",
        "model_size": 32_000_000_000,
        "type": "huggingface",
    },
    # DeepSeek
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": {
        "template_type": "qwen-instruct",
        "model_size": 1_500_000_000,
        "type": "huggingface",
    },
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": {
        "template_type": "qwen-instruct",
        "model_size": 7_000_000_000,
        "type": "huggingface",
    },
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": {
        "template_type": "qwen-instruct",
        "model_size": 8_000_000_000,
        "type": "huggingface",
    },
    # Llama
    "meta-llama/Llama-3.1-8B-Instruct": {
        "template_type": "qwen-instruct",
        "model_size": 8_000_000_000,
        "type": "huggingface",
    },
    # Deepseek api
    "deepseek-ai/deepseek-chat": {
        "template_type": "standard_api",
        "model_size": 0,
        "type": "api",
        "api_key": DEEPSEEK_API_KEY,
    },
    "deepseek-ai/deepseek-reasoner": {
        "template_type": "reasoning_api",
        "model_size": 0,
        "type": "api",
        "api_key": DEEPSEEK_API_KEY,
    },
    # OpenAI api
    "openai/gpt-4o": {
        "template_type": "standard_api",
        "model_size": 0,
        "type": "api",
        "api_key": OPENAI_API_KEY,
    },
    "openai/gpt-4o-mini-2024-07-18": {
        "template_type": "standard_api",
        "model_size": 0,
        "type": "api",
        "api_key": OPENAI_API_KEY,
    },
    "openai/o1-pro": {
        "template_type": "reasoning_api",
        "model_size": 0,
        "type": "api",
        "api_key": OPENAI_API_KEY,
    },
    "openai/o3-mini": {
        "template_type": "reasoning_api",
        "model_size": 0,
        "type": "api",
        "api_key": OPENAI_API_KEY,
    },
    "openrouter-deepseek/deepseek-r1": {
        "template_type": "reasoning_api",
        "model_size": 0,
        "type": "api",
        "api_key": OPENROUTER_API_KEY,
    },
    "claude/claude-3-7-sonnet-20250219": {
        "template_type": "standard_api",
        "model_size": 0,
        "type": "api",
        "api_key": ANTHROPIC_API_KEY,
    },
    "claude/claude-3-7-sonnet-20250219-thinking": {
        "template_type": "reasoning_api",
        "model_size": 0,
        "type": "api",
        "api_key": ANTHROPIC_API_KEY,
    },
    "claude/claude-3-5-haiku-20241022": {
        "template_type": "standard_api",
        "model_size": 0,
        "type": "api",
        "api_key": ANTHROPIC_API_KEY,
    },
    "google/gemini-2.0-flash": {
        "template_type": "standard_api",
        "model_size": 0,
        "type": "api",
        "api_key": GEMINI_API_KEY,
    },
    "google/gemini-2.0-flash-lite": {
        "template_type": "standard_api",
        "model_size": 0,
        "type": "api",
        "api_key": GEMINI_API_KEY,
    },
    "google/gemini-2.5-flash-preview-04-17": {
        "template_type": "standard_api",
        "model_size": 0,
        "type": "api",
        "api_key": GEMINI_API_KEY,
    },
    "google/gemini-2.5-pro-preview-03-25": {
        "template_type": "reasoning_api",
        "model_size": 0,
        "type": "api",
        "api_key": GEMINI_API_KEY,
    },
    "google/gemini-2.0-flash-thinking-exp": {
        "template_type": "reasoning_api",
        "model_size": 0,
        "type": "api",
        "api_key": GEMINI_API_KEY,
    },
    "openai/o1-pro": {
        "template_type": "reasoning_api",
        "model_size": 0,
        "type": "api",
        "api_key": OPENAI_API_KEY,
    },
    "openai/o3-mini": {
        "template_type": "reasoning_api",
        "model_size": 0,
        "type": "api",
        "api_key": OPENAI_API_KEY,
    },
    "openrouter-qwen/qwq-32b": {
        "template_type": "reasoning_api",
        "model_size": 0,
        "type": "api",
        "api_key": OPENROUTER_API_KEY,
    },
    "alibaba/qwen-plus-2025-04-28-thinking": {
        "template_type": "reasoning_api",
        "model_size": 0,
        "type": "api",
        "api_key": ALIBABA_API_KEY,
    },
    "alibaba/qwen-plus-2025-04-28": {
        "template_type": "standard_api",
        "model_size": 0,
        "type": "api",
        "api_key": ALIBABA_API_KEY,
    },
    "alibaba/qwq-plus-thinking": {
        "template_type": "reasoning_api",
        "model_size": 0,
        "type": "api",
        "api_key": ALIBABA_API_KEY,
    },
    "alibaba/qwen-turbo-2025-04-28-thinking": {
        "template_type": "reasoning_api",
        "model_size": 0,
        "type": "api",
        "api_key": ALIBABA_API_KEY,
    },
    "alibaba/qwen-turbo-2024-11-01": {
        "template_type": "standard_api",
        "model_size": 0,
        "type": "api",
        "api_key": ALIBABA_API_KEY,
    },
    "alibaba/qwen2.5-32b-instruct": {
        "template_type": "standard_api",
        "model_size": 0,
        "type": "api",
        "api_key": ALIBABA_API_KEY,
    },
    "alibaba/qwen2.5-14b-instruct": {
        "template_type": "standard_api",
        "model_size": 0,
        "type": "api",
        "api_key": ALIBABA_API_KEY,
    },
    "xai/grok-3-mini-beta": {
        "template_type": "reasoning_api",
        "model_size": 0,
        "type": "api",
        "api_key": XAI_API_KEY,
    },
    "openrouter-microsoft/phi-4-reasoning-plus": {
        "template_type": "standard_api",
        "model_size": 0,
        "type": "api",
        "api_key": OPENROUTER_API_KEY,
    },
    "openrouter-deepseek/deepseek-r1-distill-qwen-14b": {
        "template_type": "reasoning_api",
        "model_size": 0,
        "type": "api",
        "api_key": OPENROUTER_API_KEY,
    },
    "openrouter-deepseek/deepseek-r1-distill-qwen-32b": {
        "template_type": "reasoning_api",
        "model_size": 0,
        "type": "api",
        "api_key": OPENROUTER_API_KEY,
    },
}

supported_datasets = {
    "blobs": {
        "num_classes": 3,
        "num_features": 2,
        "feature_noise": 1.0,
        "label_noise": 0.0,
        "type": "classification",
        "difficulty": 1,
        "answer_format": "tags",
    },
    "moons": {
        "num_classes": 2,
        "num_features": 2,
        "feature_noise": 0.1,
        "label_noise": 0.0,
        "type": "classification",
        "difficulty": 2,
        "answer_format": "tags",
    },
    "linear": {
        "num_classes": 2,
        "num_features": 2,
        "feature_noise": 0.1,
        "label_noise": 0.0,
        "type": "classification",
        "difficulty": 1,
        "answer_format": "tags",
    },  
    "circles": {
        "num_classes": 2,
        "num_features": 2,
        "feature_noise": 0.01,
        "label_noise": 0.0,
        "type": "classification",
        "difficulty": 3,
        "answer_format": "tags",
    },
    "linreg": {
        "num_classes": None,
        "num_features": 2,
        "feature_noise": 0.1,
        "label_noise": 0.0,
        "type": "regression",
        "difficulty": 1,
        "answer_format": "tags",
    },
    "quadreg": {
        "num_classes": None,
        "num_features": 2,
        "feature_noise": 0.1,
        "label_noise": 0.0,
        "type": "regression",
        "difficulty": 2,
        "answer_format": "tags",
    },
    "expreg": {
        "num_classes": None,
        "num_features": 2,
        "feature_noise": 0.1,
        "label_noise": 0.0,
        "type": "regression",
        "difficulty": 1,
        "answer_format": "tags",
    },
    "cosreg": {
        "num_classes": None,
        "num_features": 2,
        "feature_noise": 0.02,
        "label_noise": 0.0,
        "type": "regression",
        "difficulty": 3,    
        "answer_format": "tags",
    },
    "l1normreg": {
        "num_classes": None,
        "num_features": 2,
        "feature_noise": 0.1,
        "label_noise": 0.0,
        "type": "regression",
        "difficulty": 2,
        "answer_format": "tags",
    },
    "pwreg": {
        "num_classes": None,
        "num_features": 2,
        "feature_noise": 0.1,
        "label_noise": 0.0,
        "type": "regression",
        "difficulty": 2,
        "answer_format": "tags",
    },
    "gsm8k": {
        "num_classes": None,
        "num_features": None,
        "feature_noise": None,
        "label_noise": None,
        "type": "other",
        "difficulty": None,
        "answer_format": "tags",
    },
    "aime": {
        "num_classes": None,
        "num_features": None,
        "feature_noise": None,
        "label_noise": None,
        "type": "other",
        "difficulty": None,
        "answer_format": "tags",
    },
    "math": {
        "num_classes": None,
        "num_features": None,
        "feature_noise": None,
        "label_noise": None,
        "type": "other",
        "difficulty": None,
        "answer_format": "box",
    },
    "math500": {
        "num_classes": None,
        "num_features": None,
        "feature_noise": None,
        "label_noise": None,
        "type": "other",
        "difficulty": None,
        "answer_format": "box",
    },
    "gpqa-diamond": {
        "num_classes": None,
        "num_features": None,
        "feature_noise": None,
        "label_noise": None,
        "type": "other",
        "difficulty": None,
        "answer_format": "tags",
    },
    "game24": {
        "num_classes": None,
        "num_features": None,
        "feature_noise": None,
        "label_noise": None,
        "type": "other",
        "difficulty": None,
        "answer_format": "tags",
    }
}