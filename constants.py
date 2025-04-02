import os, re
import pdb
root_dir = os.path.dirname(os.path.abspath(__file__))

data_dir = os.path.join(root_dir, 'datasets')

def get_model_name(
    dataset_name, 
    model_name, 
    shot, 
    template_type, 
    response_length,
    num_samples,
    noise_level,
    label_flip_rate,
    data_mode
):
    return f"{model_name.replace('/', '-')}/{dataset_name}_{shot}_shot_{template_type}_reslen_{response_length}_nsamples_{num_samples}_noise_{noise_level}_flip_rate_{label_flip_rate}_mode_{data_mode}"
def get_configs_via_model_name(model_name):
    pattern = r"(.+?)/(.+?)_(.+?)_shot_(.+)_reslen_(.+)_nsamples_(.+)_noise_(.+)_flip_rate_(.+)_mode_(.+)"
    match = re.match(pattern, model_name)
    
    if match:
        dataset_name = match.group(2)
        shot = match.group(3)
        template_type = match.group(4)
        response_length = match.group(5)
        num_samples = match.group(6)
        noise_level = match.group(7)
        label_flip_rate = match.group(8)
        data_mode = match.group(9)
    

        return {
            "dataset_name": dataset_name,
            "model_name": model_name,
            "shot": shot,
            "template_type": template_type,
            "response_length": response_length,
            "num_samples": num_samples,
            "noise_level": noise_level,
            "label_flip_rate": label_flip_rate,
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
    noise_level,
    label_flip_rate,
    data_mode,
    train_step = 0,
):
    return os.path.join(root_dir, 'checkpoints', 'TinyZero', get_model_name(dataset_name, model_name, shot, template_type, response_length, num_samples, noise_level, label_flip_rate, data_mode), "actor", f"global_step_{train_step}")
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
    noise_level,
    label_flip_rate,
    train_step = 0,
    data_mode = "default",
):
    return os.path.join(root_dir, 'results', get_model_name(dataset_name, model_name, shot, template_type, response_length, num_samples, noise_level, label_flip_rate, data_mode), f"global_step_{train_step}")
def get_configs_via_result_dir(result_dir):
    # Extract model name from the result directory path using regex
    pattern = r".*results[/\\](.+)[/\\]global_step_(\d+)$"
    match = re.match(pattern, result_dir)
    if match:
        model_name = match.group(1)  # The full model name with all parameters
        steps = match.group(2)      # The training step number
    else:
        raise ValueError(f"Invalid result directory structure: {result_dir}")
    configs = get_configs_via_model_name(model_name)
    configs["train_step"] = int(steps)
    return configs

    
    
def get_dataset_dir(
    dataset_name, 
    shot, 
    template_type, 
    num_samples, 
    noise_level = 0,
    label_flip_rate = 0,
    data_mode = "default",
):
    return os.path.join(root_dir, 'datasets', dataset_name, f"{shot}_shot", template_type, f"{num_samples}_samples_{noise_level}_noise_{label_flip_rate}_flip_rate_{data_mode}_mode")
def get_configs_via_dataset_dir(dataset_dir):

    basename = dataset_dir.replace(f"{root_dir}/datasets/", "")
    pattern = r"(.+)/(\d+)_shot/(.+)/(.+)_samples_(.+)_noise_(.+)_flip_rate_(.+)_mode"
    match = re.match(pattern, basename)
    if match:
        dataset_name = match.group(1)
        shot = match.group(2)
        template_type = match.group(3)
        num_samples = match.group(4)
        noise_level = match.group(5)
        label_flip_rate = match.group(6)
        data_mode = match.group(7)
    else:
        return {}
    
    return {
        "dataset_name": dataset_name,
        "shot": shot,
        "template_type": template_type,
        "num_samples": num_samples,
        "noise_level": noise_level,
        "label_flip_rate": label_flip_rate,
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
    },
    "Qwen/Qwen2.5-1.5B-Instruct": {
        "template_type": "qwen-instruct",
        "model_size": 1_500_000_000,
    },
    "Qwen/Qwen2.5-3B-Instruct": {
        "template_type": "qwen-instruct",
        "model_size": 3_000_000_000,
    },
    "Qwen/Qwen2.5-7B-Instruct": {
        "template_type": "qwen-instruct",
        "model_size": 7_000_000_000,
    },
    "Qwen/Qwen2.5-14B-Instruct": {
        "template_type": "qwen-instruct",
        "model_size": 14_000_000_000,
    },
    # QwQ
    "Qwen/QwQ-32B-preview": {
        "template_type": "qwen-instruct",
        "model_size": 32_000_000_000,
    },
    # DeepSeek
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": {
        "template_type": "qwen-instruct",
        "model_size": 1_500_000_000,
    },
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": {
        "template_type": "qwen-instruct",
        "model_size": 7_000_000_000,
    },
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": {
        "template_type": "qwen-instruct",
        "model_size": 8_000_000_000,
    },
    # Llama
    "meta-llama/Llama-3.1-8B-Instruct": {
        "template_type": "qwen-instruct",
        "model_size": 8_000_000_000,
    },
}