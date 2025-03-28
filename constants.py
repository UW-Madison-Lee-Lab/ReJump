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
    label_flip_rate
):
    return f"{model_name.replace('/', '-')}_{dataset_name}_{shot}_shot_{template_type}_reslen_{response_length}_nsamples_{num_samples}_noise_{noise_level}_flip_rate_{label_flip_rate}"
def get_configs_via_model_name(model_name):
    pattern = r"(.+?)_(.+?)_(\d+)_shot_(.+)_reslen_(.+)_nsamples_(.+)_noise_(.+)_flip_rate_(.+)"
    match = re.match(pattern, model_name)
    if match:
        return {
            "dataset_name": match.group(1),
            "model_name": match.group(2),
            "shot": int(match.group(3)),
            "template_type": match.group(4),
            "response_length": int(match.group(5)),
            "num_samples": int(match.group(6)),
            "noise_level": float(match.group(7)),
            "label_flip_rate": float(match.group(8)),
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
    train_step = 0,
):
    return os.path.join(root_dir, 'checkpoints', 'TinyZero', get_model_name(dataset_name, model_name, shot, template_type, response_length, num_samples, noise_level, label_flip_rate), "actor", f"global_step_{train_step}")
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
):
    return os.path.join(root_dir, 'results', get_model_name(dataset_name, model_name, shot, template_type, response_length, num_samples, noise_level, label_flip_rate), f"global_step_{train_step}")
def get_configs_via_result_dir(result_dir):
    steps = os.path.basename(result_dir).split("_")[-1]
    dirname = os.path.dirname(result_dir)
    model_name = os.path.basename(dirname)
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
):
    return os.path.join(root_dir, 'datasets', dataset_name, f"{shot}_shot", template_type, f"{num_samples}_samples_{noise_level}_noise_{label_flip_rate}_flip_rate")
def get_configs_via_dataset_dir(dataset_dir):

    basename = dataset_dir.replace(f"{root_dir}/datasets/", "")
    pattern = r"(.+)/(\d+)_shot/(.+)/(.+)_samples_(.+)_noise_(.+)_flip_rate"
    match = re.match(pattern, basename)
    if match:
        dataset_name = match.group(1)
        shot = match.group(2)
        template_type = match.group(3)
        num_samples = match.group(4)
        noise_level = match.group(5)
        label_flip_rate = match.group(6)
    else:
        return {}
    
    return {
        "dataset_name": dataset_name,
        "shot": shot,
        "template_type": template_type,
        "num_samples": num_samples,
        "noise_level": noise_level,
        "label_flip_rate": label_flip_rate
    }

def get_mixed_dataset_dir(
    dataset_paths,
    dataset_ratios,
    num_samples,
):

    configs = dict()
    for dataset_path in dataset_paths:
        for key in get_configs_via_dataset_dir(dataset_path):
            if key in configs:
                configs[key] = f"{configs[key]}_{get_configs_via_dataset_dir(dataset_path)[key]}"
            else:
                configs[key] = get_configs_via_dataset_dir(dataset_path)[key]
                
    mix_str = "_".join(dataset_ratios)
            
    output_dir = get_dataset_dir(
        configs['dataset_name']+"_mix_"+mix_str,
        configs['shot'],
        configs['template_type'],
        num_samples,
        configs['noise_level'],
        configs['label_flip_rate']
    )
    
    return output_dir
    
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