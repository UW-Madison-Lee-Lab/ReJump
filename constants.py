import os, re
root_dir = os.path.dirname(os.path.abspath(__file__))

data_dir = os.path.join(root_dir, 'datasets')

def get_model_name(dataset_name, model_name, shot, template_type, response_length):
    return f"{model_name.replace('/', '_')}_{dataset_name}_{shot}_shot_{template_type}_reslen_{response_length}"
def get_model_dir(dataset_name, model_name, shot, template_type, response_length):
    return os.path.join(root_dir, 'checkpoints', 'TinyZero', get_model_name(dataset_name, model_name, shot, template_type, response_length))

def get_result_dir(dataset_name, model_name, shot, template_type, response_length):
    return os.path.join(root_dir, 'results', dataset_name, f"{model_name.replace('/', '_')}_{shot}_shot_{template_type}_reslen_{response_length}")
def get_configs_via_result_dir(result_dir):
    basename = os.path.basename(result_dir)
    dirname = os.path.dirname(result_dir)
    dataset_name = os.path.basename(dirname)
    
    pattern = r"(.+)_(\d+)_shot_(.+)_reslen_(.+)"
    match = re.match(pattern, basename)
    
    if match:
        model_name = match.group(1)
        shot = int(match.group(2))
        template_type = match.group(3)
        response_length = int(match.group(4))
        return {
            "dataset_name": dataset_name,
            "model_name": model_name,
            "shot": shot,
            "template_type": template_type,
            "response_length": response_length
        }
    else:
        return {}
    
def get_dataset_dir(dataset_name, shot, template_type):
    return os.path.join(root_dir, 'datasets', dataset_name, f"{shot}_shot", template_type)
def get_configs_via_dataset_dir(dataset_dir):
    basename = os.path.basename(dataset_dir)
    dirname = os.path.dirname(dataset_dir)
    dataset_name = os.path.basename(dirname)

    pattern = r"(.+)_(\d+)_shot_(.+)"
    match = re.match(pattern, basename)
    if match:
        shot = int(match.group(2))
        template_type = match.group(3)
        return {
            "dataset_name": dataset_name,
            "shot": shot,
            "template_type": template_type
        }
    else:
        return {}


supported_llms = {
    # Qwen
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