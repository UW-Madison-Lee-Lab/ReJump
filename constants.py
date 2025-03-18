import os
root_dir = os.path.dirname(os.path.abspath(__file__))

data_dir = os.path.join(root_dir, 'datasets')


supported_llms = {
    "Qwen/Qwen2.5-3B-Instruct": {
        "template_type": "qwen-instruct",
        "model_size": 3_000_000_000,
    },
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": {
        "template_type": "qwen-instruct",
        "model_size": 1_500_000_000,
    },
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": {
        "template_type": "qwen-instruct",
        "model_size": 7_000_000_000,
    },
    "Qwen/QwQ-32B-preview": {
        "template_type": "qwen-instruct",
        "model_size": 32_000_000_000,
    },
    "Qwen/Qwen2.5-1.5B-Instruct": {
        "template_type": "qwen-instruct",
        "model_size": 1_500_000_000,
    },
    "Qwen/Qwen2.5-7B-Instruct": {
        "template_type": "qwen-instruct",
        "model_size": 7_000_000_000,
    },
}