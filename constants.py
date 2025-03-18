import os
root_dir = os.path.dirname(os.path.abspath(__file__))

data_dir = os.path.join(root_dir, 'datasets')


supported_llms = {
    "Qwen/Qwen2.5-3B-Instruct": {
        "path": "Qwen/Qwen2.5-3B-Instruct",
        "model_type": "Qwen",
        "model_name": "Qwen2.5-3B-Instruct",
        "model_size": "3B",
        "model_type": "Qwen",
    },
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": {
        "path": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "model_type": "DeepSeek",
        "model_name": "DeepSeek-R1-Distill-Qwen-1.5B",
        "model_size": "1.5B",
        "model_type": "DeepSeek",
    }
}