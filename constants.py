import os
root_dir = os.path.dirname(os.path.abspath(__file__))

data_dir = os.path.join(root_dir, 'datasets')


supported_llms = {
    "Qwen/Qwen2.5-3B-Instruct": {
        "template_type": "qwen-instruct",
    },
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": {
        "template_type": "qwen-instruct",
    },
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": {
        "template_type": "qwen-instruct",
    },
    "Qwen/Qwen2.5-32B-Preview": {
        "template_type": "qwen-instruct",
    }
}