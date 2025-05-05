import os
root_dir = os.path.dirname(os.path.abspath(__file__))
 
OPENAI_API_KEY = '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'
HUGGINGFACE_API_KEY = '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'
ANTHROPIC_API_KEY = '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'
GEMINI_API_KEY = "AIzaSyBUhgp-FBViNj8VTxM3Tw8gXJsARgyx-dc"
# DEEPSEEK_API_KEY = "sk-14609c4a9bb04da6888a7299323cc0e7"
DEEPSEEK_API_KEY = "sk-2180124961a141829e0cf37c9e0ed30f"
# OPENROUTER_API_KEY = "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"
OPENROUTER_API_KEY = "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"
ALIBABA_API_KEY = "sk-10a48e129c8840869cad48efa799022e"
XAI_API_KEY = "your-xai-api-key"

HF_HOME = "/data/yzeng58/.cache/huggingface"
TRANSFORMERS_CACHE = "/data/yzeng58/.cache/huggingface/hub"
TRITON_CACHE_DIR="/data/yzeng58/cache/triton"

WANDB_INFO = {
    'project': 'liftr-generation',
    'entity': 'lee-lab-uw-madison'
}
 
CONDA_PATH = f"{os.path.dirname(root_dir)}/anaconda3/bin/conda"

VERTEXAI_INFO = {
    'project': 'your-project-id',
    'location': 'us-central1',
}