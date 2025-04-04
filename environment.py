import os
root_dir = os.path.dirname(os.path.abspath(__file__))
 
OPENAI_API_KEY = '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'
HUGGINGFACE_API_KEY = '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'
ANTHROPIC_API_KEY = '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'
GEMINI_API_KEY = {
    'yz': 'AIzaSyDa6IZiztMRZl_j2_em_MjruiCyZs9vEFs',
}
DEEPSEEK_API_KEY = "sk-9e47420b26dd42af990d4c8f6e1bc16e"
OPENROUTER_API_KEY = "your-openrouter-api-key"
 
HF_HOME = "/data/yzeng58/.cache/huggingface"
TRANSFORMERS_CACHE = "/data/yzeng58/.cache/huggingface/hub"
TRITON_CACHE_DIR="/data/yzeng58/cache/triton"
 
WANDB_INFO = {
    'project': 'liftr',
    'entity': 'lee-lab-uw-madison'
}
 
CONDA_PATH = f"{os.path.dirname(root_dir)}/anaconda3/bin/conda"

VERTEXAI_INFO = {
    'project': 'your-project-id',
    'location': 'us-central1',
}