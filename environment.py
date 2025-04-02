import os
root_dir = os.path.dirname(os.path.abspath(__file__))
 
OPENAI_API_KEY = {
    'yz': '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',
    'wk': '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',
}
HUGGINGFACE_API_KEY = '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'
CLAUDE_API_KEY = {
    'yz':'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',
}
GEMINI_API_KEY = {
    'yz': 'AIzaSyDa6IZiztMRZl_j2_em_MjruiCyZs9vEFs',
}
 
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