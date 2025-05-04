import os
root_dir = os.path.dirname(os.path.abspath(__file__))
 
OPENAI_API_KEY = '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'
HUGGINGFACE_API_KEY = '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'
ANTHROPIC_API_KEY = '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'
# GEMINI_API_KEY = "AIzaSyARTR4pSoM8hmIIMEg85OMHD1T9KgaGwV4"
# GEMINI_API_KEY = "AIzaSyA0vPcRshKiv5fftazBAbxdIHvtLyHCCiE"
# GEMINI_API_KEY = 'AIzaSyDQQKBdtRs8RN8RwHtfeg9aduThMFQ94jk'
# GEMINI_API_KEY = 'AIzaSyD8PQnKv6AYN_oUVmlNKG8dviEr6mO_J_0'
# GEMINI_API_KEY = 'AIzaSyBgM3S40tAiRJ5J1f-jx8xgecBmbelnPXg'
# GEMINI_API_KEY = 'AIzaSyCCnSBSjD1CgtighzPQyx03HZWvrVSWTHM'
# GEMINI_API_KEY = 'AIzaSyDQQKBdtRs8RN8RwHtfeg9aduThMFQ94jk'

# GEMINI_API_KEY = 'AIzaSyDIAq1UMSGx1-svNob46Rt616JF0UHW3VY'
GEMINI_API_KEY = 'AIzaSyCLqGz8Zebz15ITUZEoKwf7R-51jtibSss'
# DEEPSEEK_API_KEY = "sk-ce72d5c263804f29b988f2e8ab53fd62"
OPENROUTER_API_KEY = "your-openrouter-api-key"
 
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