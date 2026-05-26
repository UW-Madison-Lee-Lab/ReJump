"""Local configuration template for ReJump.

Copy this file to environment.py and fill in the values needed for your run.
Never commit environment.py.
"""

import os

root_dir = os.path.dirname(os.path.abspath(__file__))

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
HUGGINGFACE_API_KEY = os.environ.get("HUGGINGFACE_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
XAI_API_KEY = os.environ.get("XAI_API_KEY", "")
ALIBABA_API_KEY = os.environ.get("ALIBABA_API_KEY", "")

HF_HOME = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
TRANSFORMERS_CACHE = os.environ.get(
    "TRANSFORMERS_CACHE", os.path.join(HF_HOME, "hub")
)
TRITON_CACHE_DIR = os.environ.get(
    "TRITON_CACHE_DIR", os.path.expanduser("~/.cache/triton")
)

WANDB_INFO = {
    "project": os.environ.get("WANDB_PROJECT", "rejump"),
    "entity": os.environ.get("WANDB_ENTITY", ""),
}

CONDA_PATH = os.environ.get("CONDA_PATH", "conda")

VERTEXAI_INFO = {
    "project": os.environ.get("VERTEXAI_PROJECT", "your-project-id"),
    "location": os.environ.get("VERTEXAI_LOCATION", "us-central1"),
}
