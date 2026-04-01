# ReJump Codebase Handoff

> **Last updated**: 2026-03-30
> **Purpose**: Enable any agent (or future-me) to understand and run experiments without asking Yuchen.

---

## Architecture Overview

```
Raw LLM Response → ReJump Extractor (Gemini Pro) → JSON Tree + Walk → Metric Computation → Analysis
```

Three-stage pipeline:
1. **Inference** — generate LLM reasoning traces on a task
2. **Extraction** — parse each trace into tree (nodes) + jump (walk) using Gemini 2.5 Pro
3. **Analysis** — compute 6 metrics, visualize, compare models

---

## Directory Map

| Directory | What it does |
|---|---|
| `constants.py` | All config: `supported_llms` (25+ models), `supported_datasets`, path helpers |
| `environment.py` | API keys, paths — **never commit** |
| `rejump_extractor/` | **Core extraction pipeline** — one script per task type |
| `run_exps/` | Experiment orchestration — generates batch scripts |
| `examples/data_preprocess/` | Dataset prep — one module per dataset (math500.py, game24.py, aime.py, sudoku.py, etc.) |
| `datasets/` | Stored datasets (parquet files) |
| `results/` | LLM inference outputs + extracted trees + metrics (24 model subdirs) |
| `analysis/` | Jupyter notebooks for paper figures |
| `verl/` | RL training framework (PPO/GRPO) + inference engine + reward scoring |
| `icl_reasoning/` | In-context learning reasoning experiments |
| `figures/` | Generated PDF figures |
| `scripts/` | Misc shell scripts |

---

## Key Scripts

### Extraction (the core)

| Script | Task | CLI Example |
|---|---|---|
| `rejump_extractor/tree_vis_math_v3.py` | MATH-500, AIME, GSM8K, GPQA | `python -m rejump_extractor.tree_vis_math_v3 --dataset_name math500 --model_name "xai/grok-3-mini-beta" --num_samples -1 --temperature 0.0` |
| `rejump_extractor/tree_vis_game24.py` | Game of 24, Sudoku | `python -m rejump_extractor.tree_vis_game24 --dataset_name game24 --model_name "xai/grok-3-mini-beta" --num_samples 100 --temperature 0.0` |
| `rejump_extractor/tree_vis_sudoku.py` | Sudoku (same interface as game24) | `python -m rejump_extractor.tree_vis_sudoku --dataset_name sudoku --model_name "xai/grok-3-mini-beta"` |
| `rejump_extractor/compare_tree.py` | Tree similarity between models | Uses Zhang-Shasha TED + Jensen-Shannon divergence |
| `rejump_extractor/benchmark_acc.py` | Accuracy + token analysis | |

**Common CLI args for all extractors:**
- `--idx N1 N2 ...` — specific sample indices (default: all)
- `--model_name "model1" "model2"` — one or more models
- `--dataset_name` — task name
- `--num_samples N` — `-1` means all
- `--temperature T1 T2` — can specify multiple
- `--overwrite` — force re-extraction
- `--wandb` — log to W&B
- `--mode` — default, ricl_1, ricl_2, etc.
- `--corr_constraint` — None, 0 (wrong only), 1 (correct only)
- `--replicate_id` — replicate index

### Inference (generating LLM responses)

```bash
# API model
python -m verl.trainer.main_generation \
  data.path="datasets/math500/0_shot_1_query/reasoning_api/500_samples_None_noise_0.0_flip_rate_default_mode/test_default.parquet" \
  model.path="xai/grok-3-mini-beta" \
  data.output_path="results/xai-grok-3-mini-beta/math500_0_shot_1_query_reasoning_api_reslen_404_nsamples_-1_noise_None_flip_rate_0.0_mode_default/temperature_0.00/replicate_0/global_step_0/test_default.parquet" \
  rollout.temperature=0.0 \
  rollout.response_length=404
```

Or use the experiment generator:
```bash
python -m run_exps.create_exps \
  --dataset math500 \
  --model "xai/grok-3-mini-beta" \
  --mode reasoning \
  --shot 0 --n_samples -1 --n_query 1 \
  --temperature 0.0 \
  --exp_name rebuttal_math500
```

### Dataset Preparation

```bash
python -m examples.data_preprocess.math500 --template_type=reasoning_api --num_samples=500 --n_shot=0 --n_query=1
python -m examples.data_preprocess.game24 --template_type=reasoning_api --num_samples=100 --n_shot=0 --n_query=1
python -m examples.data_preprocess.aime --template_type=reasoning_api --num_samples=-1 --n_shot=0 --n_query=1
```

---

## Result Directory Structure

```
results/{model-name-with-dashes}/
  {dataset}_{shot}_shot_{n_query}_query_{template}_reslen_{len}_nsamples_{n}_noise_{noise}_flip_rate_{flip}_mode_{mode}/
    temperature_{T:.2f}/
      replicate_{R}/
        global_step_{S}/
          test_default.parquet      ← LLM responses
          tree_vis_v3/              ← Extracted trees (created by extractor)
            {idx}.json              ← Per-sample tree+walk JSON
            metric_df.csv           ← Aggregated metrics for all samples
```

**Model name mapping** (constants.py key → results dir name):
- `xai/grok-3-mini-beta` → `xai-grok-3-mini-beta`
- `deepseek-ai/deepseek-reasoner` → `deepseek-ai-deepseek-reasoner`
- `claude/claude-3-7-sonnet-20250219-thinking` → `claude-claude-3-7-sonnet-20250219-thinking`
- Rule: replace `/` with `-`

---

## Existing Results Inventory

### Models with results (in `results/`):

**Reasoning models (paper's main comparison):**
- `xai-grok-3-mini-beta` — math500, game24, gpqa-diamond, blobs, circles, moons
- `deepseek-ai-deepseek-reasoner` — math500, game24, and more
- `claude-claude-3-7-sonnet-20250219-thinking` — math500, game24, and more
- `openrouter-qwen-qwq-32b` — math500, game24
- `openrouter-microsoft-phi-4-reasoning-plus` — math500, game24
- `openrouter-deepseek-deepseek-r1-distill-qwen-14b` — math500, game24
- `openrouter-deepseek-deepseek-r1-distill-qwen-32b` — math500, game24
- `alibaba-qwq-plus-thinking` — math500

**Standard models (for LRM vs LLM comparison):**
- `openai-gpt-4o`, `openai-gpt-4o-mini-2024-07-18`
- `claude-claude-3-5-haiku-20241022`, `claude-claude-3-7-sonnet-20250219`
- `deepseek-ai-deepseek-chat`
- `google-gemini-2.0-flash`
- `alibaba-qwen2.5-14b-instruct`, `alibaba-qwen2.5-32b-instruct`

### Tree extraction results (`tree_vis_v3/` dirs exist):
- Grok: math500 (temp 0.0, 0.33, 0.66, 1.0), game24 (temp 0.0, 0.33, 0.66, 1.0)
- Other models: check with `find results/{model} -name "tree_vis_v3" -type d`

### Results already in W&B (from `results-tree-vis-v3.pkl`):

| Dataset | Models with results |
|---|---|
| **math500** | DeepSeek-R1, Grok 3 Mini, QwQ-32B, Phi-4-Reasoning+, Claude 3.7 Sonnet, DeepSeek-Chat, Gemini-2.0-Flash, Qwen2.5-14B/32B, R1-Distill-14B/32B |
| **game24** | All above + synthetic |

**Not in main pkl but in `results-tree-vis-v3-msr.pkl`**: Sudoku (Grok × 6 runs, QwQ × 1 NaN), ZebraLogic (Grok × 5 runs). Only Grok has actual metric values; QwQ sudoku is all NaN.

**AIME**: Being implemented and run by another agent (as of 2026-03-30).

### Local extraction results (`tree_vis_v3/metric_df.csv`):
- Only game24 for: Phi-4, QwQ-32B, DeepSeek-R1, Grok, synthetic
- Math500 results exist on W&B but not as local metric_df.csv
- Sudoku/ZebraLogic: results in W&B (`results-tree-vis-v3-msr.pkl`) — **only Grok model**, need more models for cross-model comparison

### Pickle caches:
- `results-tree-vis-v3.pkl` — aggregated metrics across models (used by analysis notebooks)
- `results-tree-vis-v3-msr.pkl` — MSR variant
- `results-tree-compare.pkl` — cross-model tree comparison
- `results-evaluation.pkl`, `results-generation.pkl` — generation/eval caches

---

## 6 ReJump Metrics (computed in extraction scripts)

| Metric | Variable | Definition |
|---|---|---|
| **#solution** | `average_solution_count` | Number of distinct derived solutions (leaf nodes reached via calc) |
| **d_jump** | `filtered_ajd` | Average tree distance between consecutive derived solution steps |
| **r_success** | `success_rate` | Fraction of derived solutions that are correct |
| **r_verify** | `average_verification_rate` | Fraction of all transitions labeled verify |
| **r_overthink** | `overthinking_rate` | Fraction of derived solution steps after the first correct one |
| **r_forget** | `forgetting_rate` | Binary: 1 if model revisits an already-derived leaf via calc |

---

## How to Run Rebuttal Experiments

### Task: Cross-model 6-metric comparison on Sudoku/ZebraLogic/AIME'26

**Step 1: Check if inference results exist**
```bash
# For each model, check if test_default.parquet exists
ls results/{model}/sudoku_0_shot_1_query_reasoning_api_reslen_404_nsamples_*_noise_None_flip_rate_0.0_mode_default/temperature_0.00/replicate_0/global_step_0/test_default.parquet
```

**Step 2: If not, generate responses**
```bash
python -m examples.data_preprocess.sudoku --template_type=reasoning_api --num_samples=100 --n_shot=0 --n_query=1
python -m verl.trainer.main_generation \
  data.path="datasets/sudoku/..." \
  model.path="xai/grok-3-mini-beta" \
  ...
```

**Step 3: Extract trees**
```bash
python -m rejump_extractor.tree_vis_sudoku \
  --dataset_name sudoku \
  --model_name "xai/grok-3-mini-beta" "deepseek-ai/deepseek-reasoner" "openrouter-qwen/qwq-32b" "claude/claude-3-7-sonnet-20250219-thinking" "openrouter-microsoft/phi-4-reasoning-plus" \
  --num_samples 100 \
  --temperature 0.0
```

**Step 4: Read metrics**
```python
import pandas as pd
df = pd.read_csv("results/{model}/{config}/.../tree_vis_v3/metric_df.csv")
# Columns: filtered_ajd, average_solution_count, success_rates, 
#           average_verification_rates, overthinking_rates, forgetting_rates
```

---

## Paper Models (Section 5 main comparison)

From the paper (Table 4 / main comparison):
1. **Grok 3 Mini Beta** — `xai/grok-3-mini-beta`
2. **DeepSeek-R1** — `deepseek-ai/deepseek-reasoner`
3. **QwQ-32B** — `openrouter-qwen/qwq-32b`
4. **Claude 3.7 Sonnet (thinking)** — `claude/claude-3-7-sonnet-20250219-thinking`
5. **Phi-4-Reasoning-Plus** — `openrouter-microsoft/phi-4-reasoning-plus`

---

## Aggregating Results (Pickle Files)

The analysis notebooks read from `results/results-tree-vis-v3.pkl` — a consolidated DataFrame.

**How it's built:** Download from W&B using `analysis/wandb_download.py`:
```bash
python -m analysis.wandb_download --task tree-vis-v3
# Creates: results/results-tree-vis-v3.pkl

python -m analysis.wandb_download --task tree-compare
# Creates: results/results-tree-compare.pkl
```

This requires W&B credentials in `environment.py` (WANDB_INFO dict).

**Alternatively**, you can build metrics locally from the per-sample CSV files:
```python
import pandas as pd, glob
files = glob.glob("results/*/math500_*/temperature_0.00/replicate_0/global_step_0/tree_vis_v3/metric_df.csv")
dfs = [pd.read_csv(f).assign(model=f.split("/")[1], config=f.split("/")[2]) for f in files]
combined = pd.concat(dfs)
```

---

## Important Branches

| Branch | Purpose | Key files |
|---|---|---|
| `develop` (current) | Latest working branch | `rejump_extractor/`, up-to-date math/game24 extractors |
| `origin/yz_dev` | Yuchen's dev — has ZebraLogic, Sudoku updates | `TTT/tree_vis_zebralogic.py`, `examples/data_preprocess/zebralogic.py`, `sudoku_6x6_500.jsonl` |
| `origin/wonjun_dev` | Wonjun's experiments | Benchmark notebooks, sensitivity analysis |
| `origin/shuibai_dev` | Shuibai's experiments | Additional analysis |

**To get ZebraLogic code onto develop:**
```bash
git checkout origin/yz_dev -- TTT/tree_vis_zebralogic.py examples/data_preprocess/zebralogic.py verl/utils/reward_score/zebralogic.py
# Note: TTT/ is old dir name → may need to move to rejump_extractor/
```

---

## Known Issues / Gotchas

1. **Code is "屎山"** — many dead ends, old experiments, classification/regression code mixed with reasoning. Focus only on `rejump_extractor/` and `analysis/benchmark_*.ipynb`.

2. **`num_samples=-1`** means "all samples in the dataset". For math500 that's 500, for game24 it's 100, etc.

3. **`response_length=404`** is a default placeholder — for API models, the actual response length is determined by the API, not this parameter.

4. **Extraction uses Gemini 2.5 Pro** by default — costs ~$0.02 per sample. Set in the extractor script's global variables (`model_pro`, `llm_pro`).

5. **W&B integration** — many scripts use W&B for logging. Use `--wandb` flag. Project: `rejump-tree-vis-v3`.

6. **OneDrive results** — some older results may be on Microsoft OneDrive (old MSR account) and not in this local repo. If a result seems missing, ask Yuchen.

7. **ZebraLogic** — NOT on `develop` branch! Lives on `origin/yz_dev` branch in `TTT/tree_vis_zebralogic.py` (1453 lines). Also has `examples/data_preprocess/zebralogic.py` and `verl/utils/reward_score/zebralogic.py`. To use: `git checkout origin/yz_dev -- TTT/tree_vis_zebralogic.py examples/data_preprocess/zebralogic.py verl/utils/reward_score/zebralogic.py`.
   - Note: on `yz_dev`, extractors are in `TTT/` dir (old name), not `rejump_extractor/`.

8. **AIME dataset** — exists in `datasets/aime/` and `constants.py` supports it, but extractor is `tree_vis_math_v3.py` (same as math500). Use `--dataset_name aime`.

9. **Sudoku extractor** — uses `tree_vis_sudoku.py` (or `tree_vis_game24.py --dataset_name sudoku`). Both support sudoku.

10. **Git submodule** — this `github/` directory is a git submodule inside the submission repo. The `.git` file points to the parent repo's `.git/modules/`.
