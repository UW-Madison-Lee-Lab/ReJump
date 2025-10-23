# Sudoku Support - Tree Visualization Modifications

## 📋 Modification Summary

Added Sudoku support to `TTT/tree_vis_game24.py` while **maintaining full backward compatibility with Game24**.

---

## 🔧 Main Changes

### 1️⃣ **New Functions**

#### `get_tree_prompt_sudoku(input_str, output_str)` (Lines 29-81)
- **Purpose**: Generate tree structure conversion prompt for Sudoku
- **Key Differences**:
  - Describes Sudoku grid states instead of mathematical expressions
  - Root node: "Initial grid: 0204\\n3401\\n0012\\n2043"
  - Intermediate nodes: "Filled row 2, column 3 with 2"
  - Leaf nodes: Complete solution grid
  
#### `get_walk_prompt_sudoku(input_str, output_str, tree_json)` (Lines 138-186)
- **Purpose**: Generate reasoning path analysis prompt for Sudoku
- **Key Differences**:
  - Focuses on cell filling and constraint checking
  - Simplified categories: calculation/derivation, backtracking, verification
  - Adapted to Sudoku-specific reasoning patterns

### 2️⃣ **Modified Functions**

#### `get_analysis()` (Line 689)
Added `dataset_name` parameter to select prompts based on dataset type:

```python
# Lines 710-714: Select tree prompt
if dataset_name == "sudoku":
    tree_prompt = get_tree_prompt_sudoku(input_str, output_str)
else:
    tree_prompt = get_tree_prompt(input_str, output_str)

# Lines 721-725: Select walk prompt
if dataset_name == "sudoku":
    walk_prompt = get_walk_prompt_sudoku(input_str, output_str, tree_json)
else:
    walk_prompt = get_walk_prompt(input_str, output_str, tree_json)
```

### 3️⃣ **Dynamic Import** (Lines 778-784)
Dynamically import the corresponding `compare_answer` function based on `dataset_name`:

```python
if args.dataset_name == "game24":
    from verl.utils.reward_score.game24 import compare_answer
elif args.dataset_name == "sudoku":
    from verl.utils.reward_score.sudoku import compare_answer
```

---

## 🎯 Sudoku Tree Prompt Design Features

### Node Structure

| Node Type | Game24 Example | Sudoku Example |
|-----------|----------------|----------------|
| **Root** | `"9,3,12,8"` | `"Initial grid: 0204\\n3401\\n0012\\n2043"` |
| **Intermediate** | `"9-3, 12, 8"` | `"Filled row 2, column 3 with 2"` |
| **Leaf** | `"(9-3)*(12/8)"` | `"Complete grid: 1234\\n3421\\n4312\\n2143"` |

### Reasoning Types

- **calculation/derivation**: Filling cells or logical deduction
- **backtracking**: Discovering errors, reverting to previous state to try different values
- **verification**: Checking row/column/box constraints

---

## ✅ Compatibility Guarantee

### Game24 (Original functionality, completely unchanged)
```bash
# Default behavior, dataset_name defaults to "game24"
python TTT/tree_vis_game24.py \
    --model_name deepseek-ai/deepseek-reasoner \
    --num_samples 10 \
    --idx 0 1 2

# Explicit specification (equivalent to above)
python TTT/tree_vis_game24.py \
    --dataset_name game24 \
    --model_name deepseek-ai/deepseek-reasoner \
    --num_samples 10
```

### Sudoku (New functionality)
```bash
python TTT/tree_vis_game24.py \
    --dataset_name sudoku \
    --model_name openrouter-qwen/qwq-32b \
    --num_samples 10 \
    --temperature 0.0 \
    --idx 0 1 2
```

---

## 📊 Typical Sudoku Reasoning Tree Example

### Input (reasoning snippet)
```
Looking at the second row: 3 4 0 1. The missing number here is the third position.
The existing numbers are 3,4,1. So the missing one must be 2.
Let me check the column for that position too...
```

### Generated Tree Structure
```json
{
  "node1": {
    "Problem": "Initial grid: 0204\n3401\n0012\n2043",
    "parent": "none",
    "Result": null
  },
  "node2": {
    "Problem": "Filled row 2, column 3 with 2",
    "parent": "node1",
    "Result": null
  },
  "node3": {
    "Problem": "Complete grid: 1234\n3421\n4312\n2143",
    "parent": "node2",
    "Result": "1234\n3421\n4312\n2143"
  }
}
```

### Generated Walk
```json
[
  {"from": "node1", "to": "node2", "category": "calculation/derivation"},
  {"from": "node2", "to": "node3", "category": "calculation/derivation"}
]
```

---

## 🔍 Key Design Decisions

1. **Keep original functions unchanged**: `get_tree_prompt()` and `get_walk_prompt()` are completely unmodified
2. **Add parallel functions**: Added `*_sudoku` versions instead of modifying original functions
3. **Runtime selection**: Use if-else to select which version to use at runtime
4. **Default value maintains compatibility**: `dataset_name="game24"` ensures original behavior when not specified

---

## 📝 Required Additional Configuration

Ensure the following files are properly configured:

1. ✅ `constants.py`: Added `"sudoku"` configuration
2. ✅ `examples/data_preprocess/sudoku.py`: Data preprocessing script
3. ✅ `verl/utils/reward_score/sudoku.py`: Answer verification function
4. ✅ `examples/data_preprocess/helper.py`: Reward score mapping

---

## 🚀 Complete Workflow Example

```bash
# Step 1: Generate sudoku dataset
cd /data/szhang967/ReJump
PYTHONPATH=/data/szhang967/ReJump:$PYTHONPATH \
python examples/data_preprocess/sudoku.py \
    --num_samples 10 \
    --n_shot 0 \
    --template_type reasoning_api

# Step 2: Run model inference
python -m verl.trainer.main_generation \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=1 \
    data.path=datasets/sudoku/0_shot_1_query/reasoning_api/10_samples_None_noise_0.0_flip_rate_default_mode/test_default.parquet \
    data.output_path=results/openrouter-qwen_qwq-32b/sudoku_0_shot_1_query_reasoning_api_reslen_8192_nsamples_10_noise_None_flip_rate_0.0_mode_default/temperature_0.00/replicate_0/global_step_0/test_default.parquet \
    model.path=openrouter-qwen/qwq-32b \
    ...

# Step 3: Analyze reasoning tree
python TTT/tree_vis_game24.py \
    --dataset_name sudoku \
    --model_name openrouter-qwen/qwq-32b \
    --num_samples 10 \
    --temperature 0.0 \
    --analysis_model google/gemini-2.5-pro-preview-03-25 \
    --idx 0 1 2 3 4
```

---

## 📈 Output Content

After analysis completion, the following will be generated:

```
results/.../tree_vis_google_gemini-2.5-pro-preview-03-25/
├── 0.json          # Tree + path JSON for sample 0
├── 0.pdf           # Visualization for sample 0
├── 1.json          # Tree + path JSON for sample 1
├── 1.pdf           # Visualization for sample 1
├── ...
└── metric_df.csv   # Metrics summary for all samples
```

Each JSON contains:
- `tree`: Reasoning tree structure
- `walk`: Reasoning path steps
- `corr`: Whether the answer is correct (0/1)

---

## 🎉 Summary

Successfully added Sudoku support to `tree_vis_game24.py` with minimal changes:
- ✅ Original Game24 functionality completely unaffected
- ✅ Full Sudoku support added
- ✅ Clean code, easy to maintain and extend
- ✅ Can easily add more task types (e.g., GSM8K, MATH, etc.)
