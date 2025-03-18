import pandas as pd
import os
from constants import supported_llms
from environment import root_dir

model_list = list(supported_llms.keys())

shot_list = [10, 50, 100, 200]

dataset_list = [
    "blobs"
]

def gen_dataset(
    dataset_name, 
    shot,
    template_type="qwen-instruct"
):
    return f"""
python examples/data_preprocess/{dataset_name}.py \
    --template_type={template_type} \
    --num_samples=1000 \
    --n_features=2 \
    --centers=3 \
    --cluster_std=1.0 \
    --test_ratio=0.2 \
    --n_shot={shot}
    """
    
def inference(
    dataset_name,
    shot,
    model_name,
    temperature=0.3,
):
    prompt_length = ((24 * shot + 185) // 1000 + 1) * 1000
    response_length = prompt_length // 2
    return f"""
python -m verl.trainer.main_generation \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=2 \
    data.path=datasets/{dataset_name}/{shot}.parquet \
    data.prompt_key=prompt \
    data.n_samples=1 \
    data.batch_size=128 \
    data.output_path=results/{dataset_name}/{model_name}_{shot}_gen_test.parquet \
    model.path={model_name} \
    +model.trust_remote_code=True \
    rollout.temperature={temperature} \
    rollout.top_k=10 \
    rollout.top_p=0.9 \
    rollout.prompt_length={prompt_length} \
    rollout.response_length={response_length} \
    rollout.tensor_model_parallel_size=2 \
    rollout.gpu_memory_utilization=0.8 \
    trainer.wandb=True
    """
    
def eval(
    dataset_name,
    shot,
    model_name,
):
    return f"""
python -m verl.trainer.main_eval \
    data.path=results/{dataset_name}/{model_name}_{shot}_gen_test.parquet \
    trainer.wandb=True
    """

os.makedirs(f"{root_dir}/run_exps/auto", exist_ok=True)
 
script_names = []
for dataset in dataset_list:
    for shot in shot_list:
        for model in model_list:
            template_type = supported_llms[model]["template_type"]
            gen_command = gen_dataset(dataset, shot, template_type)
            inference_command = inference(dataset, shot, model)
            eval_command = eval(dataset, shot, model)
            
            bash_script = "\n".join([gen_command, inference_command, eval_command])
            script_name = f"{dataset}_{shot}_{model}.sh".replace("/", "_")
            script_names.append(script_name)
            with open(f"{root_dir}/run_exps/auto/{script_name}", "w") as f:
                f.write(bash_script)

run_all_scripts = f"""
for script in {script_names}; do
    bash $script
done
"""

with open(f"{root_dir}/run_exps/auto/run_all.sh", "w") as f:
    f.write(run_all_scripts)
