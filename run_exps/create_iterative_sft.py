import pandas as pd
import os
from constants import supported_llms
from environment import root_dir
import argparse

model_size_upper_limit = 10_000_000_000
supported_model_list = [model for model in supported_llms.keys() if supported_llms[model]["model_size"] <= model_size_upper_limit]

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, nargs="+", default=["blobs"], choices=["blobs", "moons", "linear"])
parser.add_argument("--model", type=str, nargs="+", default=supported_model_list, choices=supported_model_list)
parser.add_argument("--max_iterations", type=int, default=10)
parser.add_argument("--num_responses", type=int, default=5)
parser.add_argument("--shot", type=int, default=10, help="Number of examples in few-shot learning")
parser.add_argument("--num_samples", type=int, default=1000, help="Number of samples in the dataset")
parser.add_argument("--total_epochs", type=int, default=4, help="Number of training epochs")
args = parser.parse_args()

dataset_list = args.dataset
model_list = args.model
max_iterations = args.max_iterations
num_responses = args.num_responses
shot = args.shot
num_samples = args.num_samples
total_epochs = args.total_epochs

def gen_dataset(
    dataset_name, 
    shot,
    template_type="qwen-instruct"
):
    if dataset_name == "blobs":
        return f"""
        python {root_dir}/examples/data_preprocess/{dataset_name}.py \\
            --template_type={template_type} \\
            --num_samples={num_samples} \\
            --n_features=2 \\
            --centers=3 \\
            --cluster_std=1.0 \\
            --test_ratio=0.2 \\
            --n_shot={shot}
        """
    elif dataset_name in ["moons", "linear"]:
        return f"""
        python {root_dir}/examples/data_preprocess/{dataset_name}.py \\
            --template_type={template_type} \\
            --num_samples={num_samples} \\
            --n_shot={shot}
        """
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")

def generate_responses(
    dataset_name,
    shot,
    model_name,
    iteration,
    temperature=0.3,
):
    # 注意：使用 model_name.split('/')[-1] 可能在模型名为变量时出错，所以移到Bash中处理
    return f"""
        python -m verl.trainer.main_generation \\
            trainer.nnodes=1 \\
            trainer.n_gpus_per_node=1 \\
            data.path={root_dir}/datasets/{dataset_name}/{shot}_shot/train.parquet \\
            data.prompt_key=prompt \\
            data.n_samples={num_responses} \\
            data.batch_size=128 \\
            data.output_path={root_dir}/results/{dataset_name}/$(basename ${{current_model}})_{shot}_shot_iter${{iteration}}_gen_train.parquet \\
            model.path=${{current_model}} \\
            +model.trust_remote_code=True \\
            rollout.temperature={temperature} \\
            rollout.top_k=10 \\
            rollout.top_p=0.9 \\
            rollout.prompt_length=1000 \\
            rollout.response_length=500 \\
            rollout.tensor_model_parallel_size=1 \\
            rollout.gpu_memory_utilization=0.8 \\
            trainer.wandb=True
    """

def evaluate_responses(
    dataset_name,
    shot,
    model_name,
    iteration,
):
    return f"""
        python -m verl.trainer.main_eval \\
            data.path={root_dir}/results/{dataset_name}/$(basename ${{current_model}})_{shot}_shot_iter${{iteration}}_gen_train.parquet \\
            trainer.wandb=True
    """

def train_on_correct_responses(
    dataset_name,
    shot,
    model_name,
    iteration,
    local_dir,
):
    return f"""
        
        model_name_safe=$(basename ${{current_model}} | tr '/' '_')
        experiment_name="{dataset_name}-${{model_name_safe}}-iter${{iteration}}"

        torchrun --standalone --nnodes=1 --nproc_per_node=1 \\
            -m verl.trainer.fsdp_sft_trainer \\
            data.train_files={root_dir}/results/{dataset_name}/$(basename ${{current_model}})_{shot}_shot_iter${{iteration}}_correct_train.parquet \\
            data.val_files={root_dir}/datasets/{dataset_name}/{shot}_shot/test.parquet \\
            data.prompt_key=prompt \\
            data.response_key=answer \\
            data.micro_batch_size=8 \\
            model.partial_pretrain=${{current_model}} \\
            trainer.default_local_dir={local_dir} \\
            trainer.project_name={dataset_name}-iterative-sft \\
            trainer.experiment_name=${{experiment_name}} \\
            trainer.total_epochs={total_epochs} \\
            trainer.logger=['console','wandb']
    """

def filter_correct_responses(
    dataset_name,
    shot,
    model_name,
    iteration,
):
    return f"""
        python {root_dir}/scripts/filter_correct_responses.py \\
            --input_path={root_dir}/results/{dataset_name}/$(basename ${{current_model}})_{shot}_shot_iter${{iteration}}_gen_train.parquet \\
            --output_path={root_dir}/results/{dataset_name}/$(basename ${{current_model}})_{shot}_shot_iter${{iteration}}_correct_train.parquet
    """

def check_accuracy(
    dataset_name,
    shot,
    model_name,
    iteration,
):
    return f"""
        python {root_dir}/scripts/check_perfect_accuracy.py \\
            --eval_path={root_dir}/results/{dataset_name}/$(basename ${{current_model}})_{shot}_shot_iter${{iteration}}_gen_train.parquet ;
    """

os.makedirs(f"{root_dir}/run_exps/auto_iterative_sft", exist_ok=True)

script_paths = []
for dataset in dataset_list:
    for model in model_list:
        template_type = supported_llms[model]["template_type"]
        base_local_dir = f"{root_dir}/experiments/{dataset}/{model.replace('/', '_')}/"
        
        # Create base local directory for experiments
        os.makedirs(base_local_dir, exist_ok=True)
        
        # Generate initial dataset
        commands = []
        commands.append(gen_dataset(dataset, shot=shot, template_type=template_type))
        
        # Following your pseudocode structure:
        # i = 0
        # model = base_model (without any training)
        # while(i < maximum_iteration)
        
        commands.append(f"""
# Initialize variables
current_model="{model}"  # Start with base model
iteration=0

while [ $iteration -lt {max_iterations} ]; do
    echo "Starting iteration $iteration"
    
    # Generate responses using current model
    echo "Generating responses with model: $current_model"
    {generate_responses(dataset, shot=shot, model_name="", iteration="")}
    
    # Evaluate the generated responses
    echo "Evaluating responses"
    {evaluate_responses(dataset, shot=shot, model_name="", iteration="")}
    
    # Check if we've achieved perfect accuracy
    echo "Checking accuracy"
    if {check_accuracy(dataset, shot=shot, model_name="", iteration="")} then
        echo "Achieved perfect accuracy at iteration $iteration"
        break
    else
        # Filter correct responses
        echo "Filtering correct responses"
        {filter_correct_responses(dataset, shot=shot, model_name="", iteration="")}
        
        # Create local directory for this iteration
        iteration_dir="{base_local_dir}iter${{iteration}}"
        mkdir -p "$iteration_dir"
        
        # Train on correct responses to get a new model
        echo "Training new model"
        {train_on_correct_responses(
            dataset, 
            shot=shot, 
            model_name="", 
            iteration="", 
            local_dir="$iteration_dir"
        )}
        
        # Update current model to the newly trained model
        current_model="$iteration_dir"
        
        # Increment iteration counter
        iteration=$((iteration+1))
    fi
done
        """)
        
        bash_script = '''
#!/bin/bash
set -e  # Exit immediately if a command exits with non-zero status
''' + "\n".join(commands)
        script_path = f"{root_dir}/run_exps/auto_iterative_sft/{dataset}_{model.replace('/', '_')}_iterative_sft.sh"
        script_paths.append(script_path)
        with open(script_path, "w") as f:
            f.write(bash_script)

run_all_scripts = "\n".join([f"bash {script_path}" for script_path in script_paths])

with open(f"{root_dir}/run_exps/auto_iterative_sft/run_all.sh", "w") as f:
    f.write(run_all_scripts)