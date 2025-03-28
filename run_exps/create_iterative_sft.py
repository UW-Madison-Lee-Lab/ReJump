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
parser.add_argument("--num_responses", type=int, default=1)
parser.add_argument("--label_flip_rate", type=float, default=0.0)
parser.add_argument("--noise_level", type=float, default=1.0)
parser.add_argument("--train_set_temperature", type=float, default=0, help="Temperature for generating training responses")
parser.add_argument("--test_set_temperature", type=float, default=0, help="Temperature for generating test responses")
parser.add_argument("--train_set_top_k", type=int, default=-1, help="Top-k sampling parameter for training data")
parser.add_argument("--test_set_top_k", type=int, default=-1, help="Top-k sampling parameter for test data")
parser.add_argument("--train_set_top_p", type=float, default=1.0, help="Top-p sampling parameter for training data")
parser.add_argument("--test_set_top_p", type=float, default=1.0, help="Top-p sampling parameter for test data")
parser.add_argument("--shot", type=int, default=50, help="Number of examples in few-shot learning")
parser.add_argument("--num_samples", type=int, default=10000, help="Number of samples in the dataset")
parser.add_argument("--total_epochs", type=int, default=1, help="Number of training epochs")
parser.add_argument("--nproc_per_node", type=int, default=1, help="Number of processes per node for distributed training")
parser.add_argument("--n_gpus_per_node", type=int, default=1, help="Number of GPUs per node")
parser.add_argument("--push_to_hub", action="store_true", help="Whether to push the model to the hub")
parser.add_argument("--project_prefix", type=str, default="", help="Prefix of the project name")
args = parser.parse_args()

dataset_list = args.dataset
model_list = args.model
max_iterations = args.max_iterations
num_responses = args.num_responses
shot = args.shot
num_samples = args.num_samples
label_flip_rate = args.label_flip_rate
noise_level = args.noise_level
total_epochs = args.total_epochs
nproc_per_node = args.nproc_per_node
n_gpus_per_node = args.n_gpus_per_node
train_set_temperature = args.train_set_temperature
test_set_temperature = args.test_set_temperature
train_set_top_k = args.train_set_top_k
test_set_top_k = args.test_set_top_k
train_set_top_p = args.train_set_top_p
test_set_top_p = args.test_set_top_p


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
            --label_flip_rate={label_flip_rate} \\
            --noise_level={noise_level} \\
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

def generate_train_responses(
    dataset_name,
    shot,
    model_name,
    iteration,
    temperature,
    top_k,
    top_p
):
    # 注意：使用 model_name.split('/')[-1] 可能在模型名为变量时出错，所以移到Bash中处理
    return f"""
        python -m verl.trainer.main_generation \\
            trainer.nnodes=1 \\
            trainer.n_gpus_per_node={n_gpus_per_node} \\
            data.path={root_dir}/datasets/{dataset_name}/{shot}_shot/train.parquet \\
            data.prompt_key=prompt \\
            data.n_samples={num_responses} \\
            data.batch_size=128 \\
            data.output_path={root_dir}/results/{dataset_name}/$(basename ${{current_model}})_{args.project_prefix}_iter${{iteration}}_gen_train.parquet \\
            model.path=${{current_model}} \\
            +model.trust_remote_code=True \\
            rollout.temperature={temperature} \\
            rollout.top_k={top_k} \\
            rollout.top_p={top_p} \\
            rollout.prompt_length=2048 \\
            rollout.response_length=1024 \\
            rollout.tensor_model_parallel_size=1 \\
            rollout.gpu_memory_utilization=0.8 \\
            trainer.wandb=True \\
            trainer.project_name={args.project_prefix}-train-generation_{dataset_name}_{model.replace('/', '_')}-iterative-sft
    """
def generate_test_responses(
    dataset_name,
    shot,
    model_name,
    iteration,
    temperature,
    top_k,
    top_p
):
    return f"""
        python -m verl.trainer.main_generation \\
            trainer.nnodes=1 \\
            trainer.n_gpus_per_node={n_gpus_per_node} \\
            data.path={root_dir}/datasets/{dataset_name}/{shot}_shot/test.parquet \\
            data.prompt_key=prompt \\
            data.n_samples={num_responses} \\
            data.batch_size=128 \\
            data.output_path={root_dir}/results/{dataset_name}/$(basename ${{current_model}})_{args.project_prefix}_iter${{iteration}}_gen_test.parquet \\
            model.path=${{current_model}} \\
            +model.trust_remote_code=True \\
            rollout.temperature={temperature} \\
            rollout.top_k={top_k} \\
            rollout.top_p={top_p} \\
            rollout.prompt_length=2048 \\
            rollout.response_length=1024 \\
            rollout.tensor_model_parallel_size=1 \\
            rollout.gpu_memory_utilization=0.8 \\
            trainer.wandb=True \\
            trainer.project_name={args.project_prefix}-test-generation_{dataset_name}_{model.replace('/', '_')}-iterative-sft
    """
def evaluate_test_responses(
    dataset_name,
    shot,
    model_name,
    iteration,
):
    return f"""
        python -m verl.trainer.main_eval \\
            data.path={root_dir}/results/{dataset_name}/$(basename ${{current_model}})_{args.project_prefix}_iter${{iteration}}_gen_test.parquet \\
            trainer.wandb=True \\
            trainer.project_name={args.project_prefix}-test-evaluation_{dataset_name}_{model.replace('/', '_')}-iterative-sft
    """

def evaluate_train_responses(
    dataset_name,
    shot,
    model_name,
    iteration,
):
    return f"""
        python -m verl.trainer.main_eval \\
            data.path={root_dir}/results/{dataset_name}/$(basename ${{current_model}})_{args.project_prefix}_iter${{iteration}}_gen_train.parquet \\
            trainer.wandb=True \\
            trainer.project_name={args.project_prefix}-train-evaluation_{dataset_name}_{model.replace('/', '_')}-iterative-sft
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

        torchrun --standalone --nnodes=1 --nproc_per_node={nproc_per_node} \\
            -m verl.trainer.fsdp_sft_trainer \\
            data.train_files={root_dir}/results/{dataset_name}/$(basename ${{current_model}})_{args.project_prefix}_iter${{iteration}}_correct_train.parquet \\
            data.val_files={root_dir}/results/{dataset_name}/$(basename ${{current_model}})_{args.project_prefix}_iter${{iteration}}_correct_train.parquet \\
            data.prompt_key=prompt \\
            data.response_key=answer \\
            data.micro_batch_size=1 \\
            model.partial_pretrain=${{current_model}} \\
            trainer.default_local_dir={local_dir} \\
            trainer.project_name={args.project_prefix}-finetune_{dataset_name}_{model.replace('/', '_')}-iterative-sft \\
            trainer.experiment_name=${{experiment_name}} \\
            trainer.total_epochs={total_epochs} \\
            trainer.logger=['console','wandb'] \\
            trainer.hub.push_to_hub={"true" if args.push_to_hub else "false"} \\
    """

def filter_correct_responses(
    dataset_name,
    shot,
    model_name,
    iteration,
):
    return f"""
        python {root_dir}/scripts/filter_correct_responses.py \\
            --input_path={root_dir}/results/{dataset_name}/$(basename ${{current_model}})_{args.project_prefix}_iter${{iteration}}_gen_train.parquet \\
            --output_path={root_dir}/results/{dataset_name}/$(basename ${{current_model}})_{args.project_prefix}_iter${{iteration}}_correct_train.parquet \\
            --already_trained_correct_path=${{already_trained_correct_path}} \\
            --wandb_project={args.project_prefix}-filtering_{dataset_name}_{model.replace('/', '_')}-iterative-sft \\
            --wandb_run_name=${{experiment_name}}
    """

def check_accuracy(
    dataset_name,
    shot,
    model_name,
    iteration,
):
    return f"""
        python {root_dir}/scripts/check_perfect_accuracy.py \\
            --eval_path={root_dir}/results/{dataset_name}/$(basename ${{current_model}})_{args.project_prefix}_iter${{iteration}}_gen_train.parquet ;
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
already_trained_correct_path="{root_dir}/results/{dataset}/$(basename ${{current_model}})_{args.project_prefix}_correct_train.parquet"
# Delete the file if it exists
rm -f "$already_trained_correct_path"

iteration=0

while [ $iteration -lt {max_iterations} ]; do
    echo "Starting iteration $iteration"
    
    # Generate responses using current model
    echo "Generating responses with model: $current_model"

    {generate_test_responses(dataset, shot=shot, model_name="", iteration="", temperature=test_set_temperature, top_k=test_set_top_k, top_p=test_set_top_p)}

    {evaluate_test_responses(dataset, shot=shot, model_name="", iteration="")}

    {generate_train_responses(dataset, shot=shot, model_name="", iteration="", temperature=train_set_temperature, top_k=train_set_top_k, top_p=train_set_top_p)}
    
    # Evaluate the generated responses
    echo "Evaluating responses"
    {evaluate_train_responses(dataset, shot=shot, model_name="", iteration="")}
    
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
        script_path = f"{root_dir}/run_exps/auto_iterative_sft/{args.project_prefix}_{dataset}_{model.replace('/', '_')}_shot{shot}_epochs{total_epochs}_maxiter{max_iterations}_samples{num_samples}_responses{num_responses}_iterative_sft.sh"
        script_paths.append(script_path)
        with open(script_path, "w") as f:
            f.write(bash_script)

print("-----------Run following scripts:-----------")
for script_path in script_paths:
    print(f"bash {script_path}")
    
# run_all_scripts = "\n".join([f"bash {script_path}" for script_path in script_paths])

# with open(f"{root_dir}/run_exps/auto_iterative_sft/run_all.sh", "w") as f:
#     f.write(run_all_scripts)