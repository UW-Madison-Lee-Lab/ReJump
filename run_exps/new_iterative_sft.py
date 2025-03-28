import pandas as pd
import os
import subprocess
import argparse
import sys
from constants import supported_llms, get_dataset_dir
from environment import root_dir
from datetime import datetime
def parse_args():
    """Parse command line arguments"""
    model_size_upper_limit = 10_000_000_000
    supported_model_list = [model for model in supported_llms.keys() 
                           if supported_llms[model]["model_size"] <= model_size_upper_limit]
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, nargs="+", default=["blobs"], 
                       choices=["blobs", "moons", "linear", "circles"])
    parser.add_argument("--model", type=str, nargs="+", default=supported_model_list, 
                       choices=supported_model_list)
    parser.add_argument("--max_iterations", type=int, default=10)
    parser.add_argument("--num_responses", type=int, default=1)
    parser.add_argument("--label_flip_rate", type=float, default=0.0)
    parser.add_argument("--noise_level", type=float, default=1.0)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--train_set_temperature", type=float, default=0)
    parser.add_argument("--test_set_temperature", type=float, default=0)
    parser.add_argument("--train_set_top_k", type=int, default=-1)
    parser.add_argument("--test_set_top_k", type=int, default=-1)
    parser.add_argument("--train_set_top_p", type=float, default=1.0)
    parser.add_argument("--test_set_top_p", type=float, default=1.0)
    parser.add_argument("--shot", type=int, default=50)
    parser.add_argument("--num_samples", type=int, default=10000)
    parser.add_argument("--total_epochs", type=int, default=1)
    parser.add_argument("--nproc_per_node", type=int, default=1)
    parser.add_argument("--n_gpus_per_node", type=int, default=1)
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--project_prefix", type=str, default="")
    
    return parser.parse_args()

def run_command(command, step_name):
    """Execute shell command and terminate program if command fails"""
    print(f"Executing: {step_name}")
    print(f"Command: {command}")
    
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    
    print(stdout.decode())
    
    if process.returncode != 0:
        print(f"ERROR: {step_name} failed with return code {process.returncode}")
        print(f"Error message: {stderr.decode()}")
        sys.exit(1)  # Terminate program with error code
    
    print(f"Successfully completed: {step_name}")
    return True

def gen_dataset(dataset_name, shot, template_type, args):
    """Generate dataset based on dataset type"""
    if dataset_name in ["blobs", "moons", "linear", "circles"]:
        cmd = f"python {root_dir}/examples/data_preprocess/{dataset_name}.py " \
              f"--template_type={template_type} " \
              f"--num_samples={args.num_samples} " \
              f"--label_flip_rate={args.label_flip_rate} " \
              f"--noise_level={args.noise_level} " \
              f"--test_ratio={args.test_ratio} " \
              f"--n_shot={shot}"
    else:
        print(f"ERROR: Dataset {dataset_name} not supported")
        sys.exit(1)
    
    return run_command(cmd, f"Generate {dataset_name} dataset")

def generate_responses(dataset_name, dataset_local_dir, model_path, iteration, is_train, args, model_basename):
    """Generate model responses for either training or test data"""
    data_type = "train" if is_train else "test"
    temperature = args.train_set_temperature if is_train else args.test_set_temperature
    top_k = args.train_set_top_k if is_train else args.test_set_top_k
    top_p = args.train_set_top_p if is_train else args.test_set_top_p
    
    output_path = f"{root_dir}/results/{dataset_name}/{model_basename}_{args.project_prefix}_iter{iteration}_gen_{data_type}.parquet"
    
    cmd = f"python -m verl.trainer.main_generation " \
          f"trainer.nnodes=1 " \
          f"trainer.n_gpus_per_node={args.n_gpus_per_node} " \
          f"data.path={dataset_local_dir}/{data_type}.parquet " \
          f"data.prompt_key=prompt " \
          f"data.n_samples={args.num_responses} " \
          f"data.batch_size=128 " \
          f"data.output_path={output_path} " \
          f"model.path={model_path} " \
          f"+model.trust_remote_code=True " \
          f"rollout.temperature={temperature} " \
          f"rollout.top_k={top_k} " \
          f"rollout.top_p={top_p} " \
          f"rollout.prompt_length=2048 " \
          f"rollout.response_length=1024 " \
          f"rollout.tensor_model_parallel_size=1 " \
          f"rollout.gpu_memory_utilization=0.8 " \
          f"trainer.wandb=True " \
          f"trainer.project_name={args.project_prefix}-{data_type}-generation_{dataset_name}_{model_path.replace('/', '_')}-iterative-sft"
    
    run_command(cmd, f"Generate {data_type} responses for iteration {iteration}")
    return output_path

def evaluate_responses(input_path, model_path, data_type, dataset_name, args):
    """Evaluate model responses"""
    cmd = f"python -m verl.trainer.main_eval " \
          f"data.path={input_path} " \
          f"trainer.wandb=True " \
          f"trainer.project_name={args.project_prefix}-{data_type}-evaluation_{dataset_name}_{model_path.replace('/', '_')}-iterative-sft"
    
    return run_command(cmd, f"Evaluate {data_type} responses")

def filter_correct_responses(dataset_name, model_path, iteration, args, model_basename, already_trained_correct_path, input_path):
    """Filter correct responses from generated data"""
    
    output_path = f"{root_dir}/results/{dataset_name}/{model_basename}_{args.project_prefix}_iter{iteration}_correct_train.parquet"
    
    experiment_name = f"{dataset_name}-{model_basename.replace('/', '_')}-iter{iteration}"
    
    cmd = f"python {root_dir}/scripts/filter_correct_responses.py " \
          f"--input_path={input_path} " \
          f"--output_path={output_path} " \
          f"--already_trained_correct_path={already_trained_correct_path} "
    
    cmd += f"--wandb_project={args.project_prefix}-filtering_{dataset_name}_{model_path.replace('/', '_')}-iterative-sft " \
           f"--wandb_run_name={experiment_name}"
    
    run_command(cmd, f"Filter correct responses for iteration {iteration}")
    return output_path

def train_on_correct_responses(dataset_name, model_path, iteration, args, model_basename, correct_responses_path):
    """Train model on filtered correct responses"""
    base_local_dir = f"{root_dir}/experiments/{dataset_name}/{model_path.replace('/', '_')}/"
    iteration_dir = f"{base_local_dir}iter{iteration}"
    os.makedirs(iteration_dir, exist_ok=True)
    
    experiment_name = f"{dataset_name}-{model_basename.replace('/', '_')}-iter{iteration}"
    
    cmd = f"torchrun --standalone --nnodes=1 --nproc_per_node={args.nproc_per_node} " \
          f"-m verl.trainer.fsdp_sft_trainer " \
          f"data.train_files={correct_responses_path} " \
          f"data.val_files={correct_responses_path} " \
          f"data.prompt_key=prompt " \
          f"data.response_key=answer " \
          f"data.micro_batch_size=1 " \
          f"model.partial_pretrain={model_path} " \
          f"trainer.default_local_dir={iteration_dir} " \
          f"trainer.project_name={args.project_prefix}-finetune_{dataset_name}_{model_path.replace('/', '_')}-iterative-sft " \
          f"trainer.experiment_name={experiment_name} " \
          f"trainer.total_epochs={args.total_epochs} " \
          f"trainer.logger=['console','wandb'] " \
          f"trainer.hub.push_to_hub={'true' if args.push_to_hub else 'false'}"
    
    run_command(cmd, f"Train model on correct responses for iteration {iteration}")
    return iteration_dir

def main():
    """Main function that orchestrates the iterative SFT process"""
    args = parse_args()
    
    for dataset in args.dataset:
        for model in args.model:
            print(f"\n=== Starting experiment with dataset: {dataset}, model: {model} ===\n")
            
            # Get model template type
            template_type = supported_llms[model]["template_type"]
            model_basename = os.path.basename(model)
            
            # Ensure directories exist
            os.makedirs(f"{root_dir}/results/{dataset}", exist_ok=True)
            base_local_dir = f"{root_dir}/experiments/{dataset}/{model.replace('/', '_')}/"
            os.makedirs(base_local_dir, exist_ok=True)
            
            # Generate dataset
            gen_dataset(dataset, args.shot, template_type, args)
            dataset_local_dir = get_dataset_dir(dataset, args.shot, template_type, args.num_samples, args.noise_level, args.label_flip_rate)
            
            # Start iterative training
            current_model = model

            already_trained_correct_path = f"{root_dir}/results/{dataset}/{args.project_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_already_trained_correct.parquet"
            #delete the file if it exists
            if os.path.exists(already_trained_correct_path):
                os.remove(already_trained_correct_path)
            
            for iteration in range(args.max_iterations):
                print(f"\n--- Starting iteration {iteration} ---\n")
                print(f"Current model: {current_model}")
                
                # Generate test responses
                test_output_path = generate_responses(dataset, dataset_local_dir, current_model, iteration, False, args, model_basename)
                
                # Evaluate test responses
                evaluate_responses(test_output_path, current_model, "test", dataset, args)
                
                # Generate training responses
                train_output_path = generate_responses(dataset, dataset_local_dir, current_model, iteration, True, args, model_basename)
                
                # Evaluate training responses
                evaluate_responses(train_output_path, current_model, "train", dataset, args)
                    
                correct_responses_path = filter_correct_responses(
                    dataset, current_model, iteration, args, model_basename, already_trained_correct_path, train_output_path
                )
                
                # Train new model
                new_model_path = train_on_correct_responses(
                    dataset, current_model, iteration, args, model_basename, correct_responses_path
                )
                
                # Update current model
                current_model = new_model_path
                print(f"Updated model to: {current_model}")
            
            print(f"\n=== Completed experiment with dataset: {dataset}, model: {model} ===\n")

if __name__ == "__main__":
    main()