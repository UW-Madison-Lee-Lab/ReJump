import pandas as pd
import os
from constants import supported_llms, get_result_dir, get_dataset_dir, get_model_name, get_model_dir
from environment import root_dir
import argparse

model_size_upper_limit = 10_000_000_000

supported_model_list = [model for model in supported_llms.keys() if supported_llms[model]["model_size"] <= model_size_upper_limit]

shot_list = [10, 50, 100, 200]

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, nargs="+", default=["blobs", "moons", "linear"], choices=["blobs", "moons", "linear"])
parser.add_argument("--model", type=str, nargs="+", default=supported_model_list, choices=supported_model_list)
parser.add_argument("--mode", type=str, nargs="+", default=["reasoning", "no_reasoning"], choices=["reasoning", "no_reasoning"])
parser.add_argument("--train", action="store_true")
parser.add_argument("--n_gpus", type=int, default=2)
parser.add_argument("--response_length_thinking_factor", type=float, default=2.0)
args = parser.parse_args()

dataset_list = args.dataset
model_list = args.model

def gen_dataset(
    dataset_name, 
    shot,
    template_type="qwen-instruct"
):
    if dataset_name == "blobs":
        return f"""
python {root_dir}/examples/data_preprocess/{dataset_name}.py \
    --template_type={template_type} \
    --num_samples=1000 \
    --n_features=2 \
    --centers=3 \
    --cluster_std=1.0 \
    --test_ratio=0.2 \
    --n_shot={shot}
    """
    elif dataset_name in ["moons", "linear"]:
        return f"""
python {root_dir}/examples/data_preprocess/{dataset_name}.py \
    --template_type={template_type} \
    --num_samples=1000 \
    --n_shot={shot}
    """
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")

def train(
    dataset_name,
    shot,
    model_name,
    template_type="qwen-instruct",
    prompt_length=256,
    response_length=1024
):
    return f"""
export VLLM_ATTENTION_BACKEND=XFORMERS

python -m verl.trainer.main_ppo \
    data.train_files={get_dataset_dir(dataset_name, shot, template_type)}/train.parquet \
    data.val_files={get_dataset_dir(dataset_name, shot, template_type)}/test.parquet \
    data.train_batch_size=256 \
    data.val_batch_size=1312 \
    data.max_prompt_length={prompt_length} \
    data.max_response_length={response_length} \
    actor_rollout_ref.model.path={model_name} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size=8 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size={args.n_gpus} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=4 \
    critic.optim.lr=1e-5 \
    critic.model.path={model_name} \
    critic.ppo_micro_batch_size=8 \
    critic.model.enable_gradient_checkpointing=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.logger=['wandb'] \
    +trainer.val_before_train=False \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node={args.n_gpus} \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=100 \
    trainer.project_name=TinyZero \
    trainer.experiment_name={get_model_name(dataset_name, model_name, shot, template_type, response_length)} \
    trainer.total_epochs=15 2>&1 | tee verl_demo.log
    """

def inference(
    dataset_name,
    shot,
    model_name,
    temperature=0.3,
    template_type="qwen-instruct",
    prompt_length=256,
    response_length=1024
):
    return f"""
python -m verl.trainer.main_generation \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node={args.n_gpus} \
    data.path={get_dataset_dir(dataset_name, shot, template_type)}/test.parquet \
    data.prompt_key=prompt \
    data.n_samples=1 \
    data.batch_size=128 \
    data.output_path={get_result_dir(dataset_name, model_name, shot, template_type, response_length)}/test.parquet \
    model.path={model_name} \
    +model.trust_remote_code=True \
    rollout.temperature={temperature} \
    rollout.top_k=10 \
    rollout.top_p=0.9 \
    rollout.prompt_length={prompt_length} \
    rollout.response_length={response_length} \
    rollout.tensor_model_parallel_size={args.n_gpus} \
    rollout.gpu_memory_utilization=0.8 \
    trainer.wandb=True
    """
    
def eval(
    dataset_name,
    shot,
    model_name,
    template_type="qwen-instruct",
    response_length=1024
):
    return f"""
python -m verl.trainer.main_eval \
    data.path={get_result_dir(dataset_name, model_name, shot, template_type, response_length)}/test.parquet \
    trainer.wandb=True
    """

os.makedirs(f"{root_dir}/run_exps/auto", exist_ok=True)
 
script_paths = []
for dataset in dataset_list:
    for shot in shot_list:
        prompt_length = ((24 * shot + 185) // 1000 + 1) * 1000
        for model in model_list:
            for mode in args.mode:
                if mode == "reasoning":
                    template_type = supported_llms[model]["template_type"]
                    response_length = int(prompt_length * args.response_length_thinking_factor)
                elif mode == "no_reasoning":
                    template_type = "no_reasoning"
                    response_length = 100
                else:
                    raise ValueError(f"Mode {mode} not supported, should be in [reasoning, no_reasoning]")
                
                command_list = []
                
                gen_command = gen_dataset(
                    dataset_name=dataset,
                    shot=shot,
                    template_type=template_type,
                )
                command_list.append(gen_command)
                if args.train:
                    train_command = train(
                        dataset_name=dataset,
                        shot=shot,
                        model_name=model,
                        template_type=template_type,
                        prompt_length=prompt_length,
                        response_length=response_length
                    )
                    # model_path= f"{get_model_dir(dataset, model, shot, template_type, response_length)}/actor/global_step_{global_step}"
                    command_list.append(train_command)
                else:
                    model_path = model
                
                    inference_command = inference(
                        dataset_name=dataset,
                        shot=shot,
                        model_name=model_path,
                        template_type=template_type,
                        prompt_length=prompt_length,
                        response_length=response_length
                    )
                    command_list.append(inference_command)
                    eval_command = eval(
                        dataset_name=dataset,
                        shot=shot,
                        model_name=model_path,
                        template_type=template_type,
                    )
                    command_list.append(eval_command)
                    
                bash_script = "\n".join(command_list)
                script_path = f"{root_dir}/run_exps/auto/{dataset}_{shot}_{model.replace('/', '_')}_{mode}_train_{args.train}.sh"
                script_paths.append(script_path)
                with open(script_path, "w") as f:
                    f.write(bash_script)

run_all_scripts = "\n".join([f"bash {script_path}" for script_path in script_paths])

with open(f"{root_dir}/run_exps/auto/run_all.sh", "w") as f:
    f.write(run_all_scripts)
