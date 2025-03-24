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
parser.add_argument("--shot", type=int, nargs="+", default=shot_list)
parser.add_argument("--train", action="store_true")
parser.add_argument("--n_gpus", type=int, default=2)
parser.add_argument("--response_length_thinking_factor", type=float, default=2.0)
parser.add_argument("--load_train_step", type=int, default=None)
parser.add_argument("--n_samples", type=int, nargs="+", default=[10000])
parser.add_argument("--noise_level", type=float, nargs="+", default=[None])
args = parser.parse_args()

if args.load_train_step is not None:
    if len(args.model) > 1:
        raise ValueError("Only one model is supported when loading train step")
    if len(args.dataset) > 1:
        raise ValueError("Only one dataset is supported when loading train step")
    if len(args.mode) > 1:
        raise ValueError("Only one mode is supported when loading train step")
    if len(args.shot) > 1:
        raise ValueError("Only one shot is supported when loading train step")
    

dataset_list = args.dataset
model_list = args.model
mode_list = args.mode
shot_list = args.shot
n_samples_list = args.n_samples
noise_level_list = args.noise_level

def gen_dataset(
    dataset_name, 
    shot,
    template_type="qwen-instruct",
    num_samples=10000,
    noise_level=None
):
    if dataset_name == "blobs":
        noise_level = 1.0 if noise_level is None else noise_level
        return f"""
python {root_dir}/examples/data_preprocess/{dataset_name}.py \
    --template_type={template_type} \
    --num_samples={num_samples} \
    --n_features=2 \
    --centers=3 \
    --noise_level={noise_level} \
    --test_ratio=0.2 \
    --n_shot={shot}
    """
    elif dataset_name in ["moons", "linear"]:
        noise_level = 0.1 if noise_level is None else noise_level
        return f"""
python {root_dir}/examples/data_preprocess/{dataset_name}.py \
    --template_type={template_type} \
    --num_samples={num_samples} \
    --n_shot={shot} \
    --noise_level={noise_level}
    --test_ratio=0.2
    """
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")

def train(
    dataset_name,
    shot,
    model_name,
    template_type="qwen-instruct",
    prompt_length=256,
    response_length=1024,
    num_samples=10000,
    noise_level=None,
):
    return f"""
export VLLM_ATTENTION_BACKEND=XFORMERS

python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files={get_dataset_dir(dataset_name, shot, template_type, num_samples, noise_level)}/train.parquet \
    data.val_files={get_dataset_dir(dataset_name, shot, template_type, num_samples, noise_level)}/test.parquet \
    data.train_batch_size=128 \
    data.val_batch_size=640 \
    data.max_prompt_length={prompt_length} \
    data.max_response_length={response_length} \
    actor_rollout_ref.model.path={model_name} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.grad_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size={args.n_gpus} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['wandb'] \
    +trainer.val_before_train=False \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node={args.n_gpus} \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.project_name=TinyZero \
    trainer.experiment_name={get_model_name(dataset, model, shot, template_type, response_length, n_samples, noise_level)} \
    trainer.total_epochs=15
    """

def inference(
    dataset_name,
    shot,
    model_name,
    temperature=0,
    template_type="qwen-instruct",
    prompt_length=256,
    response_length=1024,
    num_samples=10000,
    noise_level=None
):
    return f"""
python -m verl.trainer.main_generation \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node={args.n_gpus} \
    data.path={get_dataset_dir(dataset_name, shot, template_type, num_samples, noise_level)}/test.parquet \
    data.prompt_key=prompt \
    data.n_samples=1 \
    data.batch_size=128 \
    data.output_path={get_result_dir(dataset_name, model_name, shot, template_type, response_length, num_samples, noise_level)}/test.parquet \
    model.path={model_name} \
    +model.trust_remote_code=True \
    rollout.temperature={temperature} \
    rollout.top_k=-1 \
    rollout.top_p=1 \
    rollout.prompt_length={prompt_length} \
    rollout.response_length={response_length} \
    rollout.tensor_model_parallel_size={args.n_gpus} \
    rollout.gpu_memory_utilization=0.8 \
    trainer.wandb=True \
    rollout.n=1
    """
    
# def eval(
#     dataset_name,
#     shot,
#     model_name,
#     template_type="qwen-instruct",
#     response_length=1024
# ):
#     return f"""
# python -m verl.trainer.main_eval \
#     data.path={get_result_dir(dataset_name, model_name, shot, template_type, response_length)}/test.parquet \
#     trainer.wandb=True
#     """

os.makedirs(f"{root_dir}/run_exps/auto", exist_ok=True)
 
script_paths = []
for dataset in dataset_list:
    for shot in shot_list:
        prompt_length = int((24 * shot + 185) * 1.1)
        for model in model_list:
            for mode in mode_list:
                for n_samples in n_samples_list:
                    for noise_level in noise_level_list:
                        if mode == "reasoning":
                            template_type = supported_llms[model]["template_type"]
                            response_length = int(prompt_length * args.response_length_thinking_factor)
                        elif mode == "no_reasoning":
                            template_type = "no_reasoning"
                            response_length = 100
                        else:
                            raise ValueError(f"Mode {mode} not supported, should be in [reasoning, no_reasoning]")
                        
                        if noise_level is None:
                            if dataset == "blobs":
                                noise_level = 1.0
                            elif dataset in ["moons", "linear"]:
                                noise_level = 0.1
                            else:
                                noise_level = 0
                        
                        
                        command_list = []
                        
                        gen_command = gen_dataset(
                            dataset_name=dataset,
                            shot=shot,
                            template_type=template_type,
                            num_samples=n_samples,
                            noise_level=noise_level
                        )
                        command_list.append(gen_command)
                        if args.train:
                            train_command = train(
                                dataset_name=dataset,
                                shot=shot,
                                model_name=model,
                                template_type=template_type,
                                prompt_length=prompt_length,
                                response_length=response_length,
                                num_samples=n_samples,
                                noise_level=noise_level,
                            )
                            command_list.append(train_command)
                        else:
                            if args.load_train_step is not None:
                                model_path = get_model_dir(dataset, model, shot, template_type, response_length, n_samples, noise_level, args.load_train_step)
                            else:
                                model_path = model
                        
                            inference_command = inference(
                                dataset_name=dataset,
                                shot=shot,
                                model_name=model_path,
                                template_type=template_type,
                                prompt_length=prompt_length,
                                response_length=response_length,
                                num_samples=n_samples,
                                noise_level=noise_level
                            )
                            command_list.append(inference_command)
                            # eval_command = eval(
                            #     dataset_name=dataset,
                            #     shot=shot,
                            #     model_name=model_path,
                            #     template_type=template_type,
                            #     response_length=response_length
                            # )
                            # command_list.append(eval_command)
                            
                        bash_script = "\n".join(command_list)
                        script_path = f"{root_dir}/run_exps/auto/{get_model_name(dataset, model, shot, template_type, response_length, n_samples, noise_level)}_train_{args.train}.sh"
                        script_paths.append(script_path)
                        with open(script_path, "w") as f:
                            f.write(bash_script)

run_all_scripts = "\n".join([f"bash {script_path}" for script_path in script_paths])

with open(f"{root_dir}/run_exps/auto/run_all.sh", "w") as f:
    f.write(run_all_scripts)
