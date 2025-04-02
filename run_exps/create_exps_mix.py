import pandas as pd
import os
from constants import supported_llms, get_dataset_dir, get_model_name, get_model_dir, get_mixed_configs
from environment import root_dir
from run_exps.helper import gen_dataset, inference, rl_train, mix_dataset
import argparse

model_size_upper_limit = 10_000_000_000

supported_model_list = [model for model in supported_llms.keys() if supported_llms[model]["model_size"] <= model_size_upper_limit]

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, nargs="+", default=["blobs", "moons", "circles"], choices=["blobs", "moons", "linear"])
parser.add_argument("--model", type=str, nargs="+", default=supported_model_list, choices=supported_model_list)
parser.add_argument("--mode", type=str, nargs="+", default=["reasoning", "no_reasoning"], choices=["reasoning", "no_reasoning"])
parser.add_argument("--shot", type=int, nargs="+", default=[50, 50, 50])
parser.add_argument("--train", action="store_true")
parser.add_argument("--n_gpus", type=int, default=2)
parser.add_argument("--response_length_thinking_factor", type=float, default=2.0)
parser.add_argument("--load_train_step", type=int, default=0)
parser.add_argument("--n_samples", type=int, nargs="+", default=[10000])
parser.add_argument("--noise_level", type=float, nargs="+", default=[None, None, None])
parser.add_argument("--label_flip_rate", type=float, nargs="+", default=[0.0, 0.0, 0.0])
parser.add_argument("--dataset_ratio", type=str, nargs="+", default=[1, 1, 1])
parser.add_argument("--data_mode", type=str, default="mixed", choices=["default", "mixed"])
parser.add_argument("--wandb", type=int, default=2, choices=[1, 2])
args = parser.parse_args()

n_mix = len(args.dataset)

mix_check = True 
mix_check = mix_check and len(args.dataset_ratio) == n_mix
mix_check = mix_check and len(args.shot) == n_mix
mix_check = mix_check and len(args.noise_level) == n_mix
mix_check = mix_check and len(args.label_flip_rate) == n_mix
if not mix_check:
    raise ValueError("Make sure the number of datasets, shots, noise levels, and dataset ratios are the same")

# Normalize dataset_ratio to ensure they sum to 1
dataset_ratio_list = [float(ratio) for ratio in args.dataset_ratio]
total_ratio = sum(dataset_ratio_list)
if total_ratio != 0:
    dataset_ratio_list = [round(ratio / total_ratio, 2) for ratio in dataset_ratio_list]
    # Adjust the last value to ensure sum is exactly 1.0
    dataset_ratio_list[-1] = round(1.0 - sum(dataset_ratio_list[:-1]), 2)
else:
    # If all ratios are 0, set equal distribution
    dataset_ratio_list = [round(1.0 / n_mix, 2) for _ in range(n_mix)]
    # Adjust the last value to ensure sum is exactly 1.0
    dataset_ratio_list[-1] = round(1.0 - sum(dataset_ratio_list[:-1]), 2)

dataset_list = args.dataset
model_list = args.model
mode_list = args.mode
shot_list = args.shot
n_samples_list = args.n_samples
noise_level_list = args.noise_level
label_flip_rate_list = args.label_flip_rate

os.makedirs(f"{root_dir}/run_exps/auto", exist_ok=True)
 
script_paths = []

for model in model_list:
    for mode in mode_list:
        for n_samples in n_samples_list:
        
            command_list = []
            dataset_paths = []
            for dataset, shot, noise_level, label_flip_rate, dataset_ratio in zip(dataset_list, shot_list, noise_level_list, label_flip_rate_list, dataset_ratio_list):    
                prompt_length = int((24 * shot + 185) * 1.1)
            
                if noise_level is None:
                    if dataset == "blobs":
                        noise_level = 1.0
                    elif dataset in ["moons", "linear"]:
                        noise_level = 0.1
                    else:
                        noise_level = 0.0
                if mode == "reasoning":
                    template_type = supported_llms[model]["template_type"]
                    response_length = int(prompt_length * args.response_length_thinking_factor)
                elif mode == "no_reasoning":
                    template_type = "no_reasoning"
                    response_length = 100
                else:
                    raise ValueError(f"Mode {mode} not supported, should be in [reasoning, no_reasoning]")
                
                
                gen_command = gen_dataset(
                    dataset_name=dataset,
                    shot=shot,
                    template_type=template_type,
                    num_samples=n_samples,
                    noise_level=noise_level,
                    label_flip_rate=label_flip_rate,
                    data_mode=args.data_mode
                )   
                dataset_path = get_dataset_dir(
                    dataset_name=dataset,
                    shot=shot,
                    template_type=template_type,
                    num_samples=n_samples,
                    noise_level=noise_level,
                    label_flip_rate=label_flip_rate,
                    data_mode=args.data_mode
                )
                command_list.append(gen_command)
                dataset_paths.append(dataset_path)
                
            
            
            mix_command = mix_dataset(
                dataset_path=dataset_paths,
                dataset_ratio=dataset_ratio_list,
                num_samples=n_samples,
                data_mode=args.data_mode
            )
            command_list.append(mix_command)
            
            mixed_configs = get_mixed_configs(
                dataset_paths=dataset_paths,
                dataset_ratios=dataset_ratio_list,
                num_samples=n_samples
            )
            
            
            if args.train:
                train_command = rl_train(
                    dataset_name=mixed_configs["dataset_name"],
                    shot=mixed_configs["shot"],
                    model_name=model,
                    template_type=mixed_configs["template_type"],
                    prompt_length=prompt_length,
                    response_length=response_length,
                    num_samples=n_samples,
                    noise_level=mixed_configs["noise_level"],
                    label_flip_rate=label_flip_rate,
                    n_gpus=args.n_gpus,
                    data_mode=mixed_configs["data_mode"],
                )
                command_list.append(train_command)
            else:
                if args.load_train_step:
                    model_path = get_model_dir(
                        dataset_name=mixed_configs["dataset_name"],
                        model_name=model,
                        shot=mixed_configs["shot"],
                        template_type=mixed_configs["template_type"],
                        response_length=response_length,
                        num_samples=n_samples,
                        noise_level=mixed_configs["noise_level"],
                        label_flip_rate=mixed_configs["label_flip_rate"],
                        data_mode=args.data_mode,
                        train_step=args.load_train_step
                    )
                else:
                    model_path = model
                inference_command = inference(
                    dataset_name=mixed_configs["dataset_name"],
                    shot=mixed_configs["shot"],
                    model_name=model_path,
                    template_type=mixed_configs["template_type"],
                    prompt_length=prompt_length,
                    response_length=response_length,
                    num_samples=n_samples,
                    noise_level=mixed_configs["noise_level"],
                    label_flip_rate=mixed_configs["label_flip_rate"],
                    n_gpus=args.n_gpus,
                    data_mode=mixed_configs["data_mode"],
                    wandb=args.wandb,
                    train_step=args.load_train_step
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
            model_name = get_model_name(
                dataset_name=mixed_configs["dataset_name"],
                model_name=model,
                shot=mixed_configs["shot"],
                template_type=mixed_configs["template_type"],
                response_length=response_length,
                num_samples=n_samples,
                noise_level=mixed_configs["noise_level"], 
                label_flip_rate=mixed_configs["label_flip_rate"],
                data_mode=mixed_configs["data_mode"]
            )
            script_path = f"{root_dir}/run_exps/auto/{model_name}_train_{args.train}.sh"
            script_paths.append(script_path)
            os.makedirs(os.path.dirname(script_path), exist_ok=True)
            with open(script_path, "w") as f:
                f.write(bash_script)

run_all_scripts = "\n".join([f"bash {script_path}" for script_path in script_paths])

with open(f"{root_dir}/run_exps/auto/run_all.sh", "w") as f:
    f.write(run_all_scripts)
