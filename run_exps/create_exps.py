import pandas as pd
import os
from constants import supported_llms, get_model_name, get_model_dir
from environment import root_dir
import argparse
from run_exps.helper import gen_dataset, inference, rl_train

model_size_upper_limit = 10_000_000_000

supported_model_list = [model for model in supported_llms.keys() if supported_llms[model]["model_size"] <= model_size_upper_limit]

shot_list = [10, 50, 100, 200]

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, nargs="+", default=["blobs", "moons", "linear", "circles"], choices=["blobs", "moons", "linear", "circles"])
parser.add_argument("--model", type=str, nargs="+", default=supported_model_list, choices=supported_model_list)
parser.add_argument("--mode", type=str, nargs="+", default=["reasoning", "no_reasoning"], choices=["reasoning", "no_reasoning"])
parser.add_argument("--shot", type=int, nargs="+", default=shot_list)
parser.add_argument("--train", action="store_true")
parser.add_argument("--n_gpus", type=int, default=2)
parser.add_argument("--response_length_thinking_factor", type=float, default=2.0)
parser.add_argument("--load_train_step", type=int, default=0)
parser.add_argument("--n_samples", type=int, nargs="+", default=[10000])
parser.add_argument("--noise_level", type=float, nargs="+", default=[None])
parser.add_argument("--label_flip_rate", type=float, default=0.0)
parser.add_argument("--data_mode", type=str, default="default", choices=["default", "grid", "mixed"])
parser.add_argument("--wandb", type=int, default=2, choices=[0, 1, 2])
args = parser.parse_args()

if args.load_train_step:
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
                            template_type = supported_llms[model]["template_type"] + "_no_reasoning"
                            response_length = 100
                        else:
                            raise ValueError(f"Mode {mode} not supported, should be in [reasoning, no_reasoning]")
                        
                        if noise_level is None:
                            if dataset == "blobs":
                                noise_level = 1.0
                            elif dataset in ["moons", "linear"]:
                                noise_level = 0.1
                            else:
                                noise_level = 0.0
                        
                        
                        command_list = []
                        
                        gen_command = gen_dataset(
                            dataset_name=dataset,
                            shot=shot,
                            template_type=template_type,
                            num_samples=n_samples,
                            noise_level=noise_level,
                            label_flip_rate=args.label_flip_rate,
                            data_mode=args.data_mode
                        )
                        command_list.append(gen_command)
                        if args.train:
                            train_command = rl_train(
                                dataset_name=dataset,
                                shot=shot,
                                model_name=model,
                                template_type=template_type,
                                prompt_length=prompt_length,
                                response_length=response_length,
                                num_samples=n_samples,
                                noise_level=noise_level,
                                label_flip_rate=args.label_flip_rate,
                                n_gpus=args.n_gpus,
                                data_mode=args.data_mode
                            )
                            command_list.append(train_command)
                        else:
                            if args.load_train_step:
                                model_path = get_model_dir(
                                    dataset_name=dataset,
                                    model_name=model,
                                    shot=shot,
                                    template_type=template_type,
                                    response_length=response_length,
                                    num_samples=n_samples,
                                    noise_level=noise_level,
                                    label_flip_rate=args.label_flip_rate,
                                    data_mode=args.data_mode,
                                    train_step=args.load_train_step
                                )
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
                                noise_level=noise_level,
                                label_flip_rate=args.label_flip_rate,
                                n_gpus=args.n_gpus,
                                data_mode=args.data_mode,
                                wandb=args.wandb,
                                train_step=args.load_train_step
                            )
                            command_list.append(inference_command)

                        bash_script = "\n".join(command_list)
                        model_name = get_model_name(
                            dataset_name=dataset,
                            model_name=model,
                            shot=shot,
                            template_type=template_type,
                            response_length=response_length,
                            num_samples=n_samples,
                            noise_level=noise_level, 
                            label_flip_rate=args.label_flip_rate,
                            data_mode=args.data_mode
                        )
                        script_path = f"{root_dir}/run_exps/auto/{model_name}_train_{args.train}.sh"
                        script_paths.append(script_path)
                        os.makedirs(os.path.dirname(script_path), exist_ok=True)
                        with open(script_path, "w") as f:
                            f.write(bash_script)

run_all_scripts = "\n".join([f"bash {script_path}" for script_path in script_paths])

with open(f"{root_dir}/run_exps/auto/run_all.sh", "w") as f:
    f.write(run_all_scripts)
