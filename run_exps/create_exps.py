import pandas as pd
import os
from constants import supported_llms, get_model_name, get_model_dir, supported_datasets
from environment import root_dir
import argparse
from run_exps.helper import gen_dataset, inference, rl_train
import pdb

model_size_upper_limit = 100_000_000_000

supported_model_list = [model for model in supported_llms.keys() if supported_llms[model]["model_size"] <= model_size_upper_limit]

shot_list = [10, 50, 100, 200]

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, nargs="+", default=["blobs", "moons", "linear", "circles"], choices=supported_datasets.keys())
parser.add_argument("--model", type=str, nargs="+", default=supported_model_list, choices=supported_model_list)
parser.add_argument("--mode", type=str, nargs="+", default=["reasoning", "no_reasoning"], choices=["reasoning", "no_reasoning", "customized", "ricl_1", "ricl_2", "ricl_3"])
parser.add_argument("--shot", type=int, nargs="+", default=shot_list)
parser.add_argument("--train", action="store_true")
parser.add_argument("--n_gpus", type=int, default=2)
parser.add_argument("--response_length_thinking_factor", type=float, default=2.0)
parser.add_argument("--load_train_step", type=int, default=0)
parser.add_argument("--n_samples", type=int, nargs="+", default=[10000])
parser.add_argument("--n_query", type=int, default=10)
parser.add_argument("--feature_noise", type=float, nargs="+", default=[None])
parser.add_argument("--label_noise", type=float, default=0.0)
parser.add_argument("--data_mode", type=str, default="default", choices=["default", "grid", "mixed"])
parser.add_argument("--wandb", type=int, default=2, choices=[0, 1, 2])
parser.add_argument("--api_workers", type=int, default=16)
parser.add_argument("--exp_name", type=str, default="")
parser.add_argument("--inductive", type=bool, default=False)
parser.add_argument("--test_ratio", type=float, default=1)
parser.add_argument("--temperature", type=float, default=0.0)
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
inductive = args.inductive
n_samples_list = args.n_samples
feature_noise_list = args.feature_noise
if args.dataset == ["regression"]:
    dataset_list = ['linreg', 'pwreg', 'cosreg', 'l1normreg', 'quadreg', 'expreg']
if args.dataset == ["classification"]:
    dataset_list = ['blobs', 'moons', 'linear', 'circles']
if args.dataset == ["all"]:
    dataset_list = ['blobs', 'moons', 'linear', 'circles', 'linreg', 'pwreg', 'cosreg', 'l1normreg', 'quadreg', 'expreg']
os.makedirs(f"{root_dir}/run_exps/auto", exist_ok=True)
 
script_paths = []
for dataset in dataset_list:
    for shot in shot_list:
        prompt_length = int((24 * shot + 260 + 24 * args.n_query) * 1.1)
        for model in model_list:
            for mode in mode_list:
                for n_samples in n_samples_list:
                    for feature_noise in feature_noise_list:
                        if mode == "reasoning":
                            template_type = supported_llms[model]["template_type"]
                            response_length = int(prompt_length * args.response_length_thinking_factor)
                        elif mode == "no_reasoning":
                            template_type = supported_llms[model]["template_type"] + "_no_reasoning"
                            response_length = 100
                        elif mode == "customized":
                            template_type = supported_llms[model]["template_type"] + "_customized"
                            response_length = 100
                        elif "ricl" in mode:
                            template_type = supported_llms[model]["template_type"] + "_" + mode
                            response_length = int(prompt_length * args.response_length_thinking_factor)
                        else:
                            raise ValueError(f"Mode {mode} not supported, should be in [reasoning, no_reasoning, customized, ricl]")
                        
                        if inductive:
                            template_type = template_type + "_inductive"
                        if feature_noise is None:
                            feature_noise = supported_datasets[dataset]["feature_noise"]
                        
                        
                        command_list = []
                        gen_command = gen_dataset(
                            dataset_name=dataset,
                            shot=shot,
                            template_type=template_type,
                            num_samples=n_samples,
                            n_query=args.n_query,
                            feature_noise=feature_noise,
                            label_noise=args.label_noise,
                            data_mode=args.data_mode,
                            test_ratio=args.test_ratio
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
                                feature_noise=feature_noise,
                                label_noise=args.label_noise,
                                n_gpus=args.n_gpus,
                                data_mode=args.data_mode,
                                n_query=args.n_query,
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
                                    feature_noise=feature_noise,
                                    label_noise=args.label_noise,
                                    data_mode=args.data_mode,
                                    train_step=args.load_train_step,
                                    n_query=args.n_query,
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
                                n_query=args.n_query, 
                                feature_noise=feature_noise,
                                label_noise=args.label_noise,
                                n_gpus=args.n_gpus,
                                data_mode=args.data_mode,
                                wandb=args.wandb,
                                train_step=args.load_train_step,
                                api_workers=args.api_workers,
                                temperature=args.temperature
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
                            feature_noise=feature_noise, 
                            label_noise=args.label_noise,
                            data_mode=args.data_mode,
                            n_query=args.n_query
                        )
                        script_path = f"{root_dir}/run_exps/auto/{model_name}_train_{args.train}.sh"
                        script_paths.append(script_path)
                        os.makedirs(os.path.dirname(script_path), exist_ok=True)
                        with open(script_path, "w") as f:
                            f.write(bash_script)

run_all_scripts = "\n".join([f"bash {script_path}" for script_path in script_paths])

exp_name = args.exp_name
if exp_name: exp_name = f"_{exp_name}"
with open(f"{root_dir}/run_exps/auto/run_all{exp_name}.sh", "w") as f:
    f.write(run_all_scripts)