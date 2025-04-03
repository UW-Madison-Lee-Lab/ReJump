python run_exps/new_iterative_sft.py --dataset blobs --model Qwen/Qwen2.5-3B-Instruct --max_iterations 1 --num_responses 1 --num_samples 10000 --shot 50 --total_epochs 1 --nproc_per_node 1 --n_gpus_per_node 1 --noise_level 1.0 --project_prefix blobs_get_Initial_results --test_before_training

python run_exps/new_iterative_sft.py --dataset circles --model Qwen/Qwen2.5-3B-Instruct --max_iterations 1 --num_responses 1 --num_samples 10000 --shot 50 --total_epochs 1 --nproc_per_node 1 --n_gpus_per_node 1 --noise_level 0 --project_prefix circles_get_Initial_results --test_before_training

python run_exps/new_iterative_sft.py --dataset moons --model Qwen/Qwen2.5-3B-Instruct --max_iterations 1 --num_responses 1 --num_samples 10000 --shot 50 --total_epochs 1 --nproc_per_node 1 --n_gpus_per_node 1 --noise_level 0.1 --project_prefix moons_get_Initial_results --test_before_training

python run_exps/new_iterative_sft.py --dataset linear --model Qwen/Qwen2.5-3B-Instruct --max_iterations 1 --num_responses 1 --num_samples 10000 --shot 50 --total_epochs 1 --nproc_per_node 1 --n_gpus_per_node 1 --noise_level 0.1 --project_prefix linear_get_Initial_results --test_before_training
