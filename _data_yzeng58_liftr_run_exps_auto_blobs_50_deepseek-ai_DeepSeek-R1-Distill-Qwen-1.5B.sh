
python /data/yzeng58/liftr/examples/data_preprocess/blobs.py     --template_type=qwen-instruct     --num_samples=1000     --n_features=2     --centers=3     --cluster_std=1.0     --test_ratio=0.2     --n_shot=50
    

python -m verl.trainer.main_generation     trainer.nnodes=1     trainer.n_gpus_per_node=2     data.path=/data/yzeng58/liftr/datasets/blobs/50_shot/test.parquet     data.prompt_key=prompt     data.n_samples=1     data.batch_size=128     data.output_path=/data/yzeng58/liftr/results/blobs/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B_50_shot_gen_test.parquet     model.path=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B     +model.trust_remote_code=True     rollout.temperature=0.3     rollout.top_k=10     rollout.top_p=0.9     rollout.prompt_length=2000     rollout.response_length=1000     rollout.tensor_model_parallel_size=2     rollout.gpu_memory_utilization=0.8     trainer.wandb=True
    

python -m verl.trainer.main_eval     data.path=/data/yzeng58/liftr/results/blobs/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B_50_shot_gen_test.parquet     trainer.wandb=True
    