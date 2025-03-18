


generate the dataset
```bash
python examples/data_preprocess/blobs.py\
 --template_type=qwen-instruct \
 --num_samples=1000 \
 --n_features=2 \
 --centers=3 \
 --cluster_std=1.0 \
 --test_ratio=0.2 \
 --n_shot=500
```

# Inference
Note:
- update the batch_size doesn't seem to affect the memory usage
- for synthetic dataset, the prompt_length roughly 24*n_shot + 185, the output length can be observed from the inference part — the average length will be uploaded to wandb

```bash
python -m verl.trainer.main_generation \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=2 \
    trainer.wandb=True \
    data.path=datasets/blobs/test.parquet \
    data.prompt_key=prompt \
    data.n_samples=1 \
    data.batch_size=128 \
    data.output_path=results/blobs/qwen2.5-3b-instruct_gen_test.parquet \
    model.path=Qwen/Qwen2.5-3B-Instruct \
    +model.trust_remote_code=True \
    rollout.temperature=.3 \
    rollout.top_k=10 \
    rollout.top_p=0.9 \
    rollout.prompt_length=2048 \
    rollout.response_length=1024 \
    rollout.tensor_model_parallel_size=2 \
    rollout.gpu_memory_utilization=0.8
```    

```bash
python -m verl.trainer.main_generation \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=2 \
    data.path=datasets/blobs/test.parquet \
    data.prompt_key=prompt \
    data.n_samples=1 \
    data.batch_size=128 \
    data.output_path=results/blobs/deepseek-qwen-1.5b-instruct_gen_test.parquet \
    model.path=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    +model.trust_remote_code=True \
    rollout.temperature=.3 \
    rollout.top_k=10 \
    rollout.top_p=0.9 \
    rollout.prompt_length=2048 \
    rollout.response_length=1024 \
    rollout.tensor_model_parallel_size=2 \
    rollout.gpu_memory_utilization=0.8 \
    trainer.wandb=True 
```    

# Evaluation
```bash
python -m verl.trainer.main_eval \
    data.path=results/blobs/qwen2.5-3b-instruct_gen_test.parquet \
    trainer.wandb=True
```

```bash
python -m verl.trainer.main_eval \
    data.path=results/blobs/deepseek-qwen-1.5b-instruct_gen_test.parquet \
    trainer.wandb=True
```