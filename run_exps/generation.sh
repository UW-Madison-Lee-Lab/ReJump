#! /bin/bash

for model in "Qwen/Qwen2.5-3B-Instruct" "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"; do
python -m verl.trainer.main_generation \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=2 \
    data.path=datasets/blobs/test.parquet \
    data.prompt_key=prompt \
    data.n_samples=1 \
    data.batch_size=128 \
    data.output_path=results/blobs/qwen2.5-3b-instruct_gen_test.parquet \
    model.path=Qwen/Qwen2.5-3B-Instruct \
    +model.trust_remote_code=True \
    rollout.temperature=1.0 \
    rollout.top_k=10 \
    rollout.top_p=0.9 \
    rollout.prompt_length=2048 \
    rollout.response_length=1024 \
    rollout.tensor_model_parallel_size=2 \
    rollout.gpu_memory_utilization=0.8