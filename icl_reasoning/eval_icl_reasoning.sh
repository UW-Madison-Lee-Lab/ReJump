python -m verl.trainer.main_generation \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=2 \
    data.path=/staging/szhang967/icl_datasets/blobs_50shot_n1.0_f0.0_test10_icl3_seed42.parquet \
    data.prompt_key=prompt \
    data.n_samples=1 \
    data.batch_size=1 \
    data.output_path=/staging/szhang967/icl_dataset-output/blobs_50shot_n1.0_f0.0_test10_icl3_seed42.parquet \
    model.path=Qwen/Qwen2.5-1.5B-Instruct \
    +model.trust_remote_code=True \
    rollout.temperature=0 \
    rollout.top_k=10 \
    rollout.top_p=0.9 \
    rollout.prompt_length=16000 \
    rollout.response_length=8092 \
    rollout.tensor_model_parallel_size=1 \
    rollout.gpu_memory_utilization=0.8 \
    trainer.wandb=True \
    project_name=icl_reasoning_test
    
