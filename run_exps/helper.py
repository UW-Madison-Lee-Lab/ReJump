from environment import root_dir
from constants import get_dataset_dir, get_model_name, get_result_dir, get_dataset_filename, supported_datasets
import pdb
import re

def gen_dataset(
    dataset_name, 
    shot,
    template_type="qwen-instruct",
    num_samples=10000,
    n_query=10,
    feature_noise=None,
    label_noise=0.0,
    data_mode="default",
    test_ratio=1,
    response_length=3046,
):
    if dataset_name == "blobs":
        feature_noise = 1.0 if feature_noise is None else feature_noise
    elif dataset_name in ["moons", "linear"]:
        feature_noise = 0.1 if feature_noise is None else feature_noise
    elif dataset_name == "circles":
        feature_noise = 0.01 if feature_noise is None else feature_noise
    else:
        feature_noise = None
        
        command = f"""
python -m examples.data_preprocess.{dataset_name} \
    --template_type={template_type} \
    --num_samples={num_samples} \
    --n_shot={shot} \
    --n_query={n_query} \
    --feature_noise={feature_noise} \
    --test_ratio={test_ratio} \
    --label_noise={label_noise} \
    --data_mode={data_mode}
            """ 
    
    if "ricl" in template_type:
        if supported_datasets[dataset_name]["type"] == "regression":
            example_datasets = ["l1normreg", "cosreg", "quadreg", "expreg"]
            example_datasets.remove(dataset_name)
        elif supported_datasets[dataset_name]["type"] == "classification":
            example_datasets = ["circles", "moons", "linear", "blobs"]
            example_datasets.remove(dataset_name)
        else:
            example_datasets = [dataset_name]
            
        ricl_shot = int(re.match(r".*?ricl_(\d+)", template_type).group(1))
        dataset_path = get_dataset_dir(
            dataset_name=dataset_name,
            shot=shot,
            template_type=template_type,
            num_samples=num_samples,
            feature_noise=feature_noise,
            label_noise=label_noise,
            data_mode=data_mode,
            n_query=n_query
        )
        dataset_path = f"{dataset_path}/test_{data_mode}.parquet"
        
        result_paths = []
        for example_dataset in example_datasets:
            result_path = get_result_dir(
                dataset_name=example_dataset,
                model_name="deepseek-ai/deepseek-reasoner",
                shot=shot,
                template_type="reasoning_api",
                response_length=response_length,
                num_samples=num_samples,
                feature_noise=feature_noise,
                label_noise=label_noise,
                train_step=0,
                data_mode=data_mode,
                n_query=n_query,
                temperature=0.00,
            )
            result_paths.append(f"{result_path}/test_{data_mode}.parquet")
        
        command += f"""
python -m icl_reasoning.icl_reasoning_v2 \
    --dataset_path {dataset_path} \
    --result_path {' '.join(result_paths)} \
    --num_shot {ricl_shot} \
    --output_path {dataset_path}
        """
    return command
        

def mix_dataset(
    dataset_path,
    dataset_ratio,
    num_samples=10000,
    data_mode="mixed"
):
    return f"""
python {root_dir}/examples/data_preprocess/multitask.py \
    --dataset_path {' '.join(dataset_path)} \
    --dataset_ratio {' '.join([str(ratio) for ratio in dataset_ratio])} \
    --num_samples {num_samples} \
    --data_mode {data_mode}
    """

def rl_train(
    dataset_name,
    shot,
    model_name,
    template_type="qwen-instruct",
    prompt_length=256,
    response_length=1024,
    num_samples=10000,
    feature_noise=None,
    label_noise=0.0,
    n_gpus=2,
    data_mode="default",
    n_query=1
):
    dataset_dir = get_dataset_dir(
        dataset_name=dataset_name,
        shot=shot,
        template_type=template_type,
        num_samples=num_samples,
        feature_noise=feature_noise,
        label_noise=label_noise,
        data_mode=data_mode,
        n_query=n_query
    )
    trained_model_name = get_model_name(
        dataset_name=dataset_name,
        model_name=model_name,
        shot=shot,
        template_type=template_type,
        response_length=response_length,
        num_samples=num_samples,
        feature_noise=feature_noise,
        label_noise=label_noise,
        data_mode=data_mode,
        n_query=n_query
    )
    result_dir = get_result_dir(
        dataset_name=dataset_name,
        model_name=model_name,
        shot=shot,
        template_type=template_type,
        response_length=response_length,
        num_samples=num_samples,
        feature_noise=feature_noise,
        label_noise=label_noise,
        data_mode=data_mode,
        train_step=0,
        n_query=n_query,
        temperature=0.00,
    )
    output_file = get_dataset_filename(split="test", data_mode=data_mode)
    return f"""
export VLLM_ATTENTION_BACKEND=XFORMERS

python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files={dataset_dir}/{get_dataset_filename(split="train", data_mode=data_mode)} \
    data.val_files={dataset_dir}/{get_dataset_filename(split="test", data_mode=data_mode)} \
    data.train_batch_size=128 \
    data.val_batch_size=640 \
    data.output_path={result_dir}/{output_file} \
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
    actor_rollout_ref.rollout.tensor_model_parallel_size={n_gpus} \
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
    trainer.n_gpus_per_node={n_gpus} \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.project_name=TinyZero \
    trainer.experiment_name={trained_model_name} \
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
    n_query=1,
    feature_noise=None,
    label_noise=0.0,
    n_gpus=2,
    data_mode="default",
    train_step=0,
    wandb=2,
    api_workers=16,
):
    dataset_dir = get_dataset_dir(
        dataset_name=dataset_name,
        shot=shot,
        template_type=template_type,
        num_samples=num_samples,
        feature_noise=feature_noise,
        label_noise=label_noise,
        data_mode=data_mode,
        n_query=n_query
    )
    result_dir = get_result_dir(
        dataset_name=dataset_name,
        model_name=model_name,
        shot=shot,
        template_type=template_type,
        response_length=response_length,
        num_samples=num_samples,
        feature_noise=feature_noise,
        label_noise=label_noise,
        data_mode=data_mode,
        train_step=train_step,
        n_query=n_query,
        temperature=temperature,
    )
    output_file = get_dataset_filename(split="test", data_mode=data_mode)
    return f"""
python -m verl.trainer.main_generation \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node={n_gpus} \
    data.path={dataset_dir}/{output_file} \
    data.prompt_key=prompt \
    data.n_samples=1 \
    data.batch_size=128 \
    data.output_path={result_dir}/{output_file} \
    model.path={model_name} \
    +model.trust_remote_code=True \
    rollout.temperature={temperature} \
    rollout.top_k=-1 \
    rollout.top_p=1 \
    rollout.prompt_length={prompt_length} \
    rollout.response_length={response_length} \
    rollout.tensor_model_parallel_size={n_gpus} \
    rollout.gpu_memory_utilization=0.8 \
    trainer.wandb={wandb} \
    rollout.n=1 \
    rollout.api_workers={api_workers}
    """