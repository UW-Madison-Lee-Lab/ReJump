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
):
    if "ricl" in template_type:
        if supported_datasets[dataset_name]["type"] == "regression":
            example_datasets = ["l1normreg", "cosreg", "quadreg", "expreg"]
        else:
            example_datasets = ["circles", "moons", "linear", "blobs"]
            
        if dataset_name in example_datasets:
            example_datasets.remove(dataset_name)
        ricl_shot = int(re.match(r".*?ricl_(\d+)", template_type).group(1))
        example_datasets = example_datasets[:ricl_shot]
        
        icl_examples = []
        for i, example_dataset in enumerate(example_datasets):
            icl_example_prompt = f"""
    "+icl_examples.{i}.dataset_name={example_dataset}" \
    "+icl_examples.{i}.label_noise={label_noise}" \
    "+icl_examples.{i}.feature_noise={supported_datasets[example_dataset]['feature_noise']}" \
    "+icl_examples.{i}.shot=50" \
    "+icl_examples.{i}.n_query={n_query}" \
    "+icl_examples.{i}.response_length=3046" \
    "+icl_examples.{i}.num_samples=500" \
    "+icl_examples.{i}.num_examples=1" \
    "+icl_examples.{i}.train_step=0" \
    "+icl_examples.{i}.data_mode={data_mode}"
        """
            icl_examples.append(icl_example_prompt)
            
        max_length = 40000 if supported_datasets[dataset_name]["type"] == "regression" else 80000
        icl_examples_prompt = ''.join(icl_examples).replace('\n', '')
        command = f"""
python -m icl_reasoning.icl_reasoning \
    "+icl_examples=[]" \
    {icl_examples_prompt} \
    "+mode=reasoning" \
    "+template_type={template_type}" \
    "+tokenizer_name=Qwen/Qwen2.5-3B-Instruct" \
    "+icl_example_seed=42" \
    "+test_data_seed=42" \
    "+train_step=0" \
    "+data_mode=default" \
    "+icl_example_maxlength={max_length}" \
    "+test_data.dataset_name={dataset_name}" \
    "+test_data.label_noise={label_noise}" \
    "+test_data.feature_noise={feature_noise}" \
    "+test_data.num_samples={num_samples}" \
    "+test_data.test_ratio=0.2" \
    "+test_data_examples.dataset_name={dataset_name}" \
    "+test_data_examples.label_noise={label_noise}" \
    "+test_data_examples.feature_noise={feature_noise}" \
    "+test_data_examples.shot={shot}" \
    "+test_data_examples.n_query={n_query}"
        """
    else:
        if dataset_name == "blobs":
            feature_noise = 1.0 if feature_noise is None else feature_noise
        elif dataset_name in ["moons", "linear"]:
            feature_noise = 0.1 if feature_noise is None else feature_noise
        elif dataset_name == "circles":
            feature_noise = 0.01 if feature_noise is None else feature_noise
        else:
            feature_noise = 0
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
        n_query=n_query
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
        n_query=n_query
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