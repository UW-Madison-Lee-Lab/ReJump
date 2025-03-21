N_GPUS=2
BASE_MODEL=Qwen/Qwen2.5-3B-Instruct
DATA_DIR=datasets/blobs/50_shot/no_reasoning
PROJECT_NAME=LIFTR
EXPERIMENT_NAME=blobs-50shot-qwen2.5-3b-instruct-sft-no-reasoning
VLLM_ATTENTION_BACKEND=XFORMERS

mkdir -p checkpoints/$PROJECT_NAME/$EXPERIMENT_NAME

CUDA_VISIBLE_DEVICE=0,1 torchrun --nproc_per_node=2 verl/trainer/fsdp_sft_trainer.py \
data.train_files=$DATA_DIR/train.parquet \
data.val_files=$DATA_DIR/test.parquet \
data.train_batch_size=64 \
data.micro_batch_size=8 \
data.max_length=2048 \
data.prompt_key=prompt \
+data.prompt_dict_keys=["content"] \
data.response_key=label \
model.partial_pretrain=$BASE_MODEL \
trainer.logger=['wandb'] \
+trainer.val_before_train=False \
trainer.default_hdfs_dir=null \
trainer.project_name=$PROJECT_NAME \
trainer.experiment_name=$EXPERIMENT_NAME \
trainer.default_local_dir=checkpoints/$PROJECT_NAME/$EXPERIMENT_NAME \
trainer.total_epochs=15 2>&1 | tee verl_demo.log

