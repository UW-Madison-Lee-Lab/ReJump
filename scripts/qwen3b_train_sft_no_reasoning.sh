N_GPUS=1
BASE_MODEL=Qwen/Qwen2.5-3B
DATA_DIR=datasets/blobs/50_shot/no_reasoning
ROLLOUT_TP_SIZE=1
PROJECT_NAME=LIFTR
EXPERIMENT_NAME=blobs-qwen2.5-3b-sft-no-reasoning
VLLM_ATTENTION_BACKEND=XFORMERS

mkdir -p checkpoints/$PROJECT_NAME/$EXPERIMENT_NAME

python3 -m verl.trainer.fsdp_sft_trainer \
data.train_files=$DATA_DIR/train.parquet \
data.val_files=$DATA_DIR/test.parquet \
data.train_batch_size=64 \
data.micro_batch_size=8 \
data.max_length=1024 \
data.prompt_key=prompt \
+data.prompt_dict_keys=["content"] \
data.response_key=label \
model.partial_pretrain=$BASE_MODEL \
trainer.logger=['wandb'] \
+trainer.val_before_train=False \
trainer.default_hdfs_dir=null \
trainer.project_name=$PROJECT_NAME \
trainer.experiment_name=$EXPERIMENT_NAME \
trainer.total_epochs=15 2>&1 | tee verl_demo.log \
trainer.default_local_dir=checkpoints/$PROJECT_NAME/$EXPERIMENT_NAME
