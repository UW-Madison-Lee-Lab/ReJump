N_GPUS=1
BASE_MODEL=Qwen/Qwen2.5-3B
DATA_DIR=datasets/blobs/50_shot/no_reasoning
ROLLOUT_TP_SIZE=1
EXPERIMENT_NAME=blobs-qwen2.5-3b-sft-no-reasoning
VLLM_ATTENTION_BACKEND=XFORMERS

python3 -m verl.trainer.fsdp_sft_trainer \
data.train_files=$DATA_DIR/train.parquet \
data.val_files=$DATA_DIR/test.parquet \
data.train_batch_size=32 \

trainer.logger=['wandb'] \
+trainer.val_before_train=False \
trainer.default_hdfs_dir=null \
trainer.n_gpus_per_node=$N_GPUS \
trainer.nnodes=1 \
trainer.save_freq=10 \
trainer.test_freq=10 \
trainer.project_name=LIFTR \
trainer.experiment_name=$EXPERIMENT_NAME \
trainer.total_epochs=15 2>&1 | tee verl_demo.log
trainer.default_hdfs_dir=~/experiments/liftr/$EXPERIMENT_NAME
