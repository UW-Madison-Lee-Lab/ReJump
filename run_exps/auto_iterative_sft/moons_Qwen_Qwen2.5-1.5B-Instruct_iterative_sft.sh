
#!/bin/bash
set -e  # Exit immediately if a command exits with non-zero status

        python /home/szhang967/liftr/examples/data_preprocess/moons.py \
            --template_type=qwen-instruct \
            --num_samples=260 \
            --n_shot=1
        

# Initialize variables
current_model="Qwen/Qwen2.5-1.5B-Instruct"  # Start with base model
already_trained_correct_path="/home/szhang967/liftr/results/moons/$(basename ${current_model})_1_shot_correct_train.parquet"
# Delete the file if it exists
rm -f "$already_trained_correct_path"

iteration=0

while [ $iteration -lt 3 ]; do
    echo "Starting iteration $iteration"
    
    # Generate responses using current model
    echo "Generating responses with model: $current_model"
    
        python -m verl.trainer.main_generation \
            trainer.nnodes=1 \
            trainer.n_gpus_per_node=1 \
            data.path=/home/szhang967/liftr/datasets/moons/1_shot/train.parquet \
            data.prompt_key=prompt \
            data.n_samples=5 \
            data.batch_size=128 \
            data.output_path=/home/szhang967/liftr/results/moons/$(basename ${current_model})_1_shot_iter${iteration}_gen_train.parquet \
            model.path=${current_model} \
            +model.trust_remote_code=True \
            rollout.temperature=0.3 \
            rollout.top_k=10 \
            rollout.top_p=0.9 \
            rollout.prompt_length=1000 \
            rollout.response_length=500 \
            rollout.tensor_model_parallel_size=1 \
            rollout.gpu_memory_utilization=0.8 \
            trainer.wandb=True
    
    
    # Evaluate the generated responses
    echo "Evaluating responses"
    
        python -m verl.trainer.main_eval \
            data.path=/home/szhang967/liftr/results/moons/$(basename ${current_model})_1_shot_iter${iteration}_gen_train.parquet \
            trainer.wandb=True
    
    
    # Check if we've achieved perfect accuracy
    echo "Checking accuracy"
    if 
        python /home/szhang967/liftr/scripts/check_perfect_accuracy.py \
            --eval_path=/home/szhang967/liftr/results/moons/$(basename ${current_model})_1_shot_iter${iteration}_gen_train.parquet ;
     then
        echo "Achieved perfect accuracy at iteration $iteration"
        break
    else
        # Filter correct responses
        echo "Filtering correct responses"
        
        python /home/szhang967/liftr/scripts/filter_correct_responses.py \
            --input_path=/home/szhang967/liftr/results/moons/$(basename ${current_model})_1_shot_iter${iteration}_gen_train.parquet \
            --output_path=/home/szhang967/liftr/results/moons/$(basename ${current_model})_1_shot_iter${iteration}_correct_train.parquet \
            --already_trained_correct_path=${already_trained_correct_path}
    
        
        # Create local directory for this iteration
        iteration_dir="/home/szhang967/liftr/experiments/moons/Qwen_Qwen2.5-1.5B-Instruct/iter${iteration}"
        mkdir -p "$iteration_dir"
        
        # Train on correct responses to get a new model
        echo "Training new model"
        
        
        model_name_safe=$(basename ${current_model} | tr '/' '_')
        experiment_name="moons-${model_name_safe}-iter${iteration}"

        torchrun --standalone --nnodes=1 --nproc_per_node=1 \
            -m verl.trainer.fsdp_sft_trainer \
            data.train_files=/home/szhang967/liftr/results/moons/$(basename ${current_model})_1_shot_iter${iteration}_correct_train.parquet \
            data.val_files=/home/szhang967/liftr/results/moons/$(basename ${current_model})_1_shot_iter${iteration}_correct_train.parquet \
            data.prompt_key=prompt \
            data.response_key=answer \
            data.micro_batch_size=8 \
            model.partial_pretrain=${current_model} \
            trainer.default_local_dir=$iteration_dir \
            trainer.project_name=moons-iterative-sft \
            trainer.experiment_name=${experiment_name} \
            trainer.total_epochs=1 \
            trainer.logger=['console','wandb']
    
        
        # Update current model to the newly trained model
        current_model="$iteration_dir"
        
        # Increment iteration counter
        iteration=$((iteration+1))
    fi
done
        