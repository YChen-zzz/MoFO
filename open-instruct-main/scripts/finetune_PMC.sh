export CUDA_VISIBLE_DEVICES=0,1

MODEL_SIZE=1B
## The batch size per GPU has a deterministic impact on the forgetting of the fine-tuned model.  
## It is recommended to keep this configuration unchanged.
MODEL_SIZE=1B
NUM_GPUS=2
BATCH_SIZE_PER_GPU=2    ## for lora, 4
TOTAL_BATCH_SIZE=128
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
echo "Training llama model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"

cd open-instruct-main

## full fine-tuning
accelerate launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    --use_deepspeed \
    --deepspeed_config_file ./ds_configs/stage3_no_offloading_accelerate.conf \
    ./open-instruct/finetune.py \
    --model_name_or_path path_to_meta-llama--Llama-2-7b-chat-hf \
    --tokenizer_name path_to_meta-llama--Llama-2-7b-chat-hf tokenizer \
    --use_slow_tokenizer \
    --train_file path_to_pmc_train_file \
    --val_file ./data/metamathqa_val_dataset.jsonl \
    --max_seq_length 1024 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate 1e-5 \
    --lr_scheduler_type cosine \
    --gradient_checkpointing \
    --warmup_ratio 0.03 \
    --weight_decay 0. \
    --num_train_epochs 2 \
    --data_ratio 1.0 \
    --output_dir path_to_save_fine-tuned_ckpt \
    --with_tracking \
    --report_to tensorboard \
    --logging_steps 10 \


## Enable MoFO by adding the following options:
#    --use_AdamW_MoFO \
#    --MoFO_fraction 0.15 \            
## you can also try MoFO_fraction 0.10

