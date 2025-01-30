export CUDA_VISIBLE_DEVICES=0,1

MODEL_SIZE=1B
NUM_GPUS=2
## Lora 这里用24，全量以及shun 用12
BATCH_SIZE_PER_GPU=16
#BATCH_SIZE_PER_GPU=1
TOTAL_BATCH_SIZE=128
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
echo "Training llama model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"

# You can also set --gradient_checkpointing or use `stage3_offloading_accelerate.conf` to save memory, 
# but it will trade off speed.


## pretrain 实验
#accelerate launch \
#    --mixed_precision no \
#    --num_machines 1 \
#    --num_processes $NUM_GPUS \
#    /home/wangsenmiao/yupeng_gpt/open-instruct-main/open_instruct/finetune_mathqa_Qwen.py \
#    --model_name_or_path /home/wangsenmiao/yupeng_gpt/tinyllama \
#    --tokenizer_name /home/wangsenmiao/yupeng_gpt/tinyllama \
#    --use_slow_tokenizer \
#    --train_file /mntcephfs/lab_data/wangsenmiao/sft_datasets/tulu_v2/metamathqa_train_dataset.jsonl \
#    --max_seq_length 1024 \
#    --preprocessing_num_workers 32 \
#    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
#    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
#    --learning_rate 2e-5 \
#    --lr_scheduler_type cosine \
#    --warmup_ratio 0.1 \
#    --weight_decay 0. \
#    --num_train_epochs 2 \
#    --data_ratio 1.0 \
#    --output_dir /mntcephfs/data/ruoyusun/chenyupeng/tinyllama_3t/sft_mathqa10_warm01_full_lr2e-5 \
#    --with_tracking \
#    --use_flash_attn \
#    --report_to tensorboard \
#    --logging_steps 10 \

#sleep 10
#
#accelerate launch \
#    --mixed_precision no \
#    --num_machines 1 \
#    --num_processes $NUM_GPUS \
#    /home/wangsenmiao/yupeng_gpt/open-instruct-main/open_instruct/finetune_mathqa_Qwen.py \
#    --model_name_or_path /home/wangsenmiao/yupeng_gpt/tinyllama \
#    --tokenizer_name /home/wangsenmiao/yupeng_gpt/tinyllama \
#    --use_slow_tokenizer \
#    --train_file /mntcephfs/lab_data/wangsenmiao/sft_datasets/tulu_v2/metamathqa_train_dataset.jsonl \
#    --max_seq_length 1024 \
#    --preprocessing_num_workers 32 \
#    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
#    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
#    --learning_rate 2e-5 \
#    --lr_scheduler_type cosine \
#    --warmup_ratio 0. \
#    --weight_decay 0. \
#    --num_train_epochs 3 \
#    --data_ratio 2.0 \
#    --l2_reg 1e-2 \
#    --output_dir /mntcephfs/data/ruoyusun/chenyupeng/tinyllama_3t/sft_mathqa20_l2_1e-2_lr2e-5 \
#    --with_tracking \
#    --use_flash_attn \
#    --report_to tensorboard \
#    --logging_steps 10 \
#
#sleep 10
#
#
#accelerate launch \
#    --mixed_precision no \
#    --num_machines 1 \
#    --num_processes $NUM_GPUS \
#    /home/wangsenmiao/yupeng_gpt/open-instruct-main/open_instruct/finetune_mathqa_Qwen.py \
#    --model_name_or_path /home/wangsenmiao/yupeng_gpt/tinyllama \
#    --tokenizer_name /home/wangsenmiao/yupeng_gpt/tinyllama \
#    --use_slow_tokenizer \
#    --train_file /mntcephfs/lab_data/wangsenmiao/sft_datasets/tulu_v2/metamathqa_train_dataset.jsonl \
#    --max_seq_length 1024 \
#    --preprocessing_num_workers 32 \
#    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
#    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
#    --learning_rate 2e-5 \
#    --lr_scheduler_type cosine \
#    --warmup_ratio 0. \
#    --weight_decay 0. \
#    --num_train_epochs 3 \
#    --data_ratio 2.0 \
#    --l1_reg 1e-5 \
#    --output_dir /mntcephfs/data/ruoyusun/chenyupeng/tinyllama_3t/sft_mathqa20_l1_1e-5_lr2e-5 \
#    --with_tracking \
#    --use_flash_attn \
#    --report_to tensorboard \
#    --logging_steps 10 \
#
#sleep 10
#
accelerate launch \
    --mixed_precision no \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    --main_process_port 29300 \
    /home/wangsenmiao/yupeng_gpt/open-instruct-main/open_instruct/finetune_mathqa_Qwen.py \
    --model_name_or_path /home/wangsenmiao/yupeng_gpt/tinyllama \
    --tokenizer_name /home/wangsenmiao/yupeng_gpt/tinyllama \
    --use_slow_tokenizer \
    --train_file /mntcephfs/lab_data/wangsenmiao/sft_datasets/tulu_v2/metamathqa_train_dataset.jsonl \
    --max_seq_length 1024 \
    --preprocessing_num_workers 32 \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate 2e-5 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0. \
    --weight_decay 0. \
    --num_train_epochs 3 \
    --use_AdamW_tian_miao2 \
    --data_ratio 2.0 \
    --shun_fraction 0.10 \
    --output_dir /mntcephfs/data/ruoyusun/chenyupeng/tinyllama_3t/sft_mathqa20_tian_miao2010_lr2e-5 \
    --with_tracking \
    --use_flash_attn \
    --report_to tensorboard \
    --logging_steps 10 \


#accelerate launch \
#    --mixed_precision bf16 \
#    --num_machines 1 \
#    --num_processes $NUM_GPUS \
#    --use_deepspeed \
#    --deepspeed_config_file /home/wangsenmiao/yupeng_gpt/open-instruct-main/ds_configs/stage3_no_offloading_accelerate.conf \
#    /home/wangsenmiao/yupeng_gpt/open-instruct-main/open_instruct/finetune_mathqa_Qwen.py \
#    --model_name_or_path /mntcephfs/data/ruoyusun/liziniu/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/8a0442e81540efaeb1a0fe3e95477b5e0edfd423 \
#    --tokenizer_name /mntcephfs/data/ruoyusun/liziniu/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/8a0442e81540efaeb1a0fe3e95477b5e0edfd423 \
#    --use_slow_tokenizer \
#    --train_file /mntcephfs/lab_data/wangsenmiao/sft_datasets/tulu_v2/metamathqa_train_dataset.jsonl \
#    --max_seq_length 1024 \
#    --preprocessing_num_workers 32 \
#    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
#    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
#    --learning_rate 2e-5 \
#    --lr_scheduler_type cosine \
#    --warmup_ratio 0. \
#    --weight_decay 0. \
#    --num_train_epochs 2 \
#    --data_ratio 1.0 \
#    --output_dir /mntcephfs/data/ruoyusun/chenyupeng/llama2_7b/sft_mathqa10_e2 \
#    --with_tracking \
#    --use_flash_attn \
#    --report_to tensorboard \
#    --logging_steps 10 \
#
#
#accelerate launch \
#    --mixed_precision bf16 \
#    --num_machines 1 \
#    --num_processes $NUM_GPUS \
#    --use_deepspeed \
#    --deepspeed_config_file /home/wangsenmiao/yupeng_gpt/open-instruct-main/ds_configs/stage3_no_offloading_accelerate.conf \
#    /home/wangsenmiao/yupeng_gpt/open-instruct-main/open_instruct/finetune_mathqa_Qwen.py \
#    --model_name_or_path /mntcephfs/data/ruoyusun/liziniu/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/8a0442e81540efaeb1a0fe3e95477b5e0edfd423 \
#    --tokenizer_name /mntcephfs/data/ruoyusun/liziniu/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/8a0442e81540efaeb1a0fe3e95477b5e0edfd423 \
#    --use_slow_tokenizer \
#    --train_file /mntcephfs/lab_data/wangsenmiao/sft_datasets/tulu_v2/metamathqa_train_dataset.jsonl \
#    --max_seq_length 1024 \
#    --preprocessing_num_workers 32 \
#    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
#    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
#    --learning_rate 2e-5 \
#    --lr_scheduler_type cosine \
#    --warmup_ratio 0. \
#    --weight_decay 0. \
#    --num_train_epochs 2 \
#    --use_AdamW_tian \
#    --data_ratio 1.0 \
#    --shun_fraction 0.10 \
#    --output_dir /mntcephfs/data/ruoyusun/chenyupeng/llama2_7b/sft_mathqa10_e2_tian_010 \
#    --with_tracking \
#    --use_flash_attn \
#    --report_to tensorboard \
#    --logging_steps 10 \



#sleep 10
#
#
#accelerate launch \
#    --mixed_precision no \
#    --num_machines 1 \
#    --num_processes $NUM_GPUS \
#    /home/wangsenmiao/yupeng_gpt/open-instruct-main/open_instruct/finetune_mathqa_Qwen.py \
#    --model_name_or_path /home/wangsenmiao/yupeng_gpt/tinyllama \
#    --tokenizer_name /home/wangsenmiao/yupeng_gpt/tinyllama \
#    --use_slow_tokenizer \
#    --train_file /mntcephfs/lab_data/wangsenmiao/sft_datasets/tulu_v2/metamathqa_train_dataset.jsonl \
#    --max_seq_length 1024 \
#    --preprocessing_num_workers 32 \
#    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
#    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
#    --learning_rate 2e-5 \
#    --lr_scheduler_type cosine \
#    --warmup_ratio 0.1 \
#    --weight_decay 0. \
#    --num_train_epochs 2 \
#    --use_AdamW_tian \
#    --data_ratio 1.0 \
#    --shun_fraction 0.10 \
#    --output_dir /mntcephfs/data/ruoyusun/chenyupeng/tinyllama_3t/sft_mathqa10_warm01_tian_010_lr2e-5 \
#    --with_tracking \
#    --use_flash_attn \
#    --report_to tensorboard \
#    --logging_steps 10 \

#sleep 10
#
#accelerate launch \
#    --mixed_precision no \
#    --num_machines 1 \
#    --num_processes $NUM_GPUS \
#    /home/wangsenmiao/yupeng_gpt/open-instruct-main/open_instruct/finetune_mathqa_Qwen.py \
#    --model_name_or_path /home/wangsenmiao/yupeng_gpt/tinyllama \
#    --tokenizer_name /home/wangsenmiao/yupeng_gpt/tinyllama \
#    --use_slow_tokenizer \
#    --train_file /mntcephfs/lab_data/wangsenmiao/sft_datasets/tulu_v2/metamathqa_train_dataset.jsonl \
#    --max_seq_length 1024 \
#    --preprocessing_num_workers 32 \
#    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
#    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
#    --learning_rate 2e-5 \
#    --lr_scheduler_type cosine \
#    --warmup_ratio 0. \
#    --data_ratio 2.0 \
#    --weight_decay 0. \
#    --num_train_epochs 3 \
#    --l2_reg 1e-3 \
#    --output_dir /mntcephfs/data/ruoyusun/chenyupeng/tinyllama_3t/sft_mathqa20_l2_1e-3_lr2e-5 \
#    --with_tracking \
#    --use_flash_attn \
#    --report_to tensorboard \
#    --logging_steps 10 \
#
#sleep 10
#
#accelerate launch \
#    --mixed_precision no \
#    --num_machines 1 \
#    --num_processes $NUM_GPUS \
#    /home/wangsenmiao/yupeng_gpt/open-instruct-main/open_instruct/finetune_mathqa_Qwen.py \
#    --model_name_or_path /home/wangsenmiao/yupeng_gpt/tinyllama \
#    --tokenizer_name /home/wangsenmiao/yupeng_gpt/tinyllama \
#    --use_slow_tokenizer \
#    --train_file /mntcephfs/lab_data/wangsenmiao/sft_datasets/tulu_v2/metamathqa_train_dataset.jsonl \
#    --max_seq_length 1024 \
#    --preprocessing_num_workers 32 \
#    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
#    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
#    --learning_rate 2e-5 \
#    --lr_scheduler_type cosine \
#    --warmup_ratio 0. \
#    --weight_decay 0. \
#    --data_ratio 2.0 \
#    --num_train_epochs 3 \
#    --l1_reg 1e-6 \
#    --output_dir /mntcephfs/data/ruoyusun/chenyupeng/tinyllama_3t/sft_mathqa20_l1_1e-6_lr2e-5 \
#    --with_tracking \
#    --use_flash_attn \
#    --report_to tensorboard \
#    --logging_steps 10 \


#accelerate launch \
#    --mixed_precision no \
#    --num_machines 1 \
#    --num_processes $NUM_GPUS \
#    /home/wangsenmiao/yupeng_gpt/open-instruct-main/open_instruct/finetune_magic_code_Qwen.py \
#    --model_name_or_path /mntcephfs/data/ruoyusun/chenyupeng/qwen1_5/Qwen1_5_1dot8B_chat \
#    --tokenizer_name /mntcephfs/data/ruoyusun/chenyupeng/qwen1_5/Qwen1_5_1dot8B_chat \
#    --use_slow_tokenizer \
#    --train_file /mntcephfs/lab_data/wangsenmiao/sft_datasets/tulu_v2/magiccode_train_dataset.jsonl \
#    --max_seq_length 2048 \
#    --preprocessing_num_workers 16 \
#    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
#    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
#    --learning_rate 2e-5 \
#    --lr_scheduler_type linear \
#    --warmup_ratio 0.1 \
#    --weight_decay 0. \
#    --num_train_epochs 2 \
#    --l2_reg 1e-2 \
#    --checkpointing_steps epoch \
#    --output_dir /mntcephfs/data/ruoyusun/chenyupeng/qwen1_5/sft_magiccode_4w_full_l2_1e-2 \
#    --with_tracking \
#    --report_to tensorboard \
#    --logging_steps 1 \

#
#accelerate launch \
#    --mixed_precision no \
#    --num_machines 1 \
#    --num_processes $NUM_GPUS \
#    --main_process_port 29300 \
#    /home/wangsenmiao/yupeng_gpt/open-instruct-main/open_instruct/finetune_med_Qwen.py \
#    --model_name_or_path /mntcephfs/data/ruoyusun/chenyupeng/qwen1_5/Qwen1_5_1dot8B_chat \
#    --tokenizer_name /mntcephfs/data/ruoyusun/chenyupeng/qwen1_5/Qwen1_5_1dot8B_chat \
#    --use_slow_tokenizer \
#    --train_file /mntcephfs/lab_data/wangsenmiao/sft_datasets/tulu_v2/MedMCQA_train_dataset_noexplain.jsonl \
#    --max_seq_length 512 \
#    --preprocessing_num_workers 16 \
#    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
#    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
#    --learning_rate 2e-5 \
#    --lr_scheduler_type linear \
#    --warmup_ratio 0.03 \
#    --weight_decay 0. \
#    --num_train_epochs 2 \
#    --l2_reg 1e-1 \
#    --data_ratio 10.0 \
#    --output_dir /mntcephfs/data/ruoyusun/chenyupeng/qwen1_5/sft_med_180k_noexplain_l2_1e-1 \
#    --with_tracking \
#    --report_to tensorboard \
#    --logging_steps 1 \
#
#
#accelerate launch \
#    --mixed_precision no \
#    --num_machines 1 \
#    --num_processes $NUM_GPUS \
#    --main_process_port 29300 \
#    /home/wangsenmiao/yupeng_gpt/open-instruct-main/open_instruct/finetune_med_Qwen.py \
#    --model_name_or_path /mntcephfs/data/ruoyusun/chenyupeng/qwen1_5/Qwen1_5_1dot8B_chat \
#    --tokenizer_name /mntcephfs/data/ruoyusun/chenyupeng/qwen1_5/Qwen1_5_1dot8B_chat \
#    --use_slow_tokenizer \
#    --train_file /mntcephfs/lab_data/wangsenmiao/sft_datasets/tulu_v2/MedMCQA_train_dataset_noexplain.jsonl \
#    --max_seq_length 512 \
#    --preprocessing_num_workers 16 \
#    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
#    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
#    --learning_rate 2e-5 \
#    --lr_scheduler_type linear \
#    --warmup_ratio 0.03 \
#    --weight_decay 0. \
#    --num_train_epochs 2 \
#    --l1_reg 1e-5 \
#    --data_ratio 10.0 \
#    --output_dir /mntcephfs/data/ruoyusun/chenyupeng/qwen1_5/sft_med_180k_noexplain_l1_1e-5 \
#    --with_tracking \
#    --report_to tensorboard \
#    --logging_steps 1 \


#accelerate launch \
#    --mixed_precision no \
#    --num_machines 1 \
#    --num_processes $NUM_GPUS \
#    /home/wangsenmiao/yupeng_gpt/open-instruct-main/open_instruct/finetune_med_Qwen.py \
#    --model_name_or_path /mntcephfs/data/ruoyusun/chenyupeng/qwen1_5/Qwen1_5_1dot8B_chat \
#    --tokenizer_name /mntcephfs/data/ruoyusun/chenyupeng/qwen1_5/Qwen1_5_1dot8B_chat \
#    --use_slow_tokenizer \
#    --train_file /mntcephfs/lab_data/wangsenmiao/sft_datasets/tulu_v2/MedMCQA_train_dataset_noexplain.jsonl \
#    --max_seq_length 512 \
#    --preprocessing_num_workers 16 \
#    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
#    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
#    --learning_rate 2e-5 \
#    --lr_scheduler_type linear \
#    --warmup_ratio 0.03 \
#    --weight_decay 0. \
#    --num_train_epochs 2 \
#    --checkpointing_steps 1000 \
#    --use_AdamW_tian \
#    --shun_fraction 0.10 \
#    --data_ratio 10.0 \
#    --output_dir /mntcephfs/data/ruoyusun/chenyupeng/qwen1_5/sft_med_180k_noexplain_tian_010 \
#    --with_tracking \
#    --report_to tensorboard \
#    --logging_steps 1 \
#
#accelerate launch \
#    --mixed_precision no \
#    --num_machines 1 \
#    --num_processes $NUM_GPUS \
#    /home/wangsenmiao/yupeng_gpt/open-instruct-main/open_instruct/finetune_med_Qwen.py \
#    --model_name_or_path /mntcephfs/data/ruoyusun/chenyupeng/qwen1_5/Qwen1_5_1dot8B_chat \
#    --tokenizer_name /mntcephfs/data/ruoyusun/chenyupeng/qwen1_5/Qwen1_5_1dot8B_chat \
#    --use_slow_tokenizer \
#    --train_file /mntcephfs/lab_data/wangsenmiao/sft_datasets/tulu_v2/MedMCQA_train_dataset_noexplain.jsonl \
#    --max_seq_length 512 \
#    --preprocessing_num_workers 16 \
#    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
#    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
#    --learning_rate 2e-5 \
#    --lr_scheduler_type linear \
#    --warmup_ratio 0.03 \
#    --weight_decay 0. \
#    --num_train_epochs 2 \
#    --checkpointing_steps 1000 \
#    --use_AdamW_shun \
#    --shun_fraction 0.10 \
#    --data_ratio 10.0 \
#    --output_dir /mntcephfs/data/ruoyusun/chenyupeng/qwen1_5/sft_med_180k_noexplain_shun_010 \
#    --with_tracking \
#    --report_to tensorboard \
#    --logging_steps 1 \

#
#
#accelerate launch \
#    --mixed_precision no \
#    --num_machines 1 \
#    --num_processes $NUM_GPUS \
#    /home/wangsenmiao/yupeng_gpt/open-instruct-main/open_instruct/finetune_med_Qwen.py \
#    --model_name_or_path /mntcephfs/data/ruoyusun/chenyupeng/qwen1_5/Qwen1_5_1dot8B_chat \
#    --tokenizer_name /mntcephfs/data/ruoyusun/chenyupeng/qwen1_5/Qwen1_5_1dot8B_chat \
#    --use_slow_tokenizer \
#    --train_file /mntcephfs/lab_data/wangsenmiao/sft_datasets/tulu_v2/MedMCQA_train_dataset_noexplain.jsonl \
#    --max_seq_length 512 \
#    --preprocessing_num_workers 16 \
#    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
#    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
#    --learning_rate 2e-5 \
#    --lr_scheduler_type linear \
#    --warmup_ratio 0.03 \
#    --weight_decay 0. \
#    --use_AdamW_tian \
#    --shun_fraction 0.05 \
#    --num_train_epochs 5 \
#    --checkpointing_steps epoch \
#    --data_ratio 3.0 \
#    --output_dir /mntcephfs/data/ruoyusun/chenyupeng/qwen1_5/sft_med_54k_noexplain_tian_005_epoch5 \
#    --with_tracking \
#    --report_to tensorboard \
#    --logging_steps 1 \
#
#
#accelerate launch \
#    --mixed_precision no \
#    --num_machines 1 \
#    --num_processes $NUM_GPUS \
#    /home/wangsenmiao/yupeng_gpt/open-instruct-main/open_instruct/finetune_med_Qwen.py \
#    --model_name_or_path /mntcephfs/data/ruoyusun/chenyupeng/qwen1_5/Qwen1_5_1dot8B_chat \
#    --tokenizer_name /mntcephfs/data/ruoyusun/chenyupeng/qwen1_5/Qwen1_5_1dot8B_chat \
#    --use_slow_tokenizer \
#    --train_file /mntcephfs/lab_data/wangsenmiao/sft_datasets/tulu_v2/MedMCQA_train_dataset_noexplain.jsonl \
#    --max_seq_length 512 \
#    --preprocessing_num_workers 32 \
#    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
#    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
#    --learning_rate 2e-5 \
#    --lr_scheduler_type linear \
#    --warmup_ratio 0.03 \
#    --weight_decay 0. \
#    --num_train_epochs 2 \
#    --data_ratio 3.0 \
#    --output_dir /mntcephfs/data/ruoyusun/chenyupeng/qwen1_5/sft_med_54k_noexplain \
#    --with_tracking \
#    --report_to tensorboard \
#    --logging_steps 1 \
#
#accelerate launch \
#    --mixed_precision no \
#    --num_machines 1 \
#    --num_processes $NUM_GPUS \
#    /home/wangsenmiao/yupeng_gpt/open-instruct-main/open_instruct/finetune_med_Qwen.py \
#    --model_name_or_path /mntcephfs/data/ruoyusun/chenyupeng/qwen1_5/Qwen1_5_1dot8B_chat \
#    --tokenizer_name /mntcephfs/data/ruoyusun/chenyupeng/qwen1_5/Qwen1_5_1dot8B_chat \
#    --use_slow_tokenizer \
#    --train_file /mntcephfs/lab_data/wangsenmiao/sft_datasets/tulu_v2/MedMCQA_train_dataset_noexplain.jsonl \
#    --max_seq_length 512 \
#    --preprocessing_num_workers 32 \
#    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
#    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
#    --learning_rate 2e-5 \
#    --lr_scheduler_type linear \
#    --warmup_ratio 0.03 \
#    --weight_decay 0. \
#    --use_AdamW_shun \
#    --shun_fraction 0.10 \
#    --data_ratio 3.0 \
#    --num_train_epochs 5 \
#    --checkpointing_steps epoch \
#    --output_dir /mntcephfs/data/ruoyusun/chenyupeng/qwen1_5/sft_med_54k_noexplain_shun_010_epoch5 \
#    --with_tracking \
#    --report_to tensorboard \
#    --logging_steps 1 \
#
#
#accelerate launch \
#    --mixed_precision no \
#    --num_machines 1 \
#    --num_processes $NUM_GPUS \
#    /home/wangsenmiao/yupeng_gpt/open-instruct-main/open_instruct/finetune_med_Qwen.py \
#    --model_name_or_path /mntcephfs/data/ruoyusun/chenyupeng/qwen1_5/Qwen1_5_1dot8B_chat \
#    --tokenizer_name /mntcephfs/data/ruoyusun/chenyupeng/qwen1_5/Qwen1_5_1dot8B_chat \
#    --use_slow_tokenizer \
#    --train_file /mntcephfs/lab_data/wangsenmiao/sft_datasets/tulu_v2/MedMCQA_train_dataset_noexplain.jsonl \
#    --max_seq_length 512 \
#    --preprocessing_num_workers 32 \
#    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
#    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
#    --learning_rate 2e-5 \
#    --lr_scheduler_type linear \
#    --warmup_ratio 0.03 \
#    --weight_decay 0. \
#    --use_AdamW_shun \
#    --shun_fraction 0.05 \
#    --data_ratio 3.0 \
#    --num_train_epochs 5 \
#    --checkpointing_steps epoch \
#    --output_dir /mntcephfs/data/ruoyusun/chenyupeng/qwen1_5/sft_med_54k_noexplain_shun_005_epoch5 \
#    --with_tracking \
#    --report_to tensorboard \
#    --logging_steps 1 \

#sleep 10
#
#accelerate launch \
#    --mixed_precision bf16 \
#    --num_machines 1 \
#    --num_processes $NUM_GPUS \
#    --use_deepspeed \
#    --deepspeed_config_file /home/wangsenmiao/yupeng_gpt/open-instruct-main/ds_configs/stage3_no_offloading_accelerate.conf \
#    /home/wangsenmiao/yupeng_gpt/open-instruct-main/open_instruct/finetune_med.py \
#    --model_name_or_path /mntcephfs/data/ruoyusun/liziniu/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/8a0442e81540efaeb1a0fe3e95477b5e0edfd423 \
#    --tokenizer_name /mntcephfs/data/ruoyusun/liziniu/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/8a0442e81540efaeb1a0fe3e95477b5e0edfd423 \
#    --use_slow_tokenizer \
#    --train_file /mntcephfs/lab_data/wangsenmiao/sft_datasets/tulu_v2/MedMCQA_train_dataset_explain.jsonl \
#    --max_seq_length 512 \
#    --preprocessing_num_workers 32 \
#    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
#    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
#    --learning_rate 2e-5 \
#    --lr_scheduler_type linear \
#    --warmup_ratio 0.03 \
#    --weight_decay 0. \
#    --num_train_epochs 8 \
#    --use_AdamW_shun \
#    --shun_fraction 0.10 \
#    --output_dir /mntcephfs/data/ruoyusun/chenyupeng/llama2-7b/sft_med_36k_explain_shun_010_epoch8 \
#    --with_tracking \
#    --report_to tensorboard \
#    --logging_steps 1 \
#
#sleep 10
#
#accelerate launch \
#    --mixed_precision bf16 \
#    --num_machines 1 \
#    --num_processes $NUM_GPUS \
#    --use_deepspeed \
#    --deepspeed_config_file /home/wangsenmiao/yupeng_gpt/open-instruct-main/ds_configs/stage3_no_offloading_accelerate.conf \
#    /home/wangsenmiao/yupeng_gpt/open-instruct-main/open_instruct/finetune_med.py \
#    --model_name_or_path /mntcephfs/data/ruoyusun/liziniu/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/8a0442e81540efaeb1a0fe3e95477b5e0edfd423 \
#    --tokenizer_name /mntcephfs/data/ruoyusun/liziniu/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/8a0442e81540efaeb1a0fe3e95477b5e0edfd423 \
#    --use_slow_tokenizer \
#    --train_file /mntcephfs/lab_data/wangsenmiao/sft_datasets/tulu_v2/MedMCQA_train_dataset_explain.jsonl \
#    --max_seq_length 512 \
#    --preprocessing_num_workers 32 \
#    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
#    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
#    --learning_rate 2e-5 \
#    --lr_scheduler_type linear \
#    --warmup_ratio 0.03 \
#    --weight_decay 0. \
#    --num_train_epochs 10 \
#    --use_AdamW_shun \
#    --shun_fraction 0.05 \
#    --output_dir /mntcephfs/data/ruoyusun/chenyupeng/llama2-7b/sft_med_36k_explain_shun_005_epoch10 \
#    --with_tracking \
#    --report_to tensorboard \
#    --logging_steps 1 \
#
#
#sleep 10
#
#accelerate launch \
#    --mixed_precision bf16 \
#    --num_machines 1 \
#    --num_processes $NUM_GPUS \
#    --use_deepspeed \
#    --deepspeed_config_file /home/wangsenmiao/yupeng_gpt/open-instruct-main/ds_configs/stage3_no_offloading_accelerate.conf \
#    /home/wangsenmiao/yupeng_gpt/open-instruct-main/open_instruct/finetune_med.py \
#    --model_name_or_path /mntcephfs/data/ruoyusun/liziniu/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/8a0442e81540efaeb1a0fe3e95477b5e0edfd423 \
#    --tokenizer_name /mntcephfs/data/ruoyusun/liziniu/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/8a0442e81540efaeb1a0fe3e95477b5e0edfd423 \
#    --use_slow_tokenizer \
#    --train_file /mntcephfs/lab_data/wangsenmiao/sft_datasets/tulu_v2/MedMCQA_train_dataset_explain.jsonl \
#    --max_seq_length 512 \
#    --preprocessing_num_workers 32 \
#    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
#    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
#    --learning_rate 2e-5 \
#    --lr_scheduler_type linear \
#    --warmup_ratio 0.03 \
#    --weight_decay 0. \
#    --num_train_epochs 12 \
#    --use_AdamW_shun \
#    --shun_fraction 0.02 \
#    --output_dir /mntcephfs/data/ruoyusun/chenyupeng/llama2-7b/sft_med_36k_explain_shun_002_epoch15 \
#    --with_tracking \
#    --report_to tensorboard \
#    --logging_steps 1 \

