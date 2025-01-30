


load_dirs=(
path_of_fine-tuned_ckpt
)
save_dirs=(
path_where_you_want_to_save_evaluation
)


cd open-instruct

for i in "${!save_dirs[@]}"; do
  save_dir=${save_dirs[$i]}
  load_dir=${load_dirs[$i]}

export CUDA_VISIBLE_DEVICES=0
python ./open-instruct/eval/mmlu/run_eval.py \
        --ntrain 0 \
        --data_dir path_of_save_eval_data/data/eval/mmlu \
        --save_dir $save_dir/results_benchmarks/mmlu \
        --model_name_or_path $load_dir  \
        --tokenizer_name_or_path $load_dir  \
        --eval_batch_size 16

export CUDA_VISIBLE_DEVICES=0,1
torchrun --nproc-per-node=2 --master_port=25902 --no-python lm_eval --model hf \
        --model_args pretrained=$load_dir \
        --tasks hellaswag,arc_challenge,arc_easy \
        --output_path $save_dir/results_benchmarks/commonsense \
        --batch_size 32

export CUDA_VISIBLE_DEVICES=0,1
torchrun --nproc-per-node=2 --master_port=25902 --no-python lm_eval --model hf \
        --model_args pretrained=$load_dir \
        --tasks gsm8k \
        --output_path $save_dir/results_benchmarks/gsm8k_lm_eval \
        --batch_size 16

#export CUDA_VISIBLE_DEVICES=0
#python ./open-instruct/eval/codex_humaneval/run_eval_vllm.py \
#    --data_file path_of_save_eval_data/data/eval/codex_humaneval/HumanEval.jsonl.gz \
#    --eval_pass_at_ks 10 \
#    --unbiased_sampling_size_n 20 \
#    --temperature 0.8 \
#    --save_dir $save_dir/results_benchmarks/codex_humaneval_10_8_vllm \
#    --model $load_dir \
#    --tokenizer $load_dir \
#    --use_vllm

export CUDA_VISIBLE_DEVICES=0
python ./open-instruct/eval/codex_humaneval/run_eval.py \
    --data_file path_of_save_eval_data/eval/codex_humaneval/HumanEval.jsonl.gz \
    --eval_pass_at_ks 10 \
    --unbiased_sampling_size_n 20 \
    --temperature 0.8 \
    --save_dir $save_dir/results_benchmarks/codex_humaneval_10_8\
    --eval_batch_size 16 \
    --model $load_dir \
    --tokenizer $load_dir

#torchrun --nproc-per-node=2 --master_port=25902 --no-python lm_eval --model hf \
#        --model_args pretrained=$load_dir \
#        --tasks ifeval \
#        --output_path $load_dir/results_benchmarks/ifeval \
#        --batch_size 32
#
#torchrun --nproc-per-node=2 --master_port=25902 --no-python lm_eval --model hf \
#        --model_args pretrained=$load_dir \
#        --tasks pubmedqa,medqa_4options,medmcqa \
#        --output_path $save_dir/results_benchmarks/medqa \
#        --batch_size 32

done

