#!/bin/bash
set -xeuo pipefail

# Model
model_name_or_path="meta-llama/Llama-3.2-3B-Instruct"
model_name="Llama-3.2-3B-Instruct"

# Dataset
dataset_path="./data/train.parquet"

# Generation settings
K=10
max_input_length=2048
max_new_tokens=8192
temperature=1.0
top_p=1.0
seed=42

# GPU settings
export CUDA_VISIBLE_DEVICES=4
tensor_parallel_size=1
gpu_memory_utilization=0.9

# Output
output_path="./results/${model_name}_train_pass_rate.jsonl"

python3 scripts/compute_pass_rate.py \
    --model_name_or_path ${model_name_or_path} \
    --dataset_path ${dataset_path} \
    --K ${K} \
    --max_input_length ${max_input_length} \
    --max_new_tokens ${max_new_tokens} \
    --temperature ${temperature} \
    --top_p ${top_p} \
    --seed ${seed} \
    --tensor_parallel_size ${tensor_parallel_size} \
    --gpu_memory_utilization ${gpu_memory_utilization} \
    --use_oat_grader \
    --output_path ${output_path}
