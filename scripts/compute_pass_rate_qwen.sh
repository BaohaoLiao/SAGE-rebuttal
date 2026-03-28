#!/bin/bash
set -xeuo pipefail

# Model
model_name_or_path="Qwen/Qwen2.5-7B-Instruct"
model_name="Qwen2.5-7B-Instruct"

# Dataset
dataset_path="./data/train.parquet"

# Generation settings
K=10
max_input_length=2048
max_new_tokens=8192
temperature=1.0
top_p=1.0
seed=42

# GPU settings — use free GPUs only
GPUS=(2 3 4 5 7)
NUM_GPUS=${#GPUS[@]}
gpu_memory_utilization=0.9
max_num_seqs=512

# Output
output_path="./results/${model_name}_train_pass_rate.jsonl"
tmp_dir="./results/.tmp_${model_name}_chunks"
mkdir -p "${tmp_dir}"

# Split dataset into chunks (one per GPU)
python3 -c "
import pandas as pd, math
df = pd.read_parquet('${dataset_path}')
n = len(df)
num_chunks = ${NUM_GPUS}
chunk_size = math.ceil(n / num_chunks)
for i in range(num_chunks):
    chunk = df.iloc[i*chunk_size : (i+1)*chunk_size]
    chunk.to_parquet('${tmp_dir}/chunk_{}.parquet'.format(i), index=False)
    print(f'Chunk {i}: {len(chunk)} rows')
"

# Launch one process per GPU in parallel
pids=()
for i in $(seq 0 $((NUM_GPUS - 1))); do
    gpu_id=${GPUS[$i]}
    chunk_path="${tmp_dir}/chunk_${i}.parquet"
    chunk_output="${tmp_dir}/chunk_${i}.jsonl"

    export CUDA_VISIBLE_DEVICES=${gpu_id}
    python3 scripts/compute_pass_rate.py \
        --model_name_or_path ${model_name_or_path} \
        --dataset_path ${chunk_path} \
        --max_num_seqs ${max_num_seqs} \
        --K ${K} \
        --max_input_length ${max_input_length} \
        --max_new_tokens ${max_new_tokens} \
        --temperature ${temperature} \
        --top_p ${top_p} \
        --seed ${seed} \
        --tensor_parallel_size 1 \
        --gpu_memory_utilization ${gpu_memory_utilization} \
        --use_oat_grader \
        --output_path ${chunk_output} &
    pids+=($!)
    echo "Launched GPU ${gpu_id} (chunk ${i}) — PID $!"
done

# Wait for all processes
echo "Waiting for ${NUM_GPUS} workers..."
failed=0
for pid in "${pids[@]}"; do
    if ! wait ${pid}; then
        echo "Process ${pid} failed"
        failed=1
    fi
done

if [ ${failed} -ne 0 ]; then
    echo "Some workers failed. Check logs above."
    exit 1
fi

# Merge chunk results into final output
echo "Merging results..."
cat ${tmp_dir}/chunk_*.jsonl > ${output_path}

# Generate summary
python3 -c "
import json, numpy as np
results = [json.loads(l) for l in open('${output_path}')]
pass_at_1 = [r['pass_at_1'] for r in results]
any_correct = [any(r['scores']) for r in results]
print(f'Total problems: {len(results)}')
print(f'Average pass@1: {np.mean(pass_at_1):.4f}')
print(f'Pass rate (any/K): {np.mean(any_correct):.4f}')

summary_path = '${output_path}'.replace('.jsonl', '_summary.txt')
with open(summary_path, 'w') as f:
    f.write(f'model: ${model_name_or_path}\n')
    f.write(f'dataset: ${dataset_path}\n')
    f.write(f'num_problems: {len(results)}\n')
    f.write(f'K: ${K}\n')
    f.write(f'avg_pass_at_1: {np.mean(pass_at_1):.4f}\n')
    f.write(f'pass_rate_any: {np.mean(any_correct):.4f}\n')
print(f'Summary saved to {summary_path}')
"

echo "Done! Results saved to ${output_path}"
