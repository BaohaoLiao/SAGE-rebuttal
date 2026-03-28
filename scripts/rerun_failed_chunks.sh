#!/bin/bash
set -xeuo pipefail

model_name_or_path="meta-llama/Llama-3.2-3B-Instruct"
tmp_dir="./results/.tmp_Llama-3.2-3B-Instruct_chunks"

FAILED_CHUNKS=(0 1 6)

pids=()
for i in "${FAILED_CHUNKS[@]}"; do
    export CUDA_VISIBLE_DEVICES=$i
    python3 scripts/compute_pass_rate.py \
        --model_name_or_path ${model_name_or_path} \
        --dataset_path ${tmp_dir}/chunk_${i}.parquet \
        --max_num_seqs 1024 \
        --K 10 \
        --max_input_length 2048 \
        --max_new_tokens 8192 \
        --temperature 1.0 \
        --top_p 1.0 \
        --seed 42 \
        --tensor_parallel_size 1 \
        --gpu_memory_utilization 0.9 \
        --use_oat_grader \
        --output_path ${tmp_dir}/chunk_${i}.jsonl &
    pids+=($!)
    echo "Launched chunk $i on GPU $i — PID $!"
done

echo "Waiting for ${#pids[@]} workers..."
for pid in "${pids[@]}"; do
    wait ${pid}
done
echo "All failed chunks re-done!"

# Merge all chunks
output_path="./results/Llama-3.2-3B-Instruct_train_pass_rate.jsonl"
cat ${tmp_dir}/chunk_*.jsonl > ${output_path}

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
    f.write(f'dataset: ./data/train.parquet\n')
    f.write(f'num_problems: {len(results)}\n')
    f.write(f'K: 10\n')
    f.write(f'avg_pass_at_1: {np.mean(pass_at_1):.4f}\n')
    f.write(f'pass_rate_any: {np.mean(any_correct):.4f}\n')
print(f'Summary saved to {summary_path}')
"
echo "Done! Results saved to ${output_path}"
