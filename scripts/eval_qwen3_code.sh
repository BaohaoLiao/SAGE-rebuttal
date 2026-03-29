#!/bin/bash
# Evaluate Qwen3-4B-Instruct-2507 on LiveCodeBench + Codeforces
# using the rllm framework (DeepCoder-style evaluation)

set -e

MODEL_NAME="Qwen/Qwen3-4B-Instruct-2507"
PORT=30000
N_PARALLEL=64
MAX_RESPONSE_LENGTH=8192
MAX_PROMPT_LENGTH=4096
TEMPERATURE=0.6
TOP_P=0.95

# Number of GPUs for vLLM (tensor parallel)
TP_SIZE=1  # Qwen3-4B fits on a single GPU

cd /workspace/baohao/SAGE-rebuttal

# ============================================================
# Step 1: Start vLLM server in background
# ============================================================
echo "Starting vLLM server for ${MODEL_NAME} on port ${PORT}..."
python -m vllm.entrypoints.openai.api_server \
    --model ${MODEL_NAME} \
    --host 0.0.0.0 \
    --port ${PORT} \
    --dtype bfloat16 \
    --tensor-parallel-size ${TP_SIZE} \
    --max-model-len 32768 \
    --trust-remote-code &
VLLM_PID=$!

# Wait for server to be ready (use python since curl may not be installed)
echo "Waiting for vLLM server to start..."
for i in $(seq 1 120); do
    if python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT}/health')" > /dev/null 2>&1; then
        echo "vLLM server is ready."
        break
    fi
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "ERROR: vLLM server process died."
        exit 1
    fi
    sleep 5
done

if ! python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT}/health')" > /dev/null 2>&1; then
    echo "ERROR: vLLM server failed to start within timeout."
    kill $VLLM_PID 2>/dev/null
    exit 1
fi

# ============================================================
# Step 2: Run evaluation
# ============================================================
echo "Running evaluation on LiveCodeBench + Codeforces..."
cd /workspace/baohao/SAGE-rebuttal/rllm

python -c "
import asyncio
import os
import sys
from datetime import datetime

os.environ['TOKENIZERS_PARALLELISM'] = 'true'
sys.path.insert(0, '.')

from transformers import AutoTokenizer

from rllm.agents.code_agent import CompetitionCodingAgent
from rllm.data.dataset import DatasetRegistry
from rllm.engine.agent_execution_engine import AgentExecutionEngine
from rllm.environments.base.single_turn_env import SingleTurnEnvironment
from rllm.rewards.reward_fn import code_reward_fn
from rllm.utils.compute_pass_at_k import save_trajectories

model_name = '${MODEL_NAME}'
port = ${PORT}
n_parallel = ${N_PARALLEL}

tokenizer = AutoTokenizer.from_pretrained(model_name)
reward_fn = code_reward_fn

env_args = {'reward_fn': reward_fn}
sampling_params = {
    'temperature': ${TEMPERATURE},
    'top_p': ${TOP_P},
    'model': model_name,
}

engine = AgentExecutionEngine(
    agent_class=CompetitionCodingAgent,
    env_class=SingleTurnEnvironment,
    agent_args={},
    env_args=env_args,
    engine_name='openai',
    tokenizer=tokenizer,
    sampling_params=sampling_params,
    rollout_engine_args={
        'base_url': f'http://localhost:{port}/v1',
        'api_key': 'None',
    },
    max_response_length=${MAX_RESPONSE_LENGTH},
    max_prompt_length=${MAX_PROMPT_LENGTH},
    n_parallel_agents=n_parallel,
)

# Load or prepare the dataset (LiveCodeBench + Codeforces)
test_dataset = DatasetRegistry.load_dataset('deepcoder', 'test')
if test_dataset is None:
    print('Dataset not found, preparing test dataset in memory...')
    import json
    from datasets import concatenate_datasets
    from datasets import load_dataset as hf_load_dataset
    from rllm.data.utils import fetch_live_code_bench_system_prompt
    from rllm.data.dataset import Dataset

    raw_test = concatenate_datasets([
        hf_load_dataset('agentica-org/DeepCoder-Preview-Dataset', name='codeforces', split='test'),
        hf_load_dataset('agentica-org/DeepCoder-Preview-Dataset', name='lcbv5', split='test'),
    ])

    def preprocess_fn(example, idx):
        starter_code = example.get('starter_code', '')
        question = fetch_live_code_bench_system_prompt(example['problem'], starter_code if starter_code else None)
        tests_raw = example['tests']
        if isinstance(tests_raw, str):
            tests = json.loads(tests_raw)
        else:
            tests = tests_raw
        metadata = example.get('metadata', {})
        if isinstance(tests, dict) and 'inputs' in tests and 'outputs' in tests:
            normalized_tests = []
            for input_val, output_val in zip(tests['inputs'], tests['outputs'], strict=False):
                normalized_tests.append({'input': input_val, 'output': output_val, 'testtype': 'stdin_stdout'})
            tests = normalized_tests
        if not isinstance(tests, list):
            tests = [tests] if tests else []
        for test in tests:
            if test.get('testtype') == 'functional' and metadata.get('func_name') is not None:
                test['metadata'] = {'func_name': str(metadata['func_name'])}
            else:
                test['metadata'] = {'func_name': None}
        return {'question': question, 'ground_truth': json.dumps(tests), 'data_source': 'livecodebench', 'uid': f'deepcoder_{idx}', 'index': idx, 'starter_code': starter_code, 'metadata': json.dumps(metadata)}

    raw_test = raw_test.map(preprocess_fn, with_indices=True, writer_batch_size=10, num_proc=16)
    tasks = [dict(row) for row in raw_test]
    test_dataset = Dataset(data=tasks, name='deepcoder', split='test')
    print(f'Prepared {len(tasks)} test examples in memory (skipped parquet).')
else:
    tasks = test_dataset.get_data()
n_codeforces = 279  # first 279 tasks are Codeforces
n_lcb = len(tasks) - n_codeforces  # remaining are LiveCodeBench v5
print(f'Evaluating {len(tasks)} tasks ({n_codeforces} Codeforces + {n_lcb} LiveCodeBench)...')

results = asyncio.run(engine.execute_tasks(tasks))

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
try:
    output_file = f'qwen3_4b_code_eval_{timestamp}.pt'
    save_trajectories(results, save_dir='/workspace/baohao/SAGE-rebuttal/outputs', filename=output_file)
    print(f'Results saved to /workspace/baohao/SAGE-rebuttal/outputs/{output_file}')
except Exception as e:
    print(f'Warning: torch save failed ({e}), saving as JSON...')
    import json as json2
    output_file = f'/workspace/baohao/SAGE-rebuttal/outputs/qwen3_4b_code_eval_{timestamp}.json'
    json_results = [r.to_dict() if hasattr(r, 'to_dict') else str(r) for r in results]
    with open(output_file, 'w') as f:
        json2.dump(json_results, f)
    print(f'Results saved to {output_file}')

# Print per-dataset summary
cf_correct = sum(1 for r in results[:n_codeforces] if getattr(r, 'reward', 0) and r.reward > 0)
lcb_correct = sum(1 for r in results[n_codeforces:] if getattr(r, 'reward', 0) and r.reward > 0)
total_correct = cf_correct + lcb_correct
print(f'Codeforces  : {cf_correct}/{n_codeforces} = {cf_correct/n_codeforces*100:.1f}%')
print(f'LiveCodeBench: {lcb_correct}/{n_lcb} = {lcb_correct/n_lcb*100:.1f}%')
print(f'Overall      : {total_correct}/{len(results)} = {total_correct/len(results)*100:.1f}%')
"

# ============================================================
# Step 3: Clean up
# ============================================================
echo "Shutting down vLLM server..."
kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null || true

echo "Evaluation complete!"
