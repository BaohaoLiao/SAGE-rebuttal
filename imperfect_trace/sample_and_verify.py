"""
Sample responses from Qwen3-4B-Instruct-2507 for train.parquet and verify answers
using the same math_verify reward function from the SAGE training pipeline.

Usage (single GPU):
    python imperfect_trace/sample_and_verify.py --tp 1 --num_samples 1000

Usage (8-GPU data parallel):
    python imperfect_trace/sample_and_verify.py --dp 8 --num_samples 1000
"""

import argparse
import os
import subprocess
import sys
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from verl.utils.reward_score.math_verify import compute_score


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str,
                        default="/data/agenthle/baohao/LLMs/Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--data_path", type=str,
                        default="imperfect_trace/train.parquet")
    parser.add_argument("--output_path", type=str,
                        default="imperfect_trace/results.parquet")
    parser.add_argument("--n", type=int, default=8,
                        help="Number of samples per prompt")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Only process first N samples (default: all)")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--tp", type=int, default=1,
                        help="Tensor parallel size")
    parser.add_argument("--dp", type=int, default=1,
                        help="Data parallel size (number of GPUs, each runs a model replica)")
    # Internal args for DP workers
    parser.add_argument("--_dp_rank", type=int, default=None)
    parser.add_argument("--_dp_world", type=int, default=None)
    return parser.parse_args()


def run_dp_coordinator(args):
    """Launch dp workers as subprocesses, each pinned to one GPU."""
    num_gpus = args.dp
    num_samples = args.num_samples
    output_dir = os.path.dirname(args.output_path) or "."
    base_name = os.path.splitext(os.path.basename(args.output_path))[0]

    procs = []
    shard_paths = []
    for rank in range(num_gpus):
        shard_path = os.path.join(output_dir, f"{base_name}_shard{rank}.parquet")
        shard_paths.append(shard_path)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(rank)
        cmd = [
            sys.executable, __file__,
            "--model_path", args.model_path,
            "--data_path", args.data_path,
            "--output_path", shard_path,
            "--n", str(args.n),
            "--temperature", str(args.temperature),
            "--top_p", str(args.top_p),
            "--max_tokens", str(args.max_tokens),
            "--tp", str(args.tp),
            "--_dp_rank", str(rank),
            "--_dp_world", str(num_gpus),
        ]
        if num_samples is not None:
            cmd += ["--num_samples", str(num_samples)]
        print(f"[Coordinator] Launching worker {rank} on GPU {rank}")
        procs.append(subprocess.Popen(cmd, env=env))

    # Wait for all workers
    for rank, proc in enumerate(procs):
        ret = proc.wait()
        if ret != 0:
            print(f"[Coordinator] Worker {rank} failed with exit code {ret}")
            sys.exit(1)
        print(f"[Coordinator] Worker {rank} finished")

    # Merge shards
    print(f"[Coordinator] Merging {num_gpus} shards...")
    shards = [pd.read_parquet(p) for p in shard_paths]
    merged = pd.concat(shards, ignore_index=True).sort_values("index").reset_index(drop=True)

    n = args.n
    total_correct = merged["num_correct"].sum()
    total_responses = len(merged) * n
    overall_acc = total_correct / total_responses if total_responses > 0 else 0
    pass_at_1 = merged["accuracy"].mean()
    pass_at_n = (merged["num_correct"] > 0).mean()
    all_wrong = (merged["num_correct"] == 0).mean()
    all_correct = (merged["num_correct"] == n).mean()

    print(f"\n{'='*60}")
    print(f"Results Summary (merged)")
    print(f"{'='*60}")
    print(f"Total prompts:     {len(merged)}")
    print(f"Samples per prompt: {n}")
    print(f"Overall accuracy:  {overall_acc:.4f}")
    print(f"Pass@1 (avg acc):  {pass_at_1:.4f}")
    print(f"Pass@{n}:            {pass_at_n:.4f}")
    print(f"All-wrong rate:    {all_wrong:.4f} ({int(all_wrong * len(merged))} prompts)")
    print(f"All-correct rate:  {all_correct:.4f} ({int(all_correct * len(merged))} prompts)")
    print(f"{'='*60}")

    merged.to_parquet(args.output_path)
    print(f"Merged results saved to {args.output_path}")

    # Cleanup shards
    for p in shard_paths:
        os.remove(p)


def main():
    args = parse_args()

    # DP coordinator mode
    if args.dp > 1 and args._dp_rank is None:
        run_dp_coordinator(args)
        return

    # Load data
    df = pd.read_parquet(args.data_path)
    if args.num_samples is not None:
        df = df.head(args.num_samples)

    # Shard for DP worker
    if args._dp_rank is not None:
        rank, world = args._dp_rank, args._dp_world
        df = df.iloc[rank::world].reset_index(drop=True)
        print(f"[Worker {rank}/{world}] Processing {len(df)} prompts on GPU {os.environ.get('CUDA_VISIBLE_DEVICES', '?')}")

    print(f"Loaded {len(df)} prompts from {args.data_path}")

    # Load tokenizer and apply chat template
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    prompts = []
    for messages in df["prompt"]:
        messages = [dict(m) for m in messages]
        prompt_text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False,
            enable_thinking=False,
        )
        prompts.append(prompt_text)

    # Load model
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tp,
        trust_remote_code=True,
        max_model_len=args.max_tokens + 2048,
    )

    sampling_params = SamplingParams(
        n=args.n,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    # Generate
    print(f"Generating {args.n} samples per prompt for {len(prompts)} prompts...")
    outputs = llm.generate(prompts, sampling_params)

    # Verify and collect results
    results = []
    total_correct = 0
    total_responses = 0

    # Use original index from the dataframe
    original_indices = df["extra_info"].apply(lambda x: x.get("index", 0)).tolist() \
        if "extra_info" in df.columns else list(range(len(df)))

    for i, (output, row) in enumerate(tqdm(zip(outputs, df.itertuples()),
                                            total=len(df), desc="Verifying")):
        ground_truth = row.answer
        responses = []
        scores = []
        for completion in output.outputs:
            response_text = completion.text
            score = compute_score(response_text, ground_truth)
            responses.append(response_text)
            scores.append(score)
            total_correct += score
            total_responses += 1

        results.append({
            "index": original_indices[i],
            "data_source": row.data_source,
            "problem": row.problem,
            "answer": ground_truth,
            "responses": responses,
            "scores": scores,
            "num_correct": sum(scores),
            "accuracy": sum(scores) / len(scores),
        })

    # Summary statistics
    result_df = pd.DataFrame(results)
    overall_acc = total_correct / total_responses if total_responses > 0 else 0
    pass_at_1 = result_df["accuracy"].mean()
    pass_at_n = (result_df["num_correct"] > 0).mean()
    all_wrong = (result_df["num_correct"] == 0).mean()
    all_correct = (result_df["num_correct"] == args.n).mean()

    prefix = f"[Worker {args._dp_rank}] " if args._dp_rank is not None else ""
    print(f"\n{'='*60}")
    print(f"{prefix}Results Summary")
    print(f"{'='*60}")
    print(f"Total prompts:     {len(result_df)}")
    print(f"Samples per prompt: {args.n}")
    print(f"Overall accuracy:  {overall_acc:.4f}")
    print(f"Pass@1 (avg acc):  {pass_at_1:.4f}")
    print(f"Pass@{args.n}:            {pass_at_n:.4f}")
    print(f"All-wrong rate:    {all_wrong:.4f} ({int(all_wrong * len(result_df))} prompts)")
    print(f"All-correct rate:  {all_correct:.4f} ({int(all_correct * len(result_df))} prompts)")
    print(f"{'='*60}")

    # Save
    result_df.to_parquet(args.output_path)
    print(f"Results saved to {args.output_path}")


if __name__ == "__main__":
    main()
