"""
Compute pass@1 rate of a model on a parquet dataset (e.g. data/train.parquet).

Usage:
    python scripts/compute_pass_rate.py \
        --model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
        --dataset_path data/train.parquet \
        --K 8 \
        --max_input_length 2048 \
        --max_new_tokens 2048 \
        --output_path results/train_pass_rate.jsonl
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from eval.compute_score import compute_score
from eval.oat_math_grader import boxed_reward_fn as oat_evaluate


def parse_args():
    parser = argparse.ArgumentParser(description="Compute pass rate on a parquet dataset")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, default="data/train.parquet")
    parser.add_argument("--K", type=int, default=8, help="Number of samples per prompt")
    parser.add_argument("--max_input_length", type=int, default=2048)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_path", type=str, default=None, help="Save per-sample results to jsonl")
    parser.add_argument("--use_oat_grader", action="store_true", help="Use OAT grader instead of math_verify")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load dataset
    df = pd.read_parquet(args.dataset_path)
    print(f"Loaded {len(df)} problems from {args.dataset_path}")

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    llm = LLM(
        model=args.model_name_or_path,
        tokenizer=args.model_name_or_path,
        dtype="bfloat16",
        max_model_len=args.max_input_length,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=args.seed,
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
        n=args.K,
        stop_token_ids=[tokenizer.eos_token_id],
        seed=args.seed,
    )

    # Build prompts
    system_prompt = "Please reason step by step, and put your final answer within \\boxed{}."
    prompts = []
    problems = []
    ground_truths = []

    for _, row in df.iterrows():
        problem = row.get("problem") or row.get("question", "")
        answer = row.get("answer", "")
        if not answer and "reward_model" in row and isinstance(row["reward_model"], dict):
            answer = row["reward_model"].get("ground_truth", "")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": problem},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt)
        problems.append(problem)
        ground_truths.append(str(answer))

    print(f"Generating {args.K} responses for {len(prompts)} problems...")
    outputs = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=True)

    # Score
    print("Scoring responses...")
    all_scores = []
    results = []
    for i, output in enumerate(tqdm(outputs, desc="Scoring")):
        responses = [out.text for out in output.outputs]
        gt = ground_truths[i]

        scores = []
        for response in responses:
            if args.use_oat_grader:
                score = oat_evaluate(response, gt, fast=False)
                scores.append(score[1] == 1.0)
            else:
                score = compute_score(response, gt)
                scores.append(bool(score))
        all_scores.append(scores)

        results.append({
            "problem": problems[i],
            "gt": gt,
            "responses": responses,
            "scores": scores,
            "pass_at_1": float(np.mean(scores)),
        })

    # Compute metrics
    pass_at_1_per_problem = [np.mean(s) for s in all_scores]
    any_correct_per_problem = [any(s) for s in all_scores]

    avg_pass_at_1 = np.mean(pass_at_1_per_problem)
    pass_rate = np.mean(any_correct_per_problem)

    print(f"\n{'='*60}")
    print(f"Model: {args.model_name_or_path}")
    print(f"Dataset: {args.dataset_path} ({len(df)} problems)")
    print(f"K = {args.K}")
    print(f"{'='*60}")
    print(f"Average pass@1:     {avg_pass_at_1:.4f}")
    print(f"Pass rate (any/K):  {pass_rate:.4f}")
    print(f"{'='*60}")

    # Save results
    if args.output_path:
        os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
        with open(args.output_path, "w", encoding="utf8") as f:
            for r in results:
                json.dump(r, f, ensure_ascii=False)
                f.write("\n")

        summary_path = args.output_path.replace(".jsonl", "_summary.txt")
        with open(summary_path, "w") as f:
            f.write(f"model: {args.model_name_or_path}\n")
            f.write(f"dataset: {args.dataset_path}\n")
            f.write(f"num_problems: {len(df)}\n")
            f.write(f"K: {args.K}\n")
            f.write(f"avg_pass_at_1: {avg_pass_at_1:.4f}\n")
            f.write(f"pass_rate_any: {pass_rate:.4f}\n")
        print(f"Results saved to {args.output_path}")
        print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
