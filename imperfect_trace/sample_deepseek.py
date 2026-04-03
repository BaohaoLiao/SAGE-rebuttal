"""
Sample DeepSeek-V3.2 responses (with thinking) for questions that Qwen3-4B got all wrong (pass@8=0).

Sends concurrent requests to maximize throughput on the vLLM server.

Usage:
    python imperfect_trace/sample_deepseek.py
    python imperfect_trace/sample_deepseek.py --n 8 --concurrency 16
"""

import argparse
import asyncio
import pandas as pd
from tqdm import tqdm
from openai import AsyncOpenAI
from verl.utils.reward_score.math_verify import compute_score


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_path", type=str,
                        default="imperfect_trace/results.parquet")
    parser.add_argument("--output_path", type=str,
                        default="imperfect_trace/deepseek_results.parquet")
    parser.add_argument("--api_base", type=str,
                        default="http://localhost:8000/v1")
    parser.add_argument("--model", type=str,
                        default="/data/agenthle/baohao/LLMs/deepseek-ai/DeepSeek-V3.2")
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--concurrency", type=int, default=16,
                        help="Max concurrent API requests")
    return parser.parse_args()


def parse_thinking(text):
    """Split response into (reasoning, content) based on </think> tag."""
    if "</think>" in text:
        parts = text.split("</think>", 1)
        reasoning = parts[0].strip()
        content = parts[1].strip()
        return reasoning, content
    return "", text.strip()


def main():
    args = parse_args()

    # Load results and select all-wrong questions
    df = pd.read_parquet(args.results_path)
    all_wrong = df[df["num_correct"] == 0].copy()
    print(f"Total questions: {len(df)}, all-wrong (pass@8=0): {len(all_wrong)}")

    client = AsyncOpenAI(base_url=args.api_base, api_key="dummy")
    semaphore = asyncio.Semaphore(args.concurrency)
    pbar = tqdm(total=len(all_wrong), desc="Sampling")

    async def sample_row(row):
        messages = [{"role": "user", "content": row["problem"]}]
        async with semaphore:
            response = await client.chat.completions.create(
                model=args.model,
                messages=messages,
                n=args.n,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                extra_body={"chat_template_kwargs": {"enable_thinking": True}},
            )

        reasoning_list, content_list, full_response_list, scores = [], [], [], []
        for choice in response.choices:
            full_text = choice.message.content or ""
            reasoning, content = parse_thinking(full_text)
            reasoning_list.append(reasoning)
            content_list.append(content)
            full_response_list.append(full_text)
            scores.append(compute_score(content, row["answer"]))

        pbar.update(1)
        return {
            "index": row["index"],
            "data_source": row["data_source"],
            "problem": row["problem"],
            "answer": row["answer"],
            "reasoning": reasoning_list,
            "content": content_list,
            "full_responses": full_response_list,
            "scores": scores,
            "num_correct": sum(scores),
            "accuracy": sum(scores) / len(scores) if scores else 0.0,
        }

    async def run():
        tasks = [sample_row(row) for _, row in all_wrong.iterrows()]
        return await asyncio.gather(*tasks)

    results = asyncio.run(run())
    pbar.close()

    result_df = pd.DataFrame(results)
    result_df.to_parquet(args.output_path, index=False)

    # Stats
    n = args.n
    total = len(result_df)
    pass1 = result_df["accuracy"].mean()
    pass_n = (result_df["num_correct"] > 0).mean()
    still_wrong = (result_df["num_correct"] == 0).sum()
    print(f"\n=== DeepSeek-V3.2 Results on {total} all-wrong questions ===")
    print(f"Pass@1: {pass1:.4f}")
    print(f"Pass@{n}: {pass_n:.4f}")
    print(f"Still all-wrong: {still_wrong}/{total} ({still_wrong/total*100:.1f}%)")
    print(f"Saved to {args.output_path}")


if __name__ == "__main__":
    main()
