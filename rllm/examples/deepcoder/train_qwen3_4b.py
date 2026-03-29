import json
import os
import sys

import hydra
from datasets import concatenate_datasets, load_dataset as hf_load_dataset

from rllm.agents.code_agent import CompetitionCodingAgent
from rllm.data.dataset import Dataset, DatasetRegistry
from rllm.data.utils import fetch_live_code_bench_system_prompt
from rllm.environments.base.single_turn_env import SingleTurnEnvironment
from rllm.rewards.reward_fn import code_reward_fn
from rllm.trainer.agent_trainer import AgentTrainer


VERL_PARQUET_DIR = os.path.join(os.path.dirname(__file__), ".verl_cache")


def _save_verl_parquet(data_list, path):
    """Save verl-compatible parquet with small row groups to avoid INT32_MAX overflow."""
    import pandas as pd

    verl_data = DatasetRegistry.apply_verl_postprocessing(data_list)
    df = pd.DataFrame(verl_data)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Use small row_group_size to keep each row group under INT32_MAX
    df.to_parquet(path, engine="pyarrow", row_group_size=50)
    print(f"Saved verl parquet: {path} ({len(verl_data)} rows)")


def prepare_datasets():
    """Prepare DeepCoder train/test datasets in memory and save verl parquet files."""
    train_verl_path = os.path.join(VERL_PARQUET_DIR, "train_verl.parquet")
    test_verl_path = os.path.join(VERL_PARQUET_DIR, "test_verl.parquet")

    # Check if already cached
    train_dataset = DatasetRegistry.load_dataset("deepcoder", "train")
    test_dataset = DatasetRegistry.load_dataset("deepcoder", "test")

    if train_dataset is not None and test_dataset is not None:
        # Still need verl parquet files
        if not os.path.exists(train_verl_path):
            _save_verl_parquet(train_dataset.data, train_verl_path)
        if not os.path.exists(test_verl_path):
            _save_verl_parquet(test_dataset.data, test_verl_path)
        return train_dataset, test_dataset, train_verl_path, test_verl_path

    print("Preparing DeepCoder datasets from HuggingFace...")

    raw_train = concatenate_datasets([
        hf_load_dataset("agentica-org/DeepCoder-Preview-Dataset", name="primeintellect", split="train"),
        hf_load_dataset("agentica-org/DeepCoder-Preview-Dataset", name="taco", split="train"),
        # hf_load_dataset("agentica-org/DeepCoder-Preview-Dataset", name="lcbv5", split="train"),
    ])
    raw_test = concatenate_datasets([
        hf_load_dataset("agentica-org/DeepCoder-Preview-Dataset", name="codeforces", split="test"),
        hf_load_dataset("agentica-org/DeepCoder-Preview-Dataset", name="lcbv5", split="test"),
    ])

    def preprocess_fn(example, idx):
        starter_code = example.get("starter_code", "")
        question = fetch_live_code_bench_system_prompt(example["problem"], starter_code if starter_code else None)
        tests_raw = example["tests"]
        if isinstance(tests_raw, str):
            tests = json.loads(tests_raw)
        else:
            tests = tests_raw
        metadata = example.get("metadata", {})
        if isinstance(tests, dict) and "inputs" in tests and "outputs" in tests:
            normalized_tests = []
            for input_val, output_val in zip(tests["inputs"], tests["outputs"], strict=False):
                normalized_tests.append({"input": input_val, "output": output_val, "testtype": "stdin_stdout"})
            tests = normalized_tests
        if not isinstance(tests, list):
            tests = [tests] if tests else []
        for test in tests:
            if test.get("testtype") == "functional" and metadata.get("func_name") is not None:
                test["metadata"] = {"func_name": str(metadata["func_name"])}
            else:
                test["metadata"] = {"func_name": None}
        return {
            "question": question,
            "ground_truth": json.dumps(tests),
            "data_source": "livecodebench",
            "uid": f"deepcoder_{idx}",
            "index": idx,
            "starter_code": starter_code,
            "metadata": json.dumps(metadata),
        }

    raw_train = raw_train.map(preprocess_fn, with_indices=True, writer_batch_size=10, num_proc=16)
    raw_test = raw_test.map(preprocess_fn, with_indices=True, writer_batch_size=10, num_proc=16)

    train_data = [dict(row) for row in raw_train]
    test_data = [dict(row) for row in raw_test]

    # Save verl parquet files
    _save_verl_parquet(train_data, train_verl_path)
    _save_verl_parquet(test_data, test_verl_path)

    train_dataset = Dataset(data=train_data, name="deepcoder", split="train")
    test_dataset = Dataset(data=test_data, name="deepcoder", split="test")

    print(f"Train: {len(train_data)} examples, Test: {len(test_data)} examples")
    return train_dataset, test_dataset, train_verl_path, test_verl_path


@hydra.main(config_path="pkg://rllm.trainer.config", config_name="agent_ppo_trainer", version_base=None)
def main(config):
    train_dataset, test_dataset, train_verl_path, test_verl_path = prepare_datasets()

    # Set verl data paths directly in config so verl trainer can find them
    config.data.train_files = train_verl_path
    config.data.val_files = test_verl_path

    env_args = {"reward_fn": code_reward_fn}

    trainer = AgentTrainer(
        agent_class=CompetitionCodingAgent,
        agent_args={},
        env_args=env_args,
        env_class=SingleTurnEnvironment,
        config=config,
        train_dataset=train_dataset,
        val_dataset=test_dataset,
    )

    # Override after constructor since AgentTrainer.__init__ may set them to None
    trainer.config.data.train_files = train_verl_path
    trainer.config.data.val_files = test_verl_path

    trainer.train()


if __name__ == "__main__":
    main()
