#!/bin/bash
set -xeuo pipefail

# Model
model_name_or_path="Qwen/Qwen2.5-7B-Instruct"

# Data
train_files="data/Qwen2.5-7B-Instruct_sft_train.parquet"

# Training
num_gpus=8
train_batch_size=64
micro_batch_size_per_gpu=2
max_length=11000
lr=5e-6
total_epochs=3
save_freq=100

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

torchrun --nproc_per_node=${num_gpus} \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files=${train_files} \
    data.val_files=${train_files} \
    data.prompt_key=prompt \
    data.response_key=response \
    data.train_batch_size=${train_batch_size} \
    data.micro_batch_size_per_gpu=${micro_batch_size_per_gpu} \
    data.max_length=${max_length} \
    model.partial_pretrain=${model_name_or_path} \
    model.fsdp_config.model_dtype=bf16 \
    model.enable_gradient_checkpointing=true \
    model.strategy=fsdp2 \
    optim.lr=${lr} \
    optim.lr_scheduler=cosine \
    optim.lr_warmup_steps_ratio=0.1 \
    optim.weight_decay=0.01 \
    optim.clip_grad=1.0 \
    trainer.total_epochs=${total_epochs} \
    trainer.project_name=sft \
    trainer.experiment_name=Qwen2.5-7B-Instruct_sft \
    trainer.default_local_dir=outputs/sft_Qwen2.5-7B-Instruct \
    trainer.save_freq=${save_freq} \
    trainer.logger="['console']"
