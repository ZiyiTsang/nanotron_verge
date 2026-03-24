#!/usr/bin/env python3
"""
Nanotron Ray Train Launcher

Ray Train-based multi-node distributed training launcher for Nanotron.
Designed to work alongside slurm_launcher.py for Ray Cluster environments.

Usage:
    # Start local Ray
    ray start --head

    # Option 1: Use existing YAML config
    python ray_launcher.py \
        --config examples/config_tiny_llama.yaml \
        --num-nodes 1 --gpus-per-node 2

    # Option 2: Auto-generate config
    python ray_launcher.py \
        --run my_experiment \
        --num-nodes 1 --gpus-per-node 2 \
        --dp 2 --pp 1 --tp 1 \
        --model 1b --steps 1000

    # Multi-node (requires Ray Cluster)
    ray start --head --port=6379
    # Worker: ray start --address='<head-ip>:6379'
    python ray_launcher.py \
        --config examples/config_tiny_llama.yaml \
        --num-nodes 2 --gpus-per-node 8
"""
import argparse
import os
import sys
from datetime import datetime
from typing import Optional

import ray
from ray.train import ScalingConfig, RunConfig, CheckpointConfig, FailureConfig
from ray.train.torch import TorchTrainer, TorchConfig

# Nanotron imports
from nanotron.config import (
    AdamWOptimizerArgs,
    CheckpointsArgs,
    Config,
    DataArgs,
    DatasetStageArgs,
    GeneralArgs,
    LlamaConfig,
    LoggingArgs,
    LRSchedulerArgs,
    ModelArgs,
    NanosetDatasetsArgs,
    OptimizerArgs,
    ParallelismArgs,
    ProfilerArgs,
    RandomInit,
    TokenizerArgs,
    TokensArgs,
)
from nanotron.logging import human_format

# =============================================
# CONFIGURATION SECTION - MODIFY AS NEEDED
# =============================================

DEFAULT_CHECKPOINTS_PATH = "checkpoints"
DEFAULT_STORAGE_PATH = "./ray_checkpoints"

MODEL_SIZES = {
    "160m": (12, 768, 12, 12, 3072),
    "410m": (24, 1024, 16, 16, 4096),
    "1b": (16, 2048, 16, 16, 5632),
    "3b": (28, 3072, 32, 32, 8192),
    "7b": (32, 4096, 32, 32, 11008),
    "13b": (40, 5120, 40, 40, 13824),
    "30b": (60, 6656, 52, 52, 17920),
    "70b": (80, 8192, 64, 8, 28672),
    "custom": (12, 192, 4, 4, 768),
}


def parse_args():
    """Parse command line arguments for the Ray launcher."""
    parser = argparse.ArgumentParser(
        description="Nanotron Ray Train Launcher",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--run", type=str, default="nanotron-ray", help="Name for this experiment run")

    # Resource configuration
    resource_group = parser.add_argument_group("Resource Configuration")
    resource_group.add_argument("--num-nodes", type=int, default=1, help="Number of nodes")
    resource_group.add_argument("--gpus-per-node", type=int, default=8, help="GPUs per node")

    # Config file
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to the Nanotron config file. If not provided, a config will be created automatically.",
    )

    # Model configuration
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument("--model", type=str, default="custom", choices=MODEL_SIZES.keys(), help="Predefined model size")
    model_group.add_argument("--hidden-size", type=int, default=None, help="Hidden size (overrides model)")
    model_group.add_argument("--intermediate-size", type=int, default=None, help="Intermediate size (overrides model)")
    model_group.add_argument("--num-layers", type=int, default=None, help="Number of layers (overrides model)")
    model_group.add_argument("--num-heads", type=int, default=None, help="Number of attention heads (overrides model)")
    model_group.add_argument("--num-kv-heads", type=int, default=None, help="Number of KV heads (overrides model)")
    model_group.add_argument("--vocab-size", type=int, default=65536, help="Vocabulary size (overrides model)")
    model_group.add_argument("--seq", type=int, default=4096, help="Maximum sequence length")

    # Training configuration
    training_group = parser.add_argument_group("Training Configuration")
    training_group.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    training_group.add_argument("--steps", type=int, default=10000, help="Number of training steps")
    training_group.add_argument("--mbs", type=int, default=2, help="Micro batch size")
    training_group.add_argument("--acc", type=int, default=8, help="Gradient accumulation steps")
    training_group.add_argument("--learning-rate", type=float, default=3e-4, help="Peak learning rate")
    training_group.add_argument("--min-lr", type=float, default=3e-5, help="Minimum learning rate for decay")
    training_group.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    training_group.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping")
    training_group.add_argument("--warmup-steps", type=int, default=1000, help="Learning rate warmup steps")

    # Parallelism strategy
    parallel_group = parser.add_argument_group("Parallelism Configuration")
    parallel_group.add_argument("--dp", type=int, default=8, help="Data parallelism (DP) degree")
    parallel_group.add_argument("--pp", type=int, default=1, help="Pipeline parallelism (PP) degree")
    parallel_group.add_argument("--tp", type=int, default=2, help="Tensor parallelism (TP) degree")
    parallel_group.add_argument("--cp", type=int, default=1, help="Context parallelism degree")
    parallel_group.add_argument("--ep", type=int, default=1, help="Expert parallelism degree")
    parallel_group.add_argument("--zero", type=int, default=0, choices=[0, 1], help="ZeRO stage")

    # Dataset configuration
    data_group = parser.add_argument_group("Dataset Configuration")
    data_group.add_argument("--dataset", type=str, default=None, help="Hugging Face dataset name or path")
    data_group.add_argument("--text-column", type=str, default="text", help="Column name for text in the dataset")
    data_group.add_argument(
        "--tokenizer", type=str, default="robot-test/dummy-tokenizer-wordlevel", help="Tokenizer name or path"
    )

    # Paths
    paths_group = parser.add_argument_group("Paths Configuration")
    paths_group.add_argument(
        "--checkpoints-path",
        type=str,
        default=DEFAULT_CHECKPOINTS_PATH,
        help="Base directory for saving model checkpoints",
    )
    paths_group.add_argument(
        "--project", type=str, default="nanotron", help="Project name for logging"
    )
    paths_group.add_argument(
        "--storage-path",
        type=str,
        default=DEFAULT_STORAGE_PATH,
        help="Ray Train checkpoint storage path (used by Ray Train, not Nanotron)",
    )
    paths_group.add_argument(
        "--checkpoint-interval", type=int, default=1000, help="Interval for saving checkpoints (in steps) - controls both Nanotron and Ray Train"
    )
    paths_group.add_argument("--save-initial-state", action="store_true", help="Save initial state")

    # Logging configuration
    logging_group = parser.add_argument_group("Logging Configuration")
    logging_group.add_argument(
        "--profiler-export-path",
        type=str,
        default=None,
        help="Path to export the profiler tensorboard data.",
    )
    logging_group.add_argument("--log-lvl", type=str, default="info", help="Log level")
    logging_group.add_argument("--no-sanity", action="store_true", help="Ignore sanity checks")

    # Ray Train specific
    ray_group = parser.add_argument_group("Ray Train Configuration")
    ray_group.add_argument("--num-keep-checkpoints", type=int, default=3, help="Number of Ray Train checkpoints to keep")
    ray_group.add_argument("--max-failures", type=int, default=3, help="Max worker failures before stopping")
    ray_group.add_argument(
        "--verify-parallelism", action="store_true", help="Verify TP*PP*DP*CP*EP matches num_workers"
    )
    ray_group.add_argument(
        "--dry-run", action="store_true", help="Generate config but don't start training"
    )

    return parser.parse_args()


def generate_model_config(
    model_size: str = "custom",
    hidden_size: Optional[int] = None,
    intermediate_size: Optional[int] = None,
    num_hidden_layers: Optional[int] = None,
    num_attention_heads: Optional[int] = None,
    num_key_value_heads: Optional[int] = None,
    vocab_size: Optional[int] = None,
    max_position_embeddings: int = 4096,
) -> LlamaConfig:
    """Generate a model configuration based on predefined sizes or custom parameters."""
    config_params = MODEL_SIZES.get(model_size, MODEL_SIZES["custom"])
    config_params = {
        "num_hidden_layers": config_params[0],
        "hidden_size": config_params[1],
        "num_attention_heads": config_params[2],
        "num_key_value_heads": config_params[3],
        "intermediate_size": config_params[4],
    }

    if hidden_size is not None:
        config_params["hidden_size"] = hidden_size
    if intermediate_size is not None:
        config_params["intermediate_size"] = intermediate_size
    if num_hidden_layers is not None:
        config_params["num_hidden_layers"] = num_hidden_layers
    if num_attention_heads is not None:
        config_params["num_attention_heads"] = num_attention_heads
    if num_key_value_heads is not None:
        config_params["num_key_value_heads"] = num_key_value_heads
    if vocab_size is not None:
        config_params["vocab_size"] = vocab_size

    model_config = LlamaConfig(
        bos_token_id=1,
        eos_token_id=2,
        hidden_act="silu",
        initializer_range=0.02,
        max_position_embeddings=max_position_embeddings,
        pretraining_tp=1,
        rms_norm_eps=1e-05,
        rope_scaling=None,
        tie_word_embeddings=False,
        use_cache=True,
        **config_params,
    )
    return model_config


def create_nanotron_config(args) -> Config:
    """Create a Nanotron configuration object based on the provided arguments."""
    model_config = generate_model_config(
        model_size=args.model,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        num_key_value_heads=args.num_kv_heads,
        vocab_size=args.vocab_size,
        max_position_embeddings=args.seq,
    )

    num_params = human_format(
        model_config.vocab_size * model_config.hidden_size * 2
        + model_config.num_hidden_layers
        * (
            3 * model_config.hidden_size * model_config.intermediate_size
            + 4 * model_config.hidden_size * model_config.hidden_size
        )
    ).replace(".", "p")

    parallelism = ParallelismArgs(
        dp=args.dp,
        pp=args.pp,
        tp=args.tp,
        context_parallel_size=args.cp,
        expert_parallel_size=args.ep,
        pp_engine="1f1b",
        tp_mode="REDUCE_SCATTER",
        tp_linear_async_communication=True,
        recompute_layer=False,
    )

    tokens = TokensArgs(
        sequence_length=args.seq,
        train_steps=args.steps,
        micro_batch_size=args.mbs,
        batch_accumulation_per_replica=args.acc,
    )

    lr_scheduler = LRSchedulerArgs(
        learning_rate=args.learning_rate,
        lr_warmup_steps=args.warmup_steps,
        lr_warmup_style="linear",
        lr_decay_style="cosine",
        min_decay_lr=args.min_lr,
    )

    optimizer = OptimizerArgs(
        zero_stage=args.zero,
        weight_decay=args.weight_decay,
        clip_grad=args.grad_clip,
        accumulate_grad_in_fp32=True,
        learning_rate_scheduler=lr_scheduler,
        optimizer_factory=AdamWOptimizerArgs(
            adam_eps=1e-08,
            adam_beta1=0.9,
            adam_beta2=0.95,
            torch_adam_is_fused=True,
        ),
    )

    data_stages = [
        DatasetStageArgs(
            name="Stable Training Stage",
            start_training_step=1,
            data=DataArgs(
                dataset=NanosetDatasetsArgs(
                    dataset_folder="/fsx/loubna/tokenized_for_exps/mcf-dataset",
                ),
                seed=args.seed,
            ),
        ),
    ]

    os.makedirs(args.checkpoints_path, exist_ok=True)
    checkpoints = CheckpointsArgs(
        checkpoints_path=os.path.join(args.checkpoints_path, args.run),
        checkpoint_interval=args.checkpoint_interval,
        save_initial_state=args.save_initial_state,
    )

    config = Config(
        general=GeneralArgs(
            project=args.project,
            run=args.run,
            seed=args.seed,
            ignore_sanity_checks=args.no_sanity,
        ),
        checkpoints=checkpoints,
        parallelism=parallelism,
        model=ModelArgs(init_method=RandomInit(std=0.025), model_config=model_config),
        tokenizer=TokenizerArgs(args.tokenizer),
        optimizer=optimizer,
        logging=LoggingArgs(log_level=args.log_lvl, log_level_replica=args.log_lvl, iteration_step_info_interval=1),
        tokens=tokens,
        data_stages=data_stages,
        profiler=ProfilerArgs(profiler_export_path=args.profiler_export_path)
        if args.profiler_export_path is not None
        else None,
    )

    return config


def train_loop_per_worker(config: dict):
    """
    Each Ray worker executes this training function.

    Execution order:
    1. Ray sets RANK/LOCAL_RANK/WORLD_SIZE/LOCAL_WORLD_SIZE/NODE_RANK
    2. Ray initializes torch.distributed (world_pg)
    3. This function runs with all env vars ready

    Data loading compatibility:
    - HuggingFace Dataset: DistributedSampler + dp_rank from dp_pg
    - Streaming IterableDataset: .shard(dp_size, dp_rank)
    - TokenizedBytes: MegatronPretrainingSampler + dp_pg.rank()
    All three paths use parallel_context.dp_pg.rank(), independent of Ray worker allocation.
    """
    import torch
    import torch.distributed as dist
    from nanotron import logging

    logger = logging.get_logger(__name__)

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    node_rank = int(os.environ.get("NODE_RANK", "0"))

    log_rank = lambda msg: logger.info(f"[Rank {rank}] {msg}")
    log_rank(
        f"Ray Train worker: rank={rank}, local_rank={local_rank}, "
        f"world_size={world_size}, node_rank={node_rank}"
    )

    assert dist.is_initialized(), (
        "torch.distributed should be initialized by Ray Train. "
        "Check Ray Train version compatibility."
    )
    log_rank(f"torch.distributed backend: {dist.get_backend()}")

    assert torch.cuda.is_available(), "CUDA must be available for training"

    config_file = config["config_file"]
    log_rank(f"Loading config from: {config_file}")

    from nanotron.trainer import DistributedTrainer
    trainer = DistributedTrainer(config_file)

    log_rank(
        f"DistributedTrainer created: "
        f"TP={trainer.parallel_context.tensor_parallel_size}, "
        f"PP={trainer.parallel_context.pipeline_parallel_size}, "
        f"DP={trainer.parallel_context.data_parallel_size}, "
        f"local_world_size={trainer.parallel_context.local_world_size}"
    )

    from run_train import get_dataloader
    dataloader = get_dataloader(trainer)

    dp_rank = trainer.parallel_context.dp_pg.rank()
    dp_size = trainer.parallel_context.dp_pg.size()
    log_rank(f"DataLoader created: dp_rank={dp_rank}, dp_size={dp_size}")

    log_rank("Starting training...")
    trainer.train(dataloader)
    log_rank("Training completed!")


def build_ray_trainer(args, config_path: str) -> TorchTrainer:
    """Build a Ray TorchTrainer with the given configuration."""
    total_workers = args.num_nodes * args.gpus_per_node

    if args.verify_parallelism:
        from nanotron.config import get_config_from_file
        cfg = get_config_from_file(config_path)
        expected = (
            cfg.parallelism.tp
            * cfg.parallelism.pp
            * cfg.parallelism.dp
            * cfg.parallelism.context_parallel_size
            * cfg.parallelism.expert_parallel_size
        )
        if expected != total_workers:
            raise ValueError(
                f"Worker count mismatch: "
                f"num_nodes({args.num_nodes}) * gpus_per_node({args.gpus_per_node}) "
                f"= {total_workers} != expected TP*PP*DP*CP*EP = {expected}"
            )
        print(f"[Ray Launcher] Parallelism verification passed: {expected} ranks")

    scaling_config = ScalingConfig(
        num_workers=total_workers,
        num_nodes=args.num_nodes,
        use_gpu=True,
        resources_per_worker={"CPU": 8, "GPU": 1},
    )

    run_config = RunConfig(
        name=args.run,
        storage_path=args.storage_path,
        checkpoint_config=CheckpointConfig(
            num_to_keep=args.num_keep_checkpoints,
            checkpoint_frequency=args.checkpoint_interval,
        ),
        failure_config=FailureConfig(max_failures=args.max_failures),
    )

    torch_config = TorchConfig(
        backend="nccl",
        nccl_config={
            "NCCL_DEBUG": os.getenv("NCCL_DEBUG", "WARN"),
        },
    )

    trainer = TorchTrainer(
        train_loop_per_worker=train_loop_per_worker,
        train_loop_config={
            "config_file": config_path,
        },
        scaling_config=scaling_config,
        run_config=run_config,
        torch_config=torch_config,
    )

    return trainer


def main():
    args = parse_args()

    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"

    ray.init(address="auto", ignore_reinit_error=True)

    print(f"{'=' * 70}")
    print(f"Nanotron + Ray Train Training")
    print(f"{'=' * 70}")
    print(f"Run name: {args.run}")
    print(f"Num nodes: {args.num_nodes}")
    print(f"GPUs per node: {args.gpus_per_node}")
    print(f"Total workers: {args.num_nodes * args.gpus_per_node}")
    print(f"Ray cluster resources: {ray.cluster_resources()}")
    print(f"{'=' * 70}")

    if args.config is None:
        config = create_nanotron_config(args)
        dp, pp, tp, cp, ep = args.dp, args.pp, args.tp, args.cp, args.ep
    else:
        from nanotron.config import get_config_from_file

        print(f"Loading config from {args.config}")
        config = get_config_from_file(args.config)
        dp = config.parallelism.dp
        pp = config.parallelism.pp
        tp = config.parallelism.tp
        cp = config.parallelism.context_parallel_size
        ep = config.parallelism.expert_parallel_size

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    config_dir = os.path.join("logs", "configs", args.run.replace(" ", "_"))
    os.makedirs(config_dir, exist_ok=True)
    config_path = os.path.join(config_dir, f"{timestamp}-{args.run}.yaml")
    config.save_as_yaml(config_path)
    print(f"Config saved to: {config_path}")
    config.print_config_details()

    if args.dry_run:
        print("[Ray Launcher] Dry run - skipping training launch")
        ray.shutdown()
        return 0

    trainer = build_ray_trainer(args, config_path)

    print(f"Starting Ray Train training...")
    result = trainer.fit()

    print(f"{'=' * 70}")
    print(f"Training completed!")
    print(f"Result: {result}")
    print(f"{'=' * 70}")

    ray.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
