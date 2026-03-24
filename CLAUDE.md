# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Nanotron is a lightweight LLM pretraining framework (v0.4) developed by Hugging Face. It provides simple and flexible APIs for model pretraining, supporting 3D parallelism (TP+PP+DP), MoE, ZeRO-1 and other features.

### Environment Variables
- `CUDA_DEVICE_MAX_CONNECTIONS=1`: Required for TP
- `NANOTRON_BENCHMARK=1`: Enable benchmark mode
- `WANDB_MODE=disabled`: Disable WandB

### Environment
Use UV environment manager.
```bash
uv sync
uv run xxxx
```

## Core Architecture

### Entry Points
- `run_train.py`: Main training entry point
- `run_generate.py`: Generation/inference entry point
- `run_evals.py`: Evaluation entry point
- `slurm_launcher.py`: Slurm multi-node launcher

### Core Modules (`src/nanotron/`)
- `trainer.py`: Core `DistributedTrainer` class, manages training loop, parallel context, model initialization
- `parallel/context.py`: `ParallelContext` — manages TP/PP/DP/EP/CP process groups
- `parallel/tensor_parallel/`: Tensor parallel implementation
- `parallel/pipeline_parallel/`: Pipeline parallel implementation
- `models/`: Model implementations (Llama, Qwen, StarCoder2), base class `NanotronModel`
- `config/`: Configuration management (YAML/dataclass)
- `data/`: Data loading and preprocessing
- `serialize/`: Checkpoint serialization
- `optim/`: Optimizer and gradient clipping

### Parallelism Strategy
- **Tensor Parallel (TP)**: `TensorParallelColumnLinear` (column split) + `TensorParallelRowLinear` (row split)
- **Pipeline Parallel (PP)**: `PipelineBlock` + `PipelineEngine`, supports AFAB and 1F1B scheduling
- **Data Parallel (DP)**: PyTorch DDP + ZeRO-1
- **Expert Parallel (EP)**: MoE expert parallelism

### Key Patterns
- `TensorPointer`: Lazy-loading pointer for referencing tensors across pipeline stages
- `PipelineBlock`: Pipeline stage wrapper with `module_input_keys` and `module_output_keys`
- `NanotronParameter`: Parameter with shard/bind metadata

### Configuration Structure
```
Config:
├── general: GeneralArgs          # Project name, run name, seed
├── checkpoints: CheckpointsArgs # Save path, interval
├── parallelism: ParallelismArgs # DP/TP/PP configuration
├── model: ModelArgs             # Model configuration (LlamaConfig, etc.)
├── tokenizer: TokenizerArgs      # Tokenizer
├── optimizer: OptimizerArgs      # Optimizer configuration
├── logging: LoggingArgs          # Logging configuration
├── tokens: TokensArgs           # Sequence length, training steps
├── data_stages: List[DatasetStageArgs]  # Multi-stage data configuration
└── profiler: ProfilerArgs       # Performance profiling
```

### Cursor Rules
The project has 6 Cursor AI rule files under `.cursor/rules/`, containing detailed development guidelines:
- `project-overview.mdc`: Project structure overview
- `philosophy.mdc`: Development philosophy and workflow
- `performance-optimization.mdc`: Performance optimization (tensor shape annotations, minimizing data movement)
- `tensor-parallelism.mdc`: TP component reference implementation patterns
- `pipeline-parallelism.mdc`: PP best practices and communication optimization
- `troubleshooting.mdc`: Common issues troubleshooting

### Principle
Use English in code writing and comments (everywhere in the codebase). Only use Chinese in discussion with user.
