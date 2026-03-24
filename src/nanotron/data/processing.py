import logging
import warnings
from typing import Dict, List, Optional, Union
import numpy as np

try:
    from datasets import (
        Dataset,
        DatasetDict,
        Features,
        IterableDataset,
        Sequence,
        Value,
        concatenate_datasets,
        load_dataset,
    )
except ImportError:
    warnings.warn("Datasets not installed, you'll be unable to use these dataset processing functions.")

try:
    import pyarrow.dataset
except ImportError:
    pyarrow = None

# Import SFT processing functions for backward compatibility


def clm_process(
    raw_dataset: "Dataset",
    tokenizer,
    text_column_name: str,
    dataset_processing_num_proc_per_process: int,
    dataset_overwrite_cache: bool,
    sequence_length: int,
):
    """
    Concatenate all texts from raw_dataset and generate chunks of `sequence_length + 1`,
    where chunks overlap by a single token.

    Args:
        raw_dataset: Dataset containing raw text
        tokenizer: HuggingFace tokenizer
        text_column_name: Name of the column containing text data
        dataset_processing_num_proc_per_process: Number of processes for parallelization
        dataset_overwrite_cache: Whether to overwrite the cache
        sequence_length: Maximum sequence length

    Returns:
        Processed dataset with tokenized sequences
    """
    # Adapted from https://github.com/huggingface/transformers/blob/47e1676255e5dd86b9541f734cd4f4bdcbb50f4a/examples/pytorch/language-modeling/run_clm.py#L391-L439

    def group_texts(examples: Dict[str, List[np.ndarray]]) -> Dict[str, List[np.ndarray]]:
        # Concatenate all texts.
        concatenated_examples = {k: np.concatenate(v) for k, v in examples.items()}
        total_length = len(concatenated_examples[next(iter(examples.keys()))])
        # WARNING: We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= sequence_length + 1:
            total_length = ((total_length - 1) // sequence_length) * sequence_length + 1
        # Split by chunks of sequence_length.
        result = {
            k: [
                t[i : i + sequence_length + 1] for i in range(0, total_length - (sequence_length + 1), sequence_length)
            ]
            for k, t in concatenated_examples.items()
        }
        return result

    def _tokenize_and_group_texts(texts: List[str]) -> Dict[str, List[np.ndarray]]:
        tokenized_batch = tokenizer.batch_encode_plus(texts, return_attention_mask=False, return_token_type_ids=False)
        tokenized_batch = {k: [np.array(tokenized_texts) for tokenized_texts in v] for k, v in tokenized_batch.items()}
        return group_texts(tokenized_batch)

    train_dataset = raw_dataset.map(
        _tokenize_and_group_texts,
        input_columns=text_column_name,
        remove_columns=raw_dataset.column_names,
        features=Features({"input_ids": Sequence(feature=Value(dtype="int64"), length=sequence_length + 1)}),
        batched=True,
        num_proc=dataset_processing_num_proc_per_process,
        load_from_cache_file=not dataset_overwrite_cache,
        desc=f"Grouping texts in chunks of {sequence_length+1}",
    )
    return train_dataset


def clm_process_streaming(
    raw_dataset: "IterableDataset",
    tokenizer,
    text_column_name: str,
    sequence_length: int,
):
    """
    Streaming version of clm_process for HuggingFace IterableDataset.
    Tokenizes and groups texts into chunks of `sequence_length + 1` with 1-token overlap.
    Processes in batches via IterableDataset.map() for efficiency.

    Args:
        raw_dataset: IterableDataset containing raw text
        tokenizer: HuggingFace tokenizer
        text_column_name: Name of the column containing text data
        sequence_length: Maximum sequence length

    Returns:
        IterableDataset with tokenized sequences (input_ids column)
    """

    def _tokenize_and_group(examples) -> Dict:
        # HuggingFace IterableDataset.map(batched=True) with input_columns
        # passes a list of strings directly when input_columns specifies a single column,
        # or a list of dicts when input_columns is not specified
        if isinstance(examples, list) and len(examples) > 0 and isinstance(examples[0], str):
            # Case 1: List of strings - use directly
            texts = examples
        elif isinstance(examples, list):
            # Case 2: List of dicts - extract text column
            texts = [ex[text_column_name] for ex in examples]
        else:
            # Case 3: Dict of lists (standard non-streaming format)
            texts = examples[text_column_name]
        tokenized = tokenizer(
            texts,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        tokens_list = tokenized["input_ids"]

        result_input_ids = []
        for tokens in tokens_list:
            # tokens may be a list or tensor; normalize to list
            if hasattr(tokens, "tolist"):
                buffer = tokens.tolist()
            else:
                buffer = list(tokens)

            while len(buffer) >= sequence_length + 1:
                result_input_ids.append(np.array(buffer[: sequence_length + 1], dtype=np.int64))
                buffer = buffer[sequence_length:]

        return {"input_ids": result_input_ids}

    return raw_dataset.map(
        _tokenize_and_group,
        input_columns=text_column_name,
        remove_columns=raw_dataset.column_names,
        batched=True,
        batch_size=1000,
    )


def get_datasets(
    hf_dataset_or_datasets: Union[dict, str],
    hf_dataset_config_name: str,
    splits: Optional[Union[List[str], str]] = ["train", "test"],
    streaming: bool = False,
) -> "DatasetDict":
    """
    Function to load dataset directly from arguments.

    Args:
        hf_dataset_or_datasets: Dict or string defining datasets to load. When using a dict
                                with probabilities all equal to 1, datasets are concatenated
                                instead of sampled.
        hf_dataset_config_name: Configuration name for the dataset
        splits: Section of the dataset to load, defaults to ["train", "test"]
            Can be a single string or a list of splits to load.
        streaming: If True, use streaming mode to avoid downloading the full dataset.

    Returns:
        DatasetDict: DatasetDict object containing the loaded datasets
    """
    if isinstance(splits, str):
        splits = [splits]

    if isinstance(hf_dataset_or_datasets, dict):
        if streaming:
            raise ValueError(
                "Dataset mixing via dict is not supported in streaming mode. "
                "Please specify a single dataset path or disable streaming."
            )
        # Structure of the config to read the datasets and their mix
        # datasets_mixer:
        #     - 'dataset1': 0.5
        #     - 'dataset2': 0.3
        #     - 'dataset3': 0.2
        raw_datasets = _get_dataset_mix(hf_dataset_or_datasets, splits=splits)
    elif isinstance(hf_dataset_or_datasets, str):
        # e.g. Dataset = "HuggingFaceH4/testing_alpaca_small"
        # Note this returns things other than just train/test, which may not be intended
        raw_datasets = DatasetDict()

        fragment_scan_options = None
        if streaming and pyarrow is not None:
            fragment_scan_options = pyarrow.dataset.ParquetFragmentScanOptions(
                cache_options=pyarrow.CacheOptions(
                    prefetch_limit=5,
                    range_size_limit=64 << 20,  # 64MiB
                ),
            )

        for split in splits:
            raw_datasets[split] = load_dataset(
                hf_dataset_or_datasets,
                hf_dataset_config_name,
                split=split,
                streaming=streaming,
                fragment_scan_options=fragment_scan_options,
            )
    else:
        raise ValueError(f"hf_dataset_or_datasets must be a dict or string but is {type(hf_dataset_or_datasets)}")

    return raw_datasets


def _get_dataset_mix(dataset_dict: dict, splits: List[str] = None, seed=42) -> "DatasetDict":
    """
    Helper function to load dataset mix from dict configuration.

    Args:
        dataset_dict: Dictionary containing the dataset names and their training proportions.
                     By default, all test proportions are 1.
        splits: Section of the dataset to load, defaults to ["train", "test"]
        seed: Random seed for shuffling datasets

    Returns:
        DatasetDict containing the mixed datasets
    """
    raw_datasets = DatasetDict()
    raw_train_datasets = []
    raw_test_datasets = []
    fracs = []
    for ds, frac in dataset_dict.items():
        if frac < 0:
            raise ValueError(f"Dataset fraction for dataset {ds} is negative. (= {frac})")

        fracs.append(frac)
        for split in splits:
            if "train" in split:
                raw_train_datasets.append(
                    load_dataset(
                        ds,
                        split=split,
                    )
                )
            elif "test" in split:
                raw_test_datasets.append(
                    load_dataset(
                        ds,
                        split=split,
                    )
                )
            else:
                raise ValueError(f"Split type {split} not recognized as one of test or train.")

    if len(raw_train_datasets) > 0:
        train_subsets = []
        for dataset, frac in zip(raw_train_datasets, fracs):
            train_subset = dataset.select(range(int(frac * len(dataset))))
            train_subsets.append(train_subset)
        raw_datasets["train"] = concatenate_datasets(train_subsets).shuffle(seed=seed)

    # No subsampling for test datasets to enable fair comparison across models
    if len(raw_test_datasets) > 0:
        raw_datasets["test"] = concatenate_datasets(raw_test_datasets).shuffle(seed=seed)

    if len(raw_datasets) == 0:
        raise ValueError(
            f"Dataset {dataset_dict} not recognized with split {splits}. Check the dataset has been correctly formatted."
        )

    return raw_datasets
