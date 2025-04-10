"""Utility Functions for Dataset Handling and Tokenizer Configuration.

This module provides various utility functions and classes to facilitate dataset
handling, tokenizer configuration, and data loading for machine learning workflows. It
includes functions for building datasets, validating tokenizer configurations, and
creating data loaders. Additionally, it defines custom dataset classes for specific use
cases.
"""

from tempfile import TemporaryDirectory

import datasets as hf_datasets
from llmfoundry.data import ConcatTokensDataset, NoConcatDataset
from tokenizers.processors import TemplateProcessing
from torch.utils.data import DataLoader, Dataset, IterableDataset
from transformers import PreTrainedTokenizerBase

from photon.dataset.constants.dataset_constants_types import ConcatMode
from photon.utils import get_n_cpu_cores

ONE_GRAM_JSON_FILENAME = "1_gram.json"
TOKENIZER_FOLDER_NAME = "tokenizer"
CLIENT_FOLDER_PREFIX = "client_"
TOKENIZER_CONFIG_BASENAME = "tokenizer/tokenizer_config.json"


def check_tokenizer_config(
    tokenizer: PreTrainedTokenizerBase,
    bos_text: str = "",
    eos_text: str = "",
) -> None:
    """Validate the configuration of a tokenizer for BOS and EOS token insertion.

    This function checks if the provided tokenizer is an instance of
    PreTrainedTokenizerBase and validates whether it correctly inserts BOS (Beginning
    of Sequence) and EOS (End of Sequence) tokens. If both `bos_text` and `eos_text` are
    provided, the function ensures that the tokenizer inserts the BOS and EOS tokens
    correctly. If the tokenizer does not insert these tokens, a workaround is applied
    to enforce the insertion of BOS and EOS tokens. If the workaround fails, a
    ValueError is raised with an appropriate error message.

    Parameters
    ----------
    tokenizer : PreTrainedTokenizerBase
        The tokenizer to be validated. It must be of type PreTrainedTokenizerBase.
    bos_text : str, optional
        The text representing the BOS token. Default is an empty string.
    eos_text : str, optional
        The text representing the EOS token. Default is an empty string.

    Example
    -------
    >>> from transformers import AutoTokenizer
    >>> tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
    >>> check_tokenizer_config(tokenizer, bos_text="def check_tokenizer_config(
            tokenizer: PreTrainedTokenizerBase, bos_text: str = "", eos_text: str = ""
        )

    Raises
    ------
    ValueError
        If both `bos_text` and `eos_text` are provided and the tokenizer does not insert
        BOS and EOS tokens correctly, even after applying the workaround.

    """
    if bos_text and eos_text:
        test_tokens = tokenizer("test")
        if (
            test_tokens["input_ids"][0] != tokenizer.bos_token_id  # type: ignore[reportIndexIssue]
            and test_tokens["input_ids"][-1] != tokenizer.eos_token_id  # type: ignore[reportIndexIssue]
        ):
            # NOTE: This is a workaround for tokenizers that do not insert BOS and EOS
            # such as the GPT2Tokenizer(Fast). HF thinks this is the correct behavior
            # even when asking to add special tokens explicitly despite people have
            # opened issues since 2020. We won't fight these and work around it.
            tokenizer._tokenizer.post_processor = TemplateProcessing(  # type: ignore[reportAttributeAccessIssue]  # noqa: SLF001
                single=tokenizer.bos_token + " $A " + tokenizer.eos_token,
                # NOTE: We don't add any special token between $A and $B since the
                # underlying dataset (`llmfoundry.data.ConcatTokensDataset`) already
                # addresses concatenation of distinct sequences.
                pair=tokenizer.bos_token + " $A $B " + tokenizer.eos_token,
                special_tokens=[
                    (tokenizer.eos_token, tokenizer.eos_token_id),
                    (tokenizer.bos_token, tokenizer.bos_token_id),
                ],
            )
            test_tokens = tokenizer("test")
            if (
                test_tokens["input_ids"][0] != tokenizer.bos_token_id  # type: ignore[reportIndexIssue]
                and test_tokens["input_ids"][-1] != tokenizer.eos_token_id  # type: ignore[reportIndexIssue]
            ):
                tok_error_msg = "This tokenizer does not insert an EOS nor BOS token. "
                tok_error_msg += (
                    "Concatenating with this tokenizer will result in sequences being "
                )
                tok_error_msg += "attached without a separating token."
                "Please use another tokenizer, "
                tok_error_msg += (
                    "such as facebook/opt-125m, or specify EOS/BOS text with e.g. "
                )
                tok_error_msg += "--bos_text=<|endoftext|>."
                raise ValueError(tok_error_msg)


def build_hf_dataset(  # noqa: PLR0913, PLR0917
    path: str,
    split: str,
    mode: ConcatMode,
    temp_dir: TemporaryDirectory,
    max_length: int | None = None,
    bos_text: str = "",
    eos_text: str = "",
    tokenizer: PreTrainedTokenizerBase | None = None,
    name: str | None = None,
    *,
    no_wrap: bool = False,
) -> IterableDataset:
    """Build a Hugging Face dataset with optional token concatenation.

    This function loads a dataset from the Hugging Face Hub and wraps it in a dataset
    class based on the specified concatenation mode. If no concatenation is required,
    the dataset is wrapped in a NoConcatDataset. If concatenation is required, the
    dataset is wrapped in a ConcatTokensDataset, and the tokenizer configuration is
    validated.

    Parameters
    ----------
    path : str
        The path or name of the dataset to load from the Hugging Face Hub.
    split : str
        The dataset split to load (e.g., "train", "validation").
    mode : ConcatMode
        The concatenation mode to use. Must be an instance of ConcatMode.
    temp_dir : TemporaryDirectory
        A temporary directory for caching the dataset.
    max_length : int, optional
        The maximum length of concatenated tokens, required when concatenating.
    bos_text : str, optional
        The text representing the BOS token. Default is an empty string.
    eos_text : str, optional
        The text representing the EOS token. Default is an empty string.
    no_wrap : bool, optional
        Whether to disable wrapping of tokens. Default is False.
    tokenizer : PreTrainedTokenizerBase, optional
        The tokenizer to use for token concatenation, required when concatenating.
    name : str, optional
        The name of the dataset configuration. Default is None.

    Returns
    -------
    IterableDataset
        The wrapped dataset, either as a NoConcatDataset or ConcatTokensDataset.

    Example
    -------
    >>> from transformers import AutoTokenizer
    >>> from tempfile import TemporaryDirectory
    >>> tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
    >>> temp_dir = TemporaryDirectory()
    >>> dataset = build_hf_dataset(
    ...     path="allenai/c4",
    ...     split="train",
    ...     mode=ConcatMode.CONCAT,
    ...     temp_dir=temp_dir,
    ...     max_length=512,
    ...     bos_text="<s>",
    ...     eos_text="</s>",
    ...     tokenizer=tokenizer,
    ...     name="en"
    ... )

    """
    # Check whether we are int the special case of SmolLM Corpus - Python Edu
    is_special = path == "HuggingFaceTB/smollm-corpus" and name == "python-edu"
    # Load dataset from HF Hub
    if not is_special:
        hf_dataset = hf_datasets.load_dataset(  # type: ignore[attr-defined]
            path=path,  # Path or name of the dataset
            name=name,  # Defining the name of the dataset configuration.
            data_dir=None,  # We are getting it from the Hub
            split=split,  # Which split to use
            streaming=True,
            cache_dir=temp_dir.name,
            keep_in_memory=False,
            save_infos=False,
            trust_remote_code=True,
        )
    else:
        hf_dataset = hf_datasets.load_from_disk(  # type: ignore[attr-defined]
            dataset_path="/nfs-share-old/datasets_repo/python_edu",
        )
    # Wrap the dataset depending on the concatenation mode
    if mode == ConcatMode.NO_CONCAT:
        # Wrap the dataset in a NoConcatDataset
        dataset = NoConcatDataset(hf_dataset)  # type: ignore[reportArgumentType]
    else:
        assert tokenizer is not None, "Tokenizer must be provided for concatenation."
        assert max_length is not None, "Max length must be provided for concatenation."
        # Check that the tokenizer is properly configured
        check_tokenizer_config(tokenizer, bos_text, eos_text)
        # Wrap the dataset in a ConcatTokensDataset
        dataset = ConcatTokensDataset(
            hf_dataset=hf_dataset,  # type: ignore[reportArgumentType]
            tokenizer=tokenizer,
            max_length=max_length,
            bos_text=bos_text,
            eos_text=eos_text,
            no_wrap=no_wrap,
        )
    return dataset


def build_dataloader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int | None,
) -> DataLoader:
    """Build a DataLoader for a given dataset with batch size and number of workers.

    This function creates a DataLoader for the provided dataset, configuring the number
    of workers and prefetch factor based on the batch size. If `num_workers` is None,
    the function determines the number of CPU cores available and uses that value.

    Parameters
    ----------
    dataset : Dataset
        The dataset to load data from.
    batch_size : int
        The number of samples per batch to load.
    num_workers : int | None
        The number of worker processes to use for data loading. If None, the number of
        CPU cores is used.

    Returns
    -------
    DataLoader
        A DataLoader instance configured with the specified dataset, batch size, and
        number of workers.

    Notes
    -----
    - Multiple workers are only supported on Linux machines.
    - The prefetch factor is configured based on the number of workers and batch size.
      If using multiple workers, each worker is configured to prefetch as many samples
      as it can, up to the aggregate device batch size. If not using workers, a default
      prefetch factor of 2 is used.

    Example
    -------
    >>> from torch.utils.data import Dataset, DataLoader
    >>> class MyDataset(Dataset):
    ...     def __len__(self):
    ...         return 100
    ...     def __getitem__(self, idx):
    ...         return idx
    >>> dataset = MyDataset()
    >>> dataloader = build_dataloader(dataset, batch_size=10, num_workers=4)
    >>> for batch in dataloader:
    ...     print(batch)

    """
    if num_workers is None:
        # Multiple workers is only supported on linux machines
        num_workers = get_n_cpu_cores()
        assert num_workers is not None, "Could not determine number of CPU cores."

    # If using multiple workers, configure each worker to prefetch as many samples as
    # it can, up to the aggregate device batch size
    # If not using workers, the torch DataLoader expects the default value for
    # prefetch_factor, which non-intuitively must be 2.
    prefetch_factor = max(1, 2 * batch_size // num_workers) if num_workers > 0 else 2

    return DataLoader(
        dataset=dataset,
        sampler=None,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )
