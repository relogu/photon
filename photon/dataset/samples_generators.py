"""Sample Generators for Datasets.

This module provides functions to generate samples from various types of datasets,
including iterators, DataLoaders, and streaming text datasets. It includes functionality
for optional truncation of the number of samples and re-tokenization of streaming text
datasets.

"""

from collections.abc import Iterable, Iterator
from typing import cast

import numpy as np
import torch
from llmfoundry.data.text_data import StreamingTextDataset
from numpy.typing import NDArray
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase

from photon.dataset.dataset_types import TokenizersCouple


def generate_samples_from_dataloader(
    loader: DataLoader,
    truncate_num_samples: int | None = None,
) -> Iterable[dict[str, np.ndarray]]:
    """Build a Generator over samples of a dataloader.

    Notably, the DataLoader can output samples of type bytes or numpy.array[np.int32]
    depending on whether we are using llmfoundry.data.NoConcatDataset or
    llmfoundry.data.ConcatTokensDataset, respectively.

    Args:
       loader (DataLoader):
        A dataloader emitting batches like
        {key: [sample0, sample1, sample2, ...]}
       truncate_num_samples (Optional[int]): An optional number of samples to stop at.

    Yields:
    ------
        Sample dicts.

    """
    if truncate_num_samples is None:
        truncate_num_samples = -1
    for batch in loader:
        # Get the keys of the batch
        keys = list(batch.keys())
        assert len(keys) == 1, "The batch should have only one key."
        # Get the current batch size from the first key
        current_bs = len(batch[keys[0]])
        # Loop over the current batch size
        for idx in range(current_bs):
            if truncate_num_samples == 0:
                return
            truncate_num_samples -= 1
            yield {
                k: v[idx].numpy() if isinstance(v[idx], torch.Tensor) else v[idx]
                for k, v in batch.items()
            }


def stream_and_untokenize(
    loader: DataLoader,
    tokenizer: PreTrainedTokenizerBase,
    truncate_num_batches: int | None = None,
) -> Iterator[list[str]]:
    """Generate samples from a StreamingTextDataset with optional truncation.

    This function takes a DataLoader that yields batches of tokenized samples and yields
    individual decoded samples from these batches. If the `truncate_num_batches`
    parameter is provided, the function will yield up to that many batches and then
    stop. If `truncate_num_batches` is None, all batches from the DataLoader will be
    yielded.

    Parameters
    ----------
    loader : DataLoader
        A DataLoader that yields batches of tokenized samples.
    tokenizer : PreTrainedTokenizerBase
        A tokenizer to decode the tokenized samples.
    truncate_num_batches : int | None, optional
        The maximum number of batches to yield. If None, all batches from the DataLoader
        will be yielded. Default is None.

    Yields
    ------
    Iterator[list[str]]
        An iterator that yields individual decoded samples from the DataLoader, up to
        the specified number of batches.

    Example
    -------
    >>> from torch.utils.data import DataLoader
    >>> from transformers import PreTrainedTokenizerFast
    >>> data = [[101, 102], [103, 104]]
    >>> loader = DataLoader(data, batch_size=2)
    >>> tokenizer = PreTrainedTokenizerFast.from_pretrained("bert-base-uncased")
    >>> for sample in stream_and_untokenize(
    ...     loader, tokenizer, truncate_num_batches=3
    ... ):
    ...     print(sample)
    [CLS] [SEP]
    [UNK] [UNK]

    """
    msg = (
        "Underlying dataset must be a StreamingTextDataset not"
        f" a {type(loader.dataset)}"
    )
    assert isinstance(loader.dataset, StreamingTextDataset), msg
    if truncate_num_batches is None:
        truncate_num_batches = -1
    for batch in loader:
        if truncate_num_batches == 0:
            return
        truncate_num_batches -= 1
        assert isinstance(
            batch,
            torch.Tensor,
        ), "Sample must be a torch.Tensor, i.e., pre-tokenized."
        yield tokenizer.batch_decode(
            batch,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )


def generate_samples_retokenized_streaming_text_dataset(
    loader: DataLoader,
    tokenizer_couple: TokenizersCouple,
    max_length: int,
    truncate_num_samples: int | None = None,
    *,
    no_wrap: bool,
) -> Iterator[dict[str, NDArray]]:
    """Re-tokenize samples from a StreamingTextDataset with optional truncation.

    This function takes a DataLoader that yields batches of tokenized samples, decodes
    and re-encodes the samples using specified tokenizers, and yields individual samples
    with concatenated tokens. If the `truncate_num_samples` parameter is provided, the
    function will yield up to that many samples and then stop. If `truncate_num_samples`
    is None, all samples from the DataLoader will be yielded.

    Parameters
    ----------
    loader : DataLoader
        A DataLoader that yields batches of tokenized samples.
    tokenizer_couple : TokenizersCouple
        A couple of tokenizers for encode/decode samples.
    max_length : int
        Maximum length of the concatenated token sequences.
    no_wrap : bool
        Whether to disable wrapping of tokens.
    truncate_num_samples : int | None, optional
        The maximum number of samples to yield. If None, all samples from the DataLoader
        will be yielded. Default is None.

    Yields
    ------
    Iterator[dict[str, NDArray]]
        An iterator that yields dictionaries containing re-tokenized samples with
        concatenated tokens, stored as NumPy arrays.

    Example
    -------
    >>> from torch.utils.data import DataLoader
    >>> from transformers import PreTrainedTokenizerFast
    >>> data = [[101, 102], [103, 104]]
    >>> loader = DataLoader(data, batch_size=2)
    >>> decode_tokenizer = PreTrainedTokenizerFast.from_pretrained("bert-base-uncased")
    >>> encode_tokenizer = PreTrainedTokenizerFast.from_pretrained("gpt2")
    >>> for sample in generate_samples_retokenized_streaming_text_dataset(
    ...     loader,
    ...     decode_tokenizer,
    ...     encode_tokenizer,
    ...     max_length=512,
    ...     no_wrap=False,
    ...     truncate_num_samples=3
    ... ):
    ...     print(sample)
    {'tokens': array([50256, 101, 102, 50256], dtype=int32)}
    {'tokens': array([50256, 103, 104, 50256], dtype=int32)}

    """
    if truncate_num_samples is None:
        truncate_num_samples = -1
    assert tokenizer_couple.encode_tokenizer.bos_token_id is not None, (
        "The encode tokenizer must have a BOS token ID."
    )
    assert tokenizer_couple.encode_tokenizer.eos_token_id is not None, (
        "The encode tokenizer must have a EOS token ID."
    )
    buffer: list[int] = []
    for batch in loader:
        # Loop over the current batch size
        for sample in batch:
            if truncate_num_samples == 0:
                return
            truncate_num_samples -= 1
            decoded_sample = tokenizer_couple.decode_tokenizer.decode(sample)
            encoded = tokenizer_couple.encode_tokenizer(
                decoded_sample,
                truncation=False,
                padding=False,
            )
            iids = cast("list[int]", encoded["input_ids"])
            buffer.extend(
                [
                    tokenizer_couple.encode_tokenizer.bos_token_id,
                    *iids,
                    tokenizer_couple.encode_tokenizer.eos_token_id,
                ],
            )
            while len(buffer) >= max_length:
                concat_sample = buffer[:max_length]
                buffer = buffer[max_length:] if not no_wrap else []
                yield {
                    # convert to ndarray to store in MDS format
                    "tokens": np.asarray(concat_sample, dtype=np.int32),
                }
