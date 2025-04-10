"""Convert Dataset from Hugging Face Hub to MDS Format.

This module provides functionality to download a dataset from the Hugging Face Hub,
tokenize and concatenate the data, and convert it into MDS format. The resulting MDS
files can be saved to a specified remote bucket, with support for multiple clients.
"""

import json
from argparse import ArgumentParser, Namespace
from collections import Counter
from logging import INFO
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from flwr.common.logger import log
from llmfoundry.utils.builders import build_tokenizer
from streaming import MDSWriter
from tqdm.auto import tqdm

from photon.dataset.constants import DATASETS_CONSTANTS
from photon.dataset.constants.dataset_constants_types import (
    ConcatMode,
)
from photon.dataset.samples_generators import generate_samples_from_dataloader
from photon.dataset.utils import (
    CLIENT_FOLDER_PREFIX,
    ONE_GRAM_JSON_FILENAME,
    TOKENIZER_FOLDER_NAME,
    build_dataloader,
    build_hf_dataset,
)


def parse_args() -> Namespace:
    """Parse command-line arguments for the dataset conversion script.

    This function sets up an argument parser to receive various parameters for
    downloading, tokenizing, and converting a dataset from the Hugging Face Hub into
    MDS format. It parses the command-line arguments and returns them as a Namespace
    object.

    Parameters
    ----------
    None

    Returns
    -------
    Namespace
        An object containing the parsed command-line arguments:
        - path (str): Path or name of the dataset.
        - name (str | None): Name of the dataset configuration.
        - splits (list[str] | None): List of dataset splits to process.
        - compression (str): Compression method for the output MDS dataset.
        - concat_tokens (int): Number of tokens to concatenate. Default is 2048.
        - tokenizer (str): Path or name of the tokenizer to use.
        - tokenizer_kwargs (dict): Additional keyword arguments for the tokenizer.
        - bos_text (str): Text representing the Beginning of Sequence token. Default is
            None.
        - eos_text (str): Text representing the End of Sequence token. Default is
            "</s>".
        - pad_text (str): Text representing the pad token. Default is "<s>".
        - no_wrap (bool): Whether to disable wrapping of tokens. Default is False.
        - num_workers (int | None): Number of worker processes to use for data loading.
        - num_clients (int): Number of clients to compose the federated dataset. Default
            is 1.
        - client (int | None): Client id to elaborate.
        - remote_path (str): Name of the remote bucket to upload the files to.

    Example
    -------
    >>> args = parse_args()
    >>> print(args.path)
    >>> print(args.name)
    >>> print(args.splits)
    >>> print(args.compression)
    >>> print(args.concat_tokens)
    >>> print(args.tokenizer)
    >>> print(args.tokenizer_kwargs)
    >>> print(args.bos_text)
    >>> print(args.eos_text)
    >>> print(args.pad_text)
    >>> print(args.no_wrap)
    >>> print(args.num_workers)
    >>> print(args.num_clients)
    >>> print(args.client)
    >>> print(args.remote_path)

    """
    parser = ArgumentParser(
        description=(
            "Convert dataset into MDS format, optionally concatenating and tokenizing."
        ),
    )
    # Parameters for downloading the dataset from HF Hub
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help='E.g. "allenai/c4" or ""',
    )
    parser.add_argument("--name", type=str, default=None, help='E.g. "all" or "en"')
    parser.add_argument(
        "--splits",
        nargs="+",
        default=None,
        help='E.g. "train" or "validation"',
    )
    # Parameters to creating the output MDS dataset
    parser.add_argument("--compression", type=str, default="zstd")
    # Parameters for the tokenization and (potentially) concatenation
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        "--concat_tokens",
        type=int,
        help="Convert text to tokens and concatenate up to this many tokens",
        default=2048,
    )
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--tokenizer_kwargs", type=str, required=False)
    parser.add_argument("--bos_text", type=str, required=False, default=None)
    parser.add_argument("--eos_text", type=str, required=False, default="</s>")
    parser.add_argument("--pad_text", type=str, required=False, default="<s>")
    parser.add_argument("--no_wrap", default=False, action="store_true")
    parser.add_argument("--num_workers", type=int, required=False, default=None)
    # Number of clients to compose the federated dataset (this is done at a
    # dataset/configuration/split level)
    parser.add_argument("--num_clients", type=int, required=False, default=1)
    parser.add_argument("--client", type=int, required=False, default=None)
    # Arguments to use our S3-stored dataset when concatenating tokens
    parser.add_argument(
        "--remote_path",
        type=str,
        required=True,
        help="Name of the remote bucket to upload the files to",
    )

    # Parse arguments
    parsed = parser.parse_args()

    # Parse tokenizer_kwargs
    if parsed.tokenizer_kwargs is not None:
        parsed.tokenizer_kwargs = json.loads(parsed.tokenizer_kwargs)
    else:
        parsed.tokenizer_kwargs = {}

    # Make sure we have needed concat options
    if (
        parsed.concat_tokens is not None
        and isinstance(parsed.concat_tokens, int)
        and parsed.tokenizer is None
    ):
        parser.error("When setting --concat_tokens, you must specify a --tokenizer")

    # Change BOS/EOS/pad to strings if they are None
    if parsed.bos_text is None:
        parsed.bos_text = ""
    if parsed.eos_text is None:
        parsed.eos_text = ""
    if parsed.pad_text is None:
        parsed.pad_text = ""
    # Add BOS/EOS/pad tokens to tokenizer_kwargs
    parsed.tokenizer_kwargs["bos_token"] = parsed.bos_text
    parsed.tokenizer_kwargs["eos_token"] = parsed.eos_text
    parsed.tokenizer_kwargs["pad_token"] = parsed.pad_text

    # Parse splits
    if parsed.splits is not None:
        parsed.splits = set(parsed.splits)
        log(INFO, f"Converting splits: {parsed.splits}")
    return parsed


def main(args: Namespace) -> None:
    """Convert a dataset from the Hugging Face Hub into MDS format.

    This function processes specified splits of a Hugging Face dataset, tokenizes and
    concatenates the data, and converts it into MDS format. The resulting MDS files are
    saved to a specified remote bucket, with support for multiple clients.

    Parameters
    ----------
    args : Namespace
        The arguments for the function, expected to have the following attributes:
        - path (str): Path or name of the dataset.
        - name (str | None): Name of the dataset configuration.
        - splits (set[str]): Set of dataset splits to process.
        - compression (str): Compression method for the output MDS dataset.
        - concat_tokens (int): Number of tokens to concatenate.
        - tokenizer (str): Path or name of the tokenizer to use.
        - tokenizer_kwargs (dict): Additional keyword arguments for the tokenizer.
        - bos_text (str): Text representing the Beginning of Sequence token.
        - eos_text (str): Text representing the End of Sequence token.
        - pad_text (str): Text representing the pad token.
        - no_wrap (bool): Whether to disable wrapping of tokens.
        - num_workers (int | None): Number of worker processes to use for data loading.
        - num_clients (int): Number of clients to compose the federated dataset.
        - client (int | None): Client id to elaborate.
        - remote_path (str): Name of the remote bucket to upload the files to.

    Example
    -------
    >>> from argparse import Namespace
    >>> args = Namespace(
    ...     path="allenai/c4",
    ...     name="en",
    ...     splits={"train", "validation"},
    ...     compression="zstd",
    ...     concat_tokens=2048,
    ...     tokenizer="path/to/tokenizer",
    ...     tokenizer_kwargs={},
    ...     bos_text="<s>",
    ...     pad_text="<s>",
    ...     eos_text="</s>",
    ...     no_wrap=False,
    ...     num_workers=4,
    ...     num_clients=1,
    ...     client=None,
    ...     remote_path="s3://mybucket"
    ... )
    >>> main(args)

    """
    log(INFO, "Arguments received: %s", args)
    torch.multiprocessing.set_sharing_strategy("file_system")
    # Create temporary directory
    temp_dir = TemporaryDirectory()
    # Build tokenizer
    tokenizer = build_tokenizer(args.tokenizer, args.tokenizer_kwargs)
    # We will enforce length because it suppress warnings about sequences too long
    # for the model
    tokenizer.model_max_length = int(1e30)
    # Loop over passed splits
    for split_name in args.splits:
        # Create temporary directory for caching the dataset
        temp_dir = TemporaryDirectory()
        # Create the dataset given the parameters and build a batched dataloader for
        # streaming the HF dataset in batches so that we can actually take advantage of
        # multiprocessing
        # NOTE: We can't know how many samples we will get from the dataset
        loader = build_dataloader(
            dataset=build_hf_dataset(
                path=DATASETS_CONSTANTS[f"{args.path}_{args.name}"]
                .splits[split_name]
                .path,
                name=DATASETS_CONSTANTS[f"{args.path}_{args.name}"]
                .splits[split_name]
                .name,
                split=DATASETS_CONSTANTS[f"{args.path}_{args.name}"]
                .splits[split_name]
                .split,
                mode=ConcatMode.CONCAT_TOKENS,
                max_length=args.concat_tokens,
                bos_text=args.bos_text,
                eos_text=args.eos_text,
                no_wrap=args.no_wrap,
                tokenizer=tokenizer,
                temp_dir=temp_dir,
            ),
            batch_size=512,
            num_workers=args.num_workers,
        )
        # Build a generator that yields samples from the batched dataloader, truncating
        # if needed. In case of tokenized text, we need to count the number of samples
        total_num_samples = 0
        for _ in tqdm(
            generate_samples_from_dataloader(
                loader,
                truncate_num_samples=DATASETS_CONSTANTS[f"{args.path}_{args.name}"]
                .splits[split_name]
                .truncated_samples,
            ),
            desc=f"Counting tokens for {args.path}-{args.name}-{split_name}",
            mininterval=10 * 60,
        ):
            total_num_samples += 1
        # Re-generate samples iterator
        samples = generate_samples_from_dataloader(
            loader,
            truncate_num_samples=DATASETS_CONSTANTS[f"{args.path}_{args.name}"]
            .splits[split_name]
            .truncated_samples,
        )
        log(
            INFO,
            "Number of samples in %s-%s-%s is %s, using tokenizer %s.",
            args.path,
            args.name,
            split_name,
            total_num_samples,
            tokenizer,
        )

        # Estimate the number of samples for the current client
        # NOTE: The last client will get the remainder of the samples
        log(
            INFO,
            "Expected samples per client %s. "
            "A remainder of %s will be appended to the last client.",
            total_num_samples // args.num_clients,
            int(total_num_samples % args.num_clients),
        )

        # Write samples
        log(
            INFO,
            "Converting %s-%s-%s to MDS format...",
            args.path,
            args.name,
            split_name,
        )
        # Loop over the number of clients
        for i in range(args.num_clients):
            # Creating an empty frequency map for the current client
            frequency_map: Counter = Counter()
            # Create temporary directory for the client
            client_temp_dir = TemporaryDirectory()
            # Define the remote path for the client
            full_remote_path = (
                f"{args.remote_path}"
                f"/{args.name}/{CLIENT_FOLDER_PREFIX}{i}/{DATASETS_CONSTANTS[f'{args.path}_{args.name}'].splits[split_name].split}"
            )
            with MDSWriter(
                columns={"tokens": "ndarray:int32"},
                out=(client_temp_dir.name, full_remote_path),
                compression=args.compression,
            ) as out:
                for j, sample in enumerate(
                    tqdm(
                        samples,
                        desc=f"{CLIENT_FOLDER_PREFIX}{i}_{args.path}_{args.name}_{split_name}",
                        total=(
                            total_num_samples // args.num_clients
                            + int(total_num_samples % args.num_clients)
                            if i == args.num_clients - 1
                            else total_num_samples // args.num_clients
                        ),
                        mininterval=10 * 60,
                    ),
                ):
                    if args.client and i != args.client:
                        continue
                    # Count tokens in the sample and update the frequency map
                    frequency_map.update(Counter(sample["tokens"].tolist()))
                    # Writing the sample to the MDS file
                    out.write(sample)
                    # Break if we have reached the expected number of samples
                    if j == total_num_samples // args.num_clients - 1:
                        break
            # Dump 1-gram frequency maps to temporary file and schedule upload
            frequency_map_filename = Path(
                f"{full_remote_path}/{ONE_GRAM_JSON_FILENAME}",
            )
            with frequency_map_filename.open(mode="x") as f:
                json.dump(
                    dict(frequency_map),
                    f,
                    indent=4,
                )
            # Dump tokenizer files to temporary directory and schedule upload
            tokenizer_filename = Path(f"{client_temp_dir.name}/{TOKENIZER_FOLDER_NAME}")
            tokenizer_filename.mkdir(parents=True, exist_ok=True)
            tokenizer.save_pretrained(tokenizer_filename)


if __name__ == "__main__":
    main(parse_args())
