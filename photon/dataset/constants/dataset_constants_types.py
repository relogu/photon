"""Dataclasses and enums for dataset constants."""

from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum

TRAIN_CONSTANT = "train"
TRAIN_SMALL_CONSTANT = "train_small"
VALIDATION_CONSTANT = "validation"
VAL_CONSTANT = "val"
VAL_SMALL_CONSTANT = "val_small"
VAL_XSMALL_CONSTANT = "val_xsmall"
VAL_XXSMALL_CONSTANT = "val_xxsmall"


class ConcatMode(Enum):
    """Describe concatenation modes."""

    NO_CONCAT = "NO_CONCAT"
    CONCAT_TOKENS = "CONCAT_TOKENS"


@dataclass
class DataSplitConstants:
    """Describe constants for a dataset split."""

    path: str  # Path or name of the dataset
    name: str  # Defining the name of the dataset configuration.
    split: str  # Split
    folder_split: str  # Custom split (and folder name)

    truncated_samples: int | None


@dataclass
class DatasetConstants:
    """Describe constants for a dataset."""

    splits: dict[str, DataSplitConstants]

    def __iter__(self) -> Iterator[DataSplitConstants]:
        """Iterate over splits.

        Yields
        ------
        DataSplitConstants
            A split of the dataset.

        """
        yield from self.splits.values()
