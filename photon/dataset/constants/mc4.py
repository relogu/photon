"""Module for constants related to the mC4 dataset."""

from photon.dataset.constants.dataset_constants_types import (
    TRAIN_CONSTANT,
    TRAIN_SMALL_CONSTANT,
    VAL_CONSTANT,
    VAL_SMALL_CONSTANT,
    VAL_XSMALL_CONSTANT,
    VAL_XXSMALL_CONSTANT,
    VALIDATION_CONSTANT,
    DatasetConstants,
    DataSplitConstants,
)

C4_PATH = "allenai/c4"
ENGLISH_CONSTANT = "en"
SERBIAN_CONSTANT = "sr"
LATIN_CONSTANT = "la"
SWAHILI_CONSTANT = "sw"
URDU_CONSTANT = "ur"
MALAY_CONSTANT = "ms"
CHINESE_CONSTANT = "zh"
ITALIAN_CONSTANT = "it"
SPANISH_CONSTANT = "es"
GERMAN_CONSTANT = "de"
GREEK_CONSTANT = "el"
RUSSIAN_CONSTANT = "ru"
HINDI_CONSTANT = "hi"

c4_en_constants = DatasetConstants(
    splits={
        TRAIN_CONSTANT: DataSplitConstants(
            path=C4_PATH,
            name=ENGLISH_CONSTANT,
            split=TRAIN_CONSTANT,
            folder_split=TRAIN_CONSTANT,
            truncated_samples=None,
        ),
        TRAIN_SMALL_CONSTANT: DataSplitConstants(
            path=C4_PATH,
            name=ENGLISH_CONSTANT,
            split=TRAIN_CONSTANT,
            folder_split=TRAIN_SMALL_CONSTANT,
            truncated_samples=100000,
        ),
        VALIDATION_CONSTANT: DataSplitConstants(
            path=C4_PATH,
            name=ENGLISH_CONSTANT,
            split=VALIDATION_CONSTANT,
            folder_split=VAL_CONSTANT,
            truncated_samples=None,
        ),
        VAL_SMALL_CONSTANT: DataSplitConstants(
            path=C4_PATH,
            name=ENGLISH_CONSTANT,
            split=VALIDATION_CONSTANT,
            folder_split=VAL_SMALL_CONSTANT,
            truncated_samples=10000,
        ),
        VAL_XSMALL_CONSTANT: DataSplitConstants(
            path=C4_PATH,
            name=ENGLISH_CONSTANT,
            split=VALIDATION_CONSTANT,
            folder_split=VAL_XSMALL_CONSTANT,
            truncated_samples=3000,
        ),
        VAL_XXSMALL_CONSTANT: DataSplitConstants(
            path=C4_PATH,
            name=ENGLISH_CONSTANT,
            split=VALIDATION_CONSTANT,
            folder_split=VAL_XXSMALL_CONSTANT,
            truncated_samples=100,
        ),
    },
)


# ------------------- C4 (sr) ------------------- #

c4_sr_constants = DatasetConstants(
    splits={
        TRAIN_CONSTANT: DataSplitConstants(
            path=C4_PATH,
            name=SERBIAN_CONSTANT,
            split=TRAIN_CONSTANT,
            folder_split=TRAIN_CONSTANT,
            truncated_samples=None,
        ),
        VALIDATION_CONSTANT: DataSplitConstants(
            path=C4_PATH,
            name=SERBIAN_CONSTANT,
            split=VALIDATION_CONSTANT,
            folder_split=VAL_CONSTANT,
            truncated_samples=None,
        ),
    },
)


# ------------------- C4 (la) ------------------- #

c4_la_constants = DatasetConstants(
    splits={
        TRAIN_CONSTANT: DataSplitConstants(
            path=C4_PATH,
            name=LATIN_CONSTANT,
            split=TRAIN_CONSTANT,
            folder_split=TRAIN_CONSTANT,
            truncated_samples=None,
        ),
        VALIDATION_CONSTANT: DataSplitConstants(
            path=C4_PATH,
            name=LATIN_CONSTANT,
            split=VALIDATION_CONSTANT,
            folder_split=VAL_CONSTANT,
            truncated_samples=None,
        ),
    },
)


# ------------------- C4 (sw) ------------------- #

c4_sw_constants = DatasetConstants(
    splits={
        TRAIN_CONSTANT: DataSplitConstants(
            path=C4_PATH,
            name=SWAHILI_CONSTANT,
            split=TRAIN_CONSTANT,
            folder_split=TRAIN_CONSTANT,
            truncated_samples=None,
        ),
        VALIDATION_CONSTANT: DataSplitConstants(
            path=C4_PATH,
            name=SWAHILI_CONSTANT,
            split=VALIDATION_CONSTANT,
            folder_split=VAL_CONSTANT,
            truncated_samples=None,
        ),
    },
)


# ------------------- C4 (ur) ------------------- #

c4_ur_constants = DatasetConstants(
    splits={
        TRAIN_CONSTANT: DataSplitConstants(
            path=C4_PATH,
            name=URDU_CONSTANT,
            split=TRAIN_CONSTANT,
            folder_split=TRAIN_CONSTANT,
            truncated_samples=None,
        ),
        VALIDATION_CONSTANT: DataSplitConstants(
            path=C4_PATH,
            name=URDU_CONSTANT,
            split=VALIDATION_CONSTANT,
            folder_split=VAL_CONSTANT,
            truncated_samples=None,
        ),
    },
)


# ------------------- C4 (ms) ------------------- #

c4_ms_constants = DatasetConstants(
    splits={
        TRAIN_CONSTANT: DataSplitConstants(
            path=C4_PATH,
            name=MALAY_CONSTANT,
            split=TRAIN_CONSTANT,
            folder_split=TRAIN_CONSTANT,
            truncated_samples=None,
        ),
        VALIDATION_CONSTANT: DataSplitConstants(
            path=C4_PATH,
            name=MALAY_CONSTANT,
            split=VALIDATION_CONSTANT,
            folder_split=VAL_CONSTANT,
            truncated_samples=None,
        ),
    },
)


# ------------------- C4 (zh) ------------------- #

c4_zh_constants = DatasetConstants(
    splits={
        TRAIN_CONSTANT: DataSplitConstants(
            path=C4_PATH,
            name=CHINESE_CONSTANT,
            split=TRAIN_CONSTANT,
            folder_split=TRAIN_CONSTANT,
            truncated_samples=None,
        ),
        VALIDATION_CONSTANT: DataSplitConstants(
            path=C4_PATH,
            name=CHINESE_CONSTANT,
            split=VALIDATION_CONSTANT,
            folder_split=VAL_CONSTANT,
            truncated_samples=None,
        ),
    },
)


# ------------------- C4 (it) ------------------- #

c4_it_constants = DatasetConstants(
    splits={
        TRAIN_CONSTANT: DataSplitConstants(
            path=C4_PATH,
            name=ITALIAN_CONSTANT,
            split=TRAIN_CONSTANT,
            folder_split=TRAIN_CONSTANT,
            truncated_samples=None,
        ),
        VALIDATION_CONSTANT: DataSplitConstants(
            path=C4_PATH,
            name=ITALIAN_CONSTANT,
            split=VALIDATION_CONSTANT,
            folder_split=VAL_CONSTANT,
            truncated_samples=None,
        ),
    },
)


# ------------------- C4 (es) ------------------- #

c4_es_constants = DatasetConstants(
    splits={
        TRAIN_CONSTANT: DataSplitConstants(
            path=C4_PATH,
            name=SPANISH_CONSTANT,
            split=TRAIN_CONSTANT,
            folder_split=TRAIN_CONSTANT,
            truncated_samples=None,
        ),
        VALIDATION_CONSTANT: DataSplitConstants(
            path=C4_PATH,
            name=SPANISH_CONSTANT,
            split=VALIDATION_CONSTANT,
            folder_split=VAL_CONSTANT,
            truncated_samples=None,
        ),
    },
)


# ------------------- C4 (de) ------------------- #

c4_de_constants = DatasetConstants(
    splits={
        TRAIN_CONSTANT: DataSplitConstants(
            path=C4_PATH,
            name=GERMAN_CONSTANT,
            split=TRAIN_CONSTANT,
            folder_split=TRAIN_CONSTANT,
            truncated_samples=None,
        ),
        VALIDATION_CONSTANT: DataSplitConstants(
            path=C4_PATH,
            name=GERMAN_CONSTANT,
            split=VALIDATION_CONSTANT,
            folder_split=VAL_CONSTANT,
            truncated_samples=None,
        ),
    },
)


# ------------------- C4 (el) ------------------- #

c4_el_constants = DatasetConstants(
    splits={
        TRAIN_CONSTANT: DataSplitConstants(
            path=C4_PATH,
            name=GREEK_CONSTANT,
            split=TRAIN_CONSTANT,
            folder_split=TRAIN_CONSTANT,
            truncated_samples=None,
        ),
        VALIDATION_CONSTANT: DataSplitConstants(
            path=C4_PATH,
            name=GREEK_CONSTANT,
            split=VALIDATION_CONSTANT,
            folder_split=VAL_CONSTANT,
            truncated_samples=None,
        ),
    },
)


# ------------------- C4 (ru) ------------------- #

c4_ru_constants = DatasetConstants(
    splits={
        TRAIN_CONSTANT: DataSplitConstants(
            path=C4_PATH,
            name=RUSSIAN_CONSTANT,
            split=TRAIN_CONSTANT,
            folder_split=TRAIN_CONSTANT,
            truncated_samples=None,
        ),
        VALIDATION_CONSTANT: DataSplitConstants(
            path=C4_PATH,
            name=RUSSIAN_CONSTANT,
            split=VALIDATION_CONSTANT,
            folder_split=VAL_CONSTANT,
            truncated_samples=None,
        ),
    },
)


# ------------------- C4 (hi) ------------------- #

c4_hi_constants = DatasetConstants(
    splits={
        TRAIN_CONSTANT: DataSplitConstants(
            path=C4_PATH,
            name=HINDI_CONSTANT,
            split=TRAIN_CONSTANT,
            folder_split=TRAIN_CONSTANT,
            truncated_samples=None,
        ),
        VALIDATION_CONSTANT: DataSplitConstants(
            path=C4_PATH,
            name=HINDI_CONSTANT,
            split=VALIDATION_CONSTANT,
            folder_split=VAL_CONSTANT,
            truncated_samples=None,
        ),
    },
)
