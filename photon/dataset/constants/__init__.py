"""Module to store constants for the datasets."""

from photon.dataset.constants.dataset_constants_types import DatasetConstants
from photon.dataset.constants.mc4 import (
    c4_de_constants,
    c4_el_constants,
    c4_en_constants,
    c4_es_constants,
    c4_hi_constants,
    c4_it_constants,
    c4_la_constants,
    c4_ms_constants,
    c4_ru_constants,
    c4_sr_constants,
    c4_sw_constants,
    c4_ur_constants,
    c4_zh_constants,
)

DATASETS_CONSTANTS: dict[str, DatasetConstants] = {
    "c4_en": c4_en_constants,
    "c4_it": c4_it_constants,
    "c4_zh": c4_zh_constants,
    "c4_ms": c4_ms_constants,
    "c4_ur": c4_ur_constants,
    "c4_sw": c4_sw_constants,
    "c4_la": c4_la_constants,
    "c4_sr": c4_sr_constants,
    "c4_es": c4_es_constants,
    "c4_de": c4_de_constants,
    "c4_el": c4_el_constants,
    "c4_ru": c4_ru_constants,
    "c4_hi": c4_hi_constants,
}
