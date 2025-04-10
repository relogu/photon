"""Module for custom dataset types."""

from dataclasses import dataclass

from transformers import PreTrainedTokenizerBase


@dataclass
class TokenizersCouple:
    """Dataclass for coupling tokenizers in encode/decode."""

    encode_tokenizer: PreTrainedTokenizerBase
    decode_tokenizer: PreTrainedTokenizerBase
