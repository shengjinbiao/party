#
# Copyright 2017 Benjamin Kiessling
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing
# permissions and limitations under the License.

"""
Pytorch compatible codec with many-to-many mapping between labels and
graphemes.
"""
import torch
import logging
from collections import Counter
from typing import Dict, List, Sequence, Set, Union, Tuple, Optional

import numpy as np
from torch import IntTensor

from transformers.tokenization_utils import AddedToken, PreTrainedTokenizer

from kraken.lib.exceptions import KrakenCodecException, KrakenEncodeException

__all__ = ['OctetCodec']

logger = logging.getLogger(__name__)


class OctetTokenizer(PreTrainedTokenizer):
    """
    Construct an Octet tokenizer similar to ByT5Codec but with explicit BOS/EOS/pad tokens.

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.

    Args:
        bos_token (`str`, *optional*, defaults to `"<s>"`):
            The end of sequence token.
        eos_token (`str`, *optional*, defaults to `"</s>"`):
            The end of sequence token.
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.
    """

    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self,
                 bos_token="<s>",
                 eos_token="</s>",
                 pad_token="<pad>",
                 **kwargs) -> None:
        # Add extra_ids to the special token list
        pad_token = AddedToken(pad_token, lstrip=False, rstrip=False) if isinstance(pad_token, str) else pad_token
        bos_token = AddedToken(bos_token, lstrip=False, rstrip=False) if isinstance(bos_token, str) else bos_token
        eos_token = AddedToken(eos_token, lstrip=False, rstrip=False) if isinstance(eos_token, str) else eos_token

        self._added_tokens_decoder = {0: pad_token, 1: bos_token, 2: eos_token}
        self.offset = len(self._added_tokens_decoder)
        self._utf_vocab_size = 2**8  # utf is 8 bits

        super().__init__(eos_token=eos_token,
                         bos_token=bos_token,
                         pad_token=pad_token,
                         extra_ids=0,
                         **kwargs)

    @property
    def vocab_size(self):
        return self._utf_vocab_size

    def get_vocab(self):
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size + self.offset)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def _add_eos_if_not_present(self, token_ids: List[int]) -> List[int]:
        """Do not add eos again if user already added it."""
        if len(token_ids) > 0 and token_ids[-1] == self.eos_token_id:
            warnings.warn(
                f"This sequence already has {self.eos_token}. In future versions this behavior may lead to duplicated"
                " eos tokens being added."
            )
            return token_ids
        else:
            return token_ids + [self.eos_token_id]


    def create_token_type_ids_from_sequences(self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. ByT5 does not
        make use of token type ids, therefore a list of zeros is returned.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of zeros.
        """
        eos = [self.eos_token_id]

        if token_ids_1 is None:
            return len(token_ids_0 + eos) * [0]
        return len(token_ids_0 + eos + token_ids_1 + eos) * [0]

    def _tokenize(self, text: str) -> List[str]:
        """Take as input a string and return a list of strings (tokens) for words/sub-words"""
        tokens = [chr(i) for i in text.encode("utf-8")]
        return tokens

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""

        if len(token) != 1:
            token_id = None
        else:
            token_id = ord(token) + self.offset

        return token_id

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        token = chr(index - self.offset)
        return token

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        bstring = b""
        for token in tokens:
            if token in self.added_tokens_decoder:
                tok_string = self.added_tokens_decoder[token].encode("utf-8")
            elif token in self.added_tokens_encoder:
                tok_string = token.encode("utf-8")
            else:
                tok_string = bytes([ord(token)])
            bstring += tok_string
        string = bstring.decode("utf-8", errors="ignore")
        return string

    # OctetTokenizer has no vocab file
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        return ()

class OctetCodec(object):
    """
    Builds a codec converting between graphemes/code points and integer
    label sequences.

    charset may either be a string, a list or a dict. In the first case
    each code point will be assigned a label, in the second case each
    string in the list will be assigned a label, and in the final case each
    key string will be mapped to the value sequence of integers. In the
    first two cases labels will be assigned automatically. When a mapping
    is manually provided the label codes need to be a prefix-free code.

    Indices 0, 1, and 2 are reserved for padding, SOS, and EOS respectively.
    Output labels and input dictionaries are/should be 3-indexed.

    Encoded sequences have EOS labels appended automatically. The
    decoder requires them to be stripped beforehand.

    Args:
        charset: Input character set.
        strict: Flag indicating if encoding/decoding errors should be ignored
                or cause an exception.

    Raises:
        KrakenCodecException: If the character set contains duplicate
                              entries or the mapping is non-singular or
                              non-prefix-free.
    """
    tokenizer = OctetTokenizer()
    sos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id

    def __init__(self, *args, **kwargs):
        super().__init__()

    def __len__(self) -> int:
        """
        Total number of input labels the codec can decode.
        """
        return len(self.tokenizer)

    @property
    def is_valid(self) -> bool:
        """
        Returns True if the codec is prefix-free (in label space) and
        non-singular (in both directions).
        """
        return True

    @property
    def max_label(self) -> int:
        """
        Returns the maximum label value.
        """
        return max(x for x in self.tokenizer.get_vocab().values())

    def encode(self, s: str) -> IntTensor:
        """
        Encodes a string into a sequence of labels.

        If the code is non-singular we greedily encode the longest sequence first.

        Args:
            s: Input unicode string

        Returns:
            Ecoded label sequence
        """
        return torch.tensor(self.tokenizer(s).input_ids, dtype=torch.int)

    def decode(self, labels: IntTensor) -> str:
        """
        Decodes a labelling.

        Given a labelling with cuts and  confidences returns a string with the
        cuts and confidences aggregated across label-code point
        correspondences. When decoding multilabels to code points the resulting
        cuts are min/max, confidences are averaged.

        Args:
            labels: Input containing tuples (label, start, end,
                           confidence).

        Returns:
            A list of tuples (code point, start, end, confidence)
        """
        return self.tokenizer.decode(labels)

    def __repr__(self):
        return 'OctetCodec()'
