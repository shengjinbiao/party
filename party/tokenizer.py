#
# Copyright 2024 Benjamin Kiessling
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
"""
import logging
from typing import List

from torch import IntTensor

__all__ = ['OctetTokenizer']

logger = logging.getLogger(__name__)


class OctetTokenizer(object):
    """
    A non-trainable tokenizer that simple encodes strings as UTF-8 and uses
    their octets.

    Examples:
        >>> tokenizer = OctetTokenizer()
        >>> tokenized_text = tokenizer.encode("Hello world!", add_bos=True, add_eos=True)
        >>> print(tokenized_text)
        [1, 31587, 29644, 102, 2]
    """
    pad_id = 0
    bos_id = 1
    eos_id = 2
    _offset = 3

    def __init__(self):
        pass

    def __len__(self) -> int:
        """
        Total number of input labels the codec can decode.
        """
        return 256 + self._offset

    @property
    def max_label(self) -> int:
        """
        Returns the maximum label value.
        """
        return 255 + self._offset

    def encode(self,
               text: str,
               add_bos: bool = True,
               add_eos: bool = True) -> List[int]:
        """
        Encode text into token IDs.

        Args:
            text: The input text to be encoded, unbatched.
            add_bos: Whether to prepend BOS to the input, defaults to True.
            add_eos: Whether to append EOS to the input, defaults to True.

        Returns:
            List[int]: The encoded token IDs.
        """
        tokens = []
        if add_bos:
            tokens.append(self.bos_id)
        tokens.extend([i + self._offset for i in text.encode("utf-8")])
        if add_eos:
            tokens.append(self.eos_id)

        return tokens

    def decode(self, ids: IntTensor) -> str:
        """Decode token IDs to strings.

        Args:
            ids: The input token IDs to be decoded.

        Returns:
            str: The decoded text.
        """
        string = bytes([x - self._offset for x in ids]).decode("utf-8", errors="ignore")
        return string
