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
from typing import List, TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from torch import IntTensor

__all__ = ['OctetTokenizer']

logger = logging.getLogger(__name__)

LANG_TO_ISO = {'arabic': 'ara',
               'catalan': 'cat',
               'chinese': 'cmn',
               'corsican': 'cos',
               'czech': 'ces',
               'old_church_slavonic': 'chu',
               'german': 'deu',
               'english': 'eng',
               'persian': 'fas',
               'finnish': 'fin',
               'french': 'fra',
               'greek': 'grc',
               'hebrew': 'heb',
               'italian': 'ita',
               'japanese': 'jpn',
               'georgian': 'kat',
               'latin': 'lat',
               'malayalam': 'mal',
               'newari': 'new',
               'norwegian': 'nor',
               'dutch': 'nld',
               'turkish_ottoman': 'ota',
               'occitan': 'oci',
               'portuguese': 'por',
               'russian': 'rus',
               'sanskrit': 'san',
               'swedish': 'swe',
               'spanish': 'spa',
               'syriac': 'syr',
               'urdu': 'urd',
               'ukrainian': 'ukr',
               'yiddish': 'yid'}

ISO_TO_IDX = {'ara': 0,
              'cat': 1,
              'ces': 2,
              'chu': 3,
              'cmn': 4,
              'cos': 5,
              'deu': 6,
              'eng': 7,
              'fas': 8,
              'fin': 9,
              'fra': 10,
              'grc': 11,
              'heb': 12,
              'ita': 13,
              'jpn': 14,
              'kat': 15,
              'lat': 16,
              'mal': 17,
              'new': 18,
              'nld': 19,
              'nor': 20,
              'oci': 21,
              'ota': 22,
              'por': 23,
              'rus': 24,
              'san': 25,
              'spa': 26,
              'swe': 27,
              'syr': 28,
              'ukr': 29,
              'urd': 30,
              'yid': 31}


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
    # leave space for 128 lang tokens before actual text output
    _lang_offset = 3
    _offset = _lang_offset + min(len(ISO_TO_IDX), 128)

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
               langs: Optional[[List[str]]] = None,
               add_bos: bool = True,
               add_eos: bool = True) -> List[int]:
        """
        Encode text into token IDs.

        Args:
            text: The input text to be encoded, unbatched.
            langs: List of lang tokens to insert between BOS and first text token.
            add_bos: Whether to prepend BOS to the input, defaults to True.
            add_eos: Whether to append EOS to the input, defaults to True.

        Returns:
            List[int]: The encoded token IDs.
        """
        tokens = []
        if add_bos:
            tokens.append(self.bos_id)
        if langs:
            tokens.add([self._lang_offset + ISO_TO_IDX[lang] for lang in langs])
        tokens.extend([i + self._offset for i in text.encode("utf-8")])
        if add_eos:
            tokens.append(self.eos_id)

        return tokens

    def decode(self, ids: 'IntTensor') -> str:
        """Decode token IDs to strings.

        Args:
            ids: The input token IDs to be decoded.

        Returns:
            str: The decoded text.
        """
        string = bytes([x - self._offset for x in ids]).decode("utf-8", errors="ignore")
        return string
