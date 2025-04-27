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

from typing import List, TYPE_CHECKING, Optional, Set, Tuple

if TYPE_CHECKING:
    from torch import IntTensor

__all__ = ['OctetTokenizer']

logger = logging.getLogger(__name__)

LANG_TO_ISO = {'arabic': 'ara',
               'catalan': 'cat',
               'chinese': 'cmn',
               'church_slavonic': 'chu',
               'corsican': 'cos',
               'czech': 'ces',
               'german': 'deu',
               'english': 'eng',
               'persian': 'fas',
               'finnish': 'fin',
               'french': 'fra',
               'ancient_greek': 'grc',
               'hebrew': 'heb',
               'italian': 'ita',
               'japanese': 'jpn',
               'georgian': 'kat',
               'ladino': 'lad',
               'latin': 'lat',
               'malayalam': 'mal',
               'middle_dutch': 'dum',
               'middle_french': 'frm',
               'newari': 'new',
               'norwegian': 'nor',
               'dutch': 'nld',
               'ottoman_turkish': 'ota',
               'occitan': 'oci',
               'picard': 'pcd',
               'portuguese': 'por',
               'russian': 'rus',
               'sanskrit': 'san',
               'swedish': 'swe',
               'spanish': 'spa',
               'syriac': 'syr',
               'urdu': 'urd',
               'ukrainian': 'ukr',
               'undetermined': 'und',
               'shorthand_german': 'qaa',
               'yiddish': 'yid'}

ISO_TO_IDX = {'ara': 0,
              'cat': 1,
              'ces': 2,
              'chu': 3,
              'cmn': 4,
              'cos': 5,
              'deu': 6,
              'dum': 7,
              'eng': 8,
              'fas': 9,
              'fin': 10,
              'fra': 11,
              'frm': 12,
              'grc': 13,
              'heb': 14,
              'ita': 15,
              'jpn': 16,
              'kat': 17,
              'lad': 18,
              'lat': 19,
              'mal': 20,
              'new': 21,
              'nld': 22,
              'nor': 23,
              'oci': 24,
              'ota': 25,
              'pcd': 26,
              'por': 27,
              'qaa': 28,
              'rus': 29,
              'san': 30,
              'spa': 31,
              'swe': 32,
              'syr': 33,
              'ukr': 34,
              'und': 35,
              'urd': 36,
              'yid': 37}

LANG_IDX_TO_ISO = {v: k for k, v in ISO_TO_IDX.items()}

OFFSET = 3
LANG_OFFSET = OFFSET + 256
TOKEN_NUM = LANG_OFFSET + max(len(ISO_TO_IDX), 128)


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
    _offset = 3
    _lang_offset = _offset + 256

    def __init__(self):
        pass

    def __len__(self) -> int:
        """
        Total number of input labels the codec can decode.
        """
        return TOKEN_NUM

    @property
    def max_label(self) -> int:
        """
        Returns the maximum label value.
        """
        return TOKEN_NUM - 1

    def encode(self,
               text: str,
               langs: Optional[List[str]] = None,
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
            tokens.extend([LANG_OFFSET + ISO_TO_IDX[lang] for lang in langs])
        tokens.extend([i + OFFSET for i in text.encode("utf-8")])
        if add_eos:
            tokens.append(self.eos_id)

        return tokens

    def decode(self, ids: 'IntTensor') -> Tuple[str, Set[str]]:
        """Decode token IDs to strings.

        Args:
            ids: The input token IDs to be decoded.

        Returns:
            A tuple containing the decoded text and any language tags in
            the input tensor.
        """
        ids = [id - OFFSET for id in ids if OFFSET <= id < LANG_OFFSET]
        lang_ids = set(LANG_IDX_TO_ISO.get(id - LANG_OFFSET, 'und') for id in ids if id >= LANG_OFFSET)
        text = bytes(ids).decode("utf-8", errors="ignore")
        return text, lang_ids
