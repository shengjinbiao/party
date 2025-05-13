#
# Copyright 2025 Benjamin Kiessling
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
import json
import logging
import unicodedata
from typing import Any, Dict, List, Sequence, Tuple, Union

from rich import print
from rich.table import Table
from collections import Counter
from importlib import resources

from party.tokenizer import ISO_TO_LANG

logger = logging.getLogger(__name__)

__all__ = ['render_report']


def is_printable(char: str) -> bool:
    """
    Determines if a chode point is printable/visible when printed.

    Args:
        char (str): Input code point.

    Returns:
        True if printable, False otherwise.
    """
    letters = ('LC', 'Ll', 'Lm', 'Lo', 'Lt', 'Lu')
    numbers = ('Nd', 'Nl', 'No')
    punctuation = ('Pc', 'Pd', 'Pe', 'Pf', 'Pi', 'Po', 'Ps')
    symbol = ('Sc', 'Sk', 'Sm', 'So')
    printable = letters + numbers + punctuation + symbol

    return unicodedata.category(char) in printable


def make_printable(char: str) -> str:
    """
    Takes a Unicode code point and return a printable representation of it.

    Args:
        char (str): Input code point

    Returns:
        Either the original code point, the name of the code point if it is a
        combining mark, whitespace etc., or the hex code if it is a control
        symbol.
    """
    if not char or is_printable(char):
        return char
    elif unicodedata.category(char) in ('Cc', 'Cs', 'Co'):
        return '0x{:x}'.format(ord(char))
    else:
        try:
            return unicodedata.name(char)
        except ValueError:
            return '0x{:x}'.format(ord(char))


def _render_metric(metric: Union[float, str]) -> str:
    if isinstance(metric, float):
        return f'{metric:.2f}'
    else:
        return metric


def render_report(model: str,
                  micro_cer: float,
                  micro_wer: float,
                  page_macro_cer: float,
                  page_macro_wer: float,
                  per_lang_cer: Dict[str, float],
                  per_lang_wer: Dict[str, float],
                  per_lang_page_macro_cer: Dict[str, float],
                  per_lang_page_macro_wer: Dict[str, float],
                  per_script_cer: Dict[str, float],
                  per_script_page_macro_cer: Dict[str, float]):
    """
    Renders an accuracy report.

    Args:


    """
    print(f'Model: {model}')
    table = Table(title='Global metrics', show_header=True, expand=True)
    for i in ['CER', 'WER', 'CER (macro lang)', 'WER (macro lang)', 'CER (macro page)', 'WER (macro page)']:
        table.add_column(i, justify='left', no_wrap=True)
    lang_metrics = {}
    if len(per_lang_cer) > 0:
        macro_cer = 0
        macro_wer = 0
        for lang in per_lang_cer.keys():
            la = (per_lang_cer[lang],
                  per_lang_wer[lang],
                  per_lang_page_macro_cer[lang],
                  per_lang_page_macro_wer[lang])
            macro_cer += la[0]
            macro_wer += la[1]
            lang_metrics[lang] = la
        macro_cer /= (len(lang_metrics) / 100.)
        macro_wer /= (len(lang_metrics) / 100.)
    else:
        macro_cer = '-'
        macro_wer = '-'

    table.add_row(_render_metric(100*micro_cer),
                  _render_metric(100*micro_wer),
                  _render_metric(macro_cer),
                  _render_metric(macro_wer),
                  _render_metric(100*page_macro_cer),
                  _render_metric(100*page_macro_wer))
    print(table)

    if len(per_lang_cer) > 0:
        table = Table(title='Languages', show_header=True, expand=True)
        for i in ['', 'CER', 'WER', 'CER (macro page)', 'WER (macro page)']:
            table.add_column(i, justify='left', no_wrap=True)
        for lang, metrics in sorted(lang_metrics.items(), key=lambda x: x[1]):

            table.add_row(ISO_TO_LANG[lang].title(),
                          _render_metric(100*metrics[0]),
                          _render_metric(100*metrics[1]),
                          _render_metric(100*metrics[2]),
                          _render_metric(100*metrics[3]))
        print(table)
    table = Table(title='Scripts', show_header=True, expand=True)
    for i in ['', 'CER', 'CER (macro page)']:
        table.add_column(i, justify='left', no_wrap=True)
    for script, _ in sorted(per_script_cer.items(), key=lambda x: x[1], reverse=True):
        table.add_row(script,
                      _render_metric((1.0-per_script_cer[script]) * 100),
                      _render_metric((1.0-per_script_page_macro_cer[script]) * 100))

    print(table)


def global_align(seq1: Sequence[Any], seq2: Sequence[Any]) -> Tuple[List[str], List[str]]:
    """
    Computes a global alignment of two strings.

    Args:
        seq1 (Sequence[Any]):
        seq2 (Sequence[Any]):

    Returns a tuple (list(algn1), list(algn2))
    """
    # calculate cost and direction matrix
    cost = [[0] * (len(seq2) + 1) for x in range(len(seq1) + 1)]
    for i in range(1, len(cost)):
        cost[i][0] = i
    for i in range(1, len(cost[0])):
        cost[0][i] = i
    direction = [[(0, 0)] * (len(seq2) + 1) for x in range(len(seq1) + 1)]
    direction[0] = [(0, x) for x in range(-1, len(seq2))]
    for i in range(-1, len(direction) - 1):
        direction[i + 1][0] = (i, 0)
    for i in range(1, len(cost)):
        for j in range(1, len(cost[0])):
            delcost = ((i - 1, j), cost[i - 1][j] + 1)
            addcost = ((i, j - 1), cost[i][j - 1] + 1)
            subcost = ((i - 1, j - 1), cost[i - 1][j - 1] + (seq1[i - 1] != seq2[j - 1]))
            best = min(delcost, addcost, subcost, key=lambda x: x[1])
            cost[i][j] = best[1]
            direction[i][j] = best[0]
    # backtrace
    algn1: List[Any] = []
    algn2: List[Any] = []
    i = len(direction) - 1
    j = len(direction[0]) - 1
    while direction[i][j] != (-1, 0):
        k, m = direction[i][j]
        if k == i - 1 and m == j - 1:
            algn1.insert(0, seq1[i - 1])
            algn2.insert(0, seq2[j - 1])
        elif k < i:
            algn1.insert(0, seq1[i - 1])
            algn2.insert(0, '')
        elif m < j:
            algn1.insert(0, '')
            algn2.insert(0, seq2[j - 1])
        i, j = k, m
    return algn1, algn2


def compute_script_cer_from_algn(algn1: Sequence[str], algn2: Sequence[str]) -> Dict[str, float]:
    """
    Compute confusion matrices from two globally aligned strings.

    Args:
        align1 (Sequence[str]): sequence 1
        align2 (Sequence[str]): sequence 2

    Returns:
        A tuple (counts, scripts, ins, dels, subs) with `counts` being per-character
        confusions, `scripts` per-script counts, `ins` a dict with per script
        insertions, `del` an integer of the number of deletions, `subs` per
        script substitutions.
    """
    counts: Dict[Tuple[str, str], int] = Counter()
    ref = resources.files(__name__).joinpath('scripts.json')
    with ref.open('rb') as fp:
        script_map = json.load(fp)

    def _get_script(c):
        for s, e, n in script_map:
            if ord(c) == s or (e and s <= ord(c) <= e):
                return n
        return 'Unknown'

    scripts: Dict[Tuple[str, str], int] = Counter()
    ins: Dict[Tuple[str, str], int] = Counter()
    dels: int = 0
    subs: Dict[Tuple[str, str], int] = Counter()
    for u, v in zip(algn1, algn2):
        counts[(u, v)] += 1
    for k, v in counts.items():
        if k[0] == '':
            dels += v
        else:
            script = _get_script(k[0])
            scripts[script] += v
            if k[1] == '':
                ins[script] += v
            elif k[0] != k[1]:
                subs[script] += v
    return {k: (v-(ins[k] + subs[k]))/v for k, v in scripts.items()}
