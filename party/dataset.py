#
# Copyright 2015 Benjamin Kiessling
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
Utility functions for data loading and training of VGSL networks.
"""
import io
import torch
import torch.nn.functional as F
import numpy as np
import lightning.pytorch as L

import tempfile
import pyarrow as pa

from typing import (TYPE_CHECKING, Any, Callable, List, Literal, Optional,
                    Tuple, Union, Sequence)

from functools import partial
from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader

from PIL import Image

from scipy.special import comb
from shapely.geometry import LineString

from kraken.lib import functional_im_transforms as F_t
from kraken.lib.xml import XMLPage

from party.codec import ByT5Codec

if TYPE_CHECKING:
    from os import PathLike

__all__ = ['TextLineDataModule']

import logging

logger = logging.getLogger(__name__)

Image.MAX_IMAGE_PIXELS = 20000 ** 2


def _to_curve(baseline, im_size, min_points: int = 8) -> torch.Tensor:
    """
    Converts poly(base)lines to Bezier curves.
    """
    baseline = np.array(baseline)
    if len(baseline) < min_points:
        ls = LineString(baseline)
        baseline = np.stack([np.array(ls.interpolate(x, normalized=True).coords)[0] for x in np.linspace(0, 1, 8)])
    # control points
    curve = np.concatenate(([baseline[0]], bezier_fit(baseline), [baseline[-1]]))/im_size
    curve = curve.flatten()
    return pa.scalar(curve, type=pa.list_(pa.float32()))


def _to_bbox(boundary, im_size) -> torch.Tensor:
    """
    Converts a bounding polygon to a bbox in xyxyc_xc_yhw format.
    """
    flat_box = [point for pol in boundary for point in pol]
    xmin, xmax = min(flat_box[::2]), max(flat_box[::2])
    ymin, ymax = min(flat_box[1::2]), max(flat_box[1::2])
    w = xmax - xmin
    h = ymax - ymin
    cx = (xmin + xmax) / 2
    cy = (ymin + ymax) / 2
    bbox = np.array([[xmin, ymin], [xmax, ymax], [cx, cy], [w, h]]) / im_size
    bbox = bbox.flatten()
    return pa.scalar(bbox, type=pa.list_(pa.float32()))


def compile(files: Optional[List[Union[str, 'PathLike']]] = None,
            output_file: Union[str, 'PathLike'] = None,
            reorder: Union[bool, Literal['L', 'R']] = True,
            normalize_whitespace: bool = True,
            normalization: Optional[Literal['NFD', 'NFC', 'NFKD', 'NFKC']] = None,
            callback: Callable[[int, int], None] = lambda chunk, lines: None) -> None:
    """
    Compiles a collection of XML facsimile files into a binary arrow dataset.

    Args:
        files: List of XML files
        output_file: destination to write arrow file to
        reorder: text reordering
        normalize_whitespace: whether to normalize all whitespace to ' '
        normalization: Unicode normalization to apply to data.
        callback: progress callback
    """
    text_transforms: List[Callable[[str], str]] = []

    # pyarrow structs
    line_struct = pa.struct([('text', pa.list_(pa.int32())),
                             ('curve', pa.list_(pa.float32())),
                             ('bbox', pa.list_(pa.float32()))])
    page_struct = pa.struct([('im', pa.binary()), ('lines', pa.list_(line_struct))])

    codec = ByT5Codec()

    if normalization:
        text_transforms.append(partial(F_t.text_normalize, normalization=normalization))
    if normalize_whitespace:
        text_transforms.append(F_t.text_whitespace_normalize)
        if reorder:
            if reorder in ('L', 'R'):
                text_transforms.append(partial(F_t.text_reorder, base_dir=reorder))
            else:
                text_transforms.append(F_t.text_reorder)

    num_lines = 0
    # helper variables to enable padding to longest sequence without iterating
    # over set during training.
    max_lines_in_page = 0
    max_octets_in_line = 0
    schema = pa.schema([('pages', page_struct)])

    callback(0, len(files))

    with tempfile.NamedTemporaryFile() as tmpfile:
        with pa.OSFile(tmpfile.name, 'wb') as sink:
            with pa.ipc.new_file(sink, schema) as writer:
                for file in files:
                    try:
                        page = XMLPage(file).to_container()
                        # pick image format with smallest size
                        image_candidates = list(set(page.imagename.with_suffix(x) for x in ['.jxl', '.png']).union([page.imagename]))
                        cand_idxs = np.argsort([t.stat().st_size if t.exists() else np.inf for t in image_candidates])
                        im_path = None
                        for idx in cand_idxs:
                            try:
                                with Image.open(image_candidates[idx]) as im:
                                    im_size = im.size
                                im_path = image_candidates[idx]
                                break
                            except Exception:
                                continue
                    except Exception:
                        continue
                    if im_path is None:
                        continue
                    page_data = []
                    for line in page.lines:
                        try:
                            text = line.text
                            for func in text_transforms:
                                text = func(text)
                            if not text:
                                logger.info(f'Text line "{line.text}" is empty after transformations')
                                continue
                            if not line.baseline:
                                logger.info('No baseline given for line')
                                continue
                            encoded_line = codec.encode(text).numpy()
                            max_octets_in_line = max(len(encoded_line), max_octets_in_line)
                            page_data.append(pa.scalar({'text': pa.scalar(encoded_line),
                                                        'curve': _to_curve(line.baseline, im_size),
                                                        'bbox': _to_bbox(line.boundary, im_size)},
                                                       line_struct))
                            num_lines += 1
                        except Exception:
                            continue
                    if len(page_data) > 1:
                        with open(im_path, 'rb') as fp:
                            im = fp.read()
                        ar = pa.array([pa.scalar({'im': im, 'lines': page_data}, page_struct)], page_struct)
                        writer.write(pa.RecordBatch.from_arrays([ar], schema=schema))
                        max_lines_in_page = max(len(page_data), max_lines_in_page)
                    callback(1, len(files))
        with pa.memory_map(tmpfile.name, 'rb') as source:
            metadata = {'num_lines': num_lines.to_bytes(4, 'little'),
                        'max_lines_in_page': max_lines_in_page.to_bytes(4, 'little'),
                        'max_octets_in_line': max_octets_in_line.to_bytes(4, 'little')}
            schema = schema.with_metadata(metadata)
            ds_table = pa.ipc.open_file(source).read_all()
            new_table = ds_table.replace_schema_metadata(metadata)
            with pa.OSFile(output_file, 'wb') as sink:
                with pa.ipc.new_file(sink, schema=schema) as writer:
                    for batch in new_table.to_batches():
                        writer.write(batch)


def _validation_worker_init_fn(worker_id):
    """ Fix random seeds so that augmentation always produces the same
        results when validating. Temporarily increase the logging level
        for lightning because otherwise it will display a message
        at info level about the seed being changed. """
    from lightning.pytorch import seed_everything
    seed_everything(42)


def collate_null(batch):
    return batch[0]


def collate_sequences(im, page_data):
    """
    Sorts and pads image data.
    """
    if isinstance(page_data[0][0], str):
        labels = [x for x, _ in page_data]
    else:
        max_label_len = max(len(x) for x, _ in page_data)
        labels = torch.stack([F.pad(x, pad=(0, max_label_len-len(x)), value=-100) for x, _ in page_data]).long()
        # replace Mistral EOS token id with T5 ones
        labels[labels == 2] = 1
    label_lens = torch.LongTensor([len(x) for x, _ in page_data])
    curves = torch.stack([x for _, x in page_data])
    return {'image': im,
            'target': labels,
            'curves': curves,
            'target_lens': label_lens}


class TextLineDataModule(L.LightningDataModule):
    def __init__(self,
                 training_data: Union[str, 'PathLike'],
                 evaluation_data: Union[str, 'PathLike'],
                 augmentation: bool = False,
                 batch_size: int = 16,
                 num_workers: int = 8):
        super().__init__()

        self.save_hyperparameters()

        self.im_transforms = v2.Compose([v2.Resize((2560, 1920)),
                                         v2.ToImage(),
                                         v2.ToDtype(torch.float32, scale=True),
                                         v2.Normalize(mean=[0.4850, 0.4560, 0.4060], std=[0.2290, 0.2240, 0.2250])])

        # codec is stateless so we can just initiate it here
        self.codec = ByT5Codec()

        self.pad_id = self.codec.pad
        self.bos_id = self.codec.bos
        self.eos_id = self.codec.eos

        self.num_classes = self.codec.max_label + 1

    def setup(self, stage: str):
        """
        Actually builds the datasets.
        """
        self.train_set = BinnedBaselineDataset(self.hparams.training_data,
                                               im_transforms=self.im_transforms,
                                               augmentation=self.hparams.augmentation,
                                               max_batch_size=self.hparams.batch_size)
        self.val_set = BinnedBaselineDataset(self.hparams.evaluation_data,
                                             im_transforms=self.im_transforms,
                                             augmentation=self.hparams.augmentation,
                                             max_batch_size=self.hparams.batch_size)

    def train_dataloader(self):
        return DataLoader(self.train_set,
                          batch_size=1,
                          num_workers=self.hparams.num_workers,
                          pin_memory=True,
                          shuffle=False,
                          collate_fn=collate_null)

    def val_dataloader(self):
        return DataLoader(self.val_set,
                          shuffle=False,
                          batch_size=1,
                          num_workers=self.hparams.num_workers,
                          pin_memory=True,
                          collate_fn=collate_null,
                          worker_init_fn=_validation_worker_init_fn)


class BinnedBaselineDataset(Dataset):
    """
    Dataset for training a line recognition model from baseline data.

    Images are binned, so the batch_size parameter of the data loader is an
    upper limit of the number of samples returned.

    Args:
        normalization: Unicode normalization for gt
        whitespace_normalization: Normalizes unicode whitespace and strips
                                  whitespace.
        skip_empty_lines: Whether to return samples without text.
        reorder: Whether to rearrange code points in "display"/LTR order.
                 Set to L|R to change the default text direction.
        im_transforms: Function taking an PIL.Image and returning a tensor
                       suitable for forward passes.
        augmentation: Enables augmentation.
        max_batch_size: Maximum size of a batch. All samples from a batch will
                        come from a single page.
    """
    def __init__(self,
                 files: Sequence[Union[str, 'PathLike']],
                 im_transforms: Callable[[Any], torch.Tensor] = None,
                 augmentation: bool = False,
                 max_batch_size: int = 32) -> None:
        self.files = files
        self.transforms = im_transforms
        self.aug = None
        self.max_batch_size = max_batch_size
        self.max_seq_len = 0
        self._len = 0

        self.arrow_table = None

        for file in files:
            with pa.memory_map(file, 'rb') as source:
                ds_table = pa.ipc.open_file(source).read_all()
                raw_metadata = ds_table.schema.metadata
                if not raw_metadata or b'num_lines' not in raw_metadata:
                    raise ValueError(f'{file} does not contain a valid metadata record.')
                if not self.arrow_table:
                    self.arrow_table = ds_table
                else:
                    self.arrow_table = pa.concat_tables([self.arrow_table, ds_table])
                self._len += int.from_bytes(raw_metadata[b'num_lines'], 'little')
                self.max_seq_len = max(int.from_bytes(raw_metadata[b'max_octets_in_line'], 'little'), self.max_seq_len)

        if augmentation:
            from party.augmentation import DefaultAugmenter
            self.aug = DefaultAugmenter()

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # just sample from a random page
        rng = np.random.default_rng()
        idx = rng.integers(0, len(self.arrow_table))

        item = self.arrow_table.column('pages')[idx].as_py()
        logger.debug(f'Attempting to load {item["im"]}')
        im, page_data = item['im'], item['lines']
        try:
            im = Image.open(io.BytesIO(im)).convert('RGB')
        except Exception:
            return self[0]
        im = self.transforms(im)

        if self.aug:
            im = im.transpose((1, 2, 0))
            o = self.aug(image=im)
            im = torch.tensor(o['image'].transpose(2, 0, 1))

        # sample up to max_batch_size lines and targets
        num_samples = min(self.max_batch_size, len(page_data))
        # sample randomly between baselines
        lines = [page_data[x] for x in rng.choice(len(page_data), num_samples, replace=False, shuffle=False)]
        lines = [(torch.tensor(x['text'], dtype=torch.int32), torch.tensor(x['curve']).view(4, 2)) for x in lines]
        return collate_sequences(im.unsqueeze(0), lines)

    def __len__(self) -> int:
        return self._len // self.max_batch_size


# magic lsq cubic bezier fit function from the internet.
def Mtk(n, t, k):
    return t**k * (1-t)**(n-k) * comb(n, k)


def BezierCoeff(ts):
    return [[Mtk(3, t, k) for k in range(4)] for t in ts]


def bezier_fit(bl):
    x = bl[:, 0]
    y = bl[:, 1]
    dy = y[1:] - y[:-1]
    dx = x[1:] - x[:-1]
    dt = (dx ** 2 + dy ** 2)**0.5
    t = dt/dt.sum()
    t = np.hstack(([0], t))
    t = t.cumsum()

    Pseudoinverse = np.linalg.pinv(BezierCoeff(t))  # (9,4) -> (4,9)

    control_points = Pseudoinverse.dot(bl)  # (4,9)*(9,2) -> (4,2)
    medi_ctp = control_points[1:-1, :]
    return medi_ctp
