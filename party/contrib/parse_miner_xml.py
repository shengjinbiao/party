#! /usr/bin/env python
"""
A small script compiling PDFMiner-generated XML and the corresponding PDFs into
a binary dataset file for party.
"""
import io
import os
import sys
import regex
import numpy as np
from lxml import etree
import pypdfium2 as pdfium

import tempfile
import pyarrow as pa

from scipy.special import comb
from shapely.geometry import LineString

from party.codec import OctetCodec

from rich.progress import track


def attr_to_bbox(s):
    x1, y1, x2, y2 = s.split(',')
    return float(x1), float(y1), float(x2), float(y2)


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


def _to_curve(baseline, im_size, min_points: int = 8):
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


def _to_bbox(coords, im_size):
    """
    Converts a bounding polygon to a bbox in xyxyc_xc_yhw format.
    """
    xmin, ymin, xmax, ymax = coords
    w = xmax - xmin
    h = ymax - ymin
    cx = (xmin + xmax) / 2
    cy = (ymin + ymax) / 2
    bbox = np.array([[xmin, ymin], [xmax, ymax], [cx, cy], [w, h]]) / im_size
    bbox = bbox.flatten()
    return pa.scalar(bbox, type=pa.list_(pa.float32()))


line_struct = pa.struct([('text', pa.list_(pa.int32())), ('curve', pa.list_(pa.float32())), ('bbox', pa.list_(pa.float32()))])
page_struct = pa.struct([('im', pa.binary()), ('lines', pa.list_(line_struct))])
schema = pa.schema([('pages', page_struct)])

codec = OctetCodec()

output_file = sys.argv[1]

docs = [x.strip() for x in open(sys.argv[2], 'r').readlines()]
parser = etree.XMLParser(recover=True)

with tempfile.NamedTemporaryFile() as tmpfile:
    with pa.OSFile(tmpfile.name, 'wb') as sink:
        with pa.ipc.new_file(sink, schema) as writer:
            num_lines = 0
            max_lines_in_page = 0
            max_octets_in_line = 0
            for doc in track(docs, description='Reading XML files'):
                with open(doc, 'r') as fp:
                    try:
                        tree = etree.parse(doc, parser=parser)
                        pdf = pdfium.PdfDocument(os.path.splitext(doc)[0] + '.pdf')
                    except Exception:
                        continue
                    for pdf_page, page in zip(pdf, tree.findall('.//page')):
                        _, _, *page_dim = attr_to_bbox(page.get('bbox'))
                        page_dim = tuple(page_dim)
                        page_data = []
                        for line in page.findall('.//textline'):
                            try:
                                x1, y1, x2, y2 = attr_to_bbox(line.get('bbox'))
                                # approximate baseline by straight line placed 10% into the
                                # line bbox from the bottom.
                                bl_y = y2 - ((y2 - y1) * 0.1)
                                baseline = _to_curve(np.array((x1, bl_y, x2, bl_y)).reshape(2, 2), page_dim)
                                bbox = _to_bbox((x1, y1, x2, y2), page_dim)
                                text = codec.encode(regex.sub(r'\s', ' ', ''.join(c.text for c in line.iterfind('text'))).strip()).numpy()
                                max_octets_in_line = max(len(text), max_octets_in_line)
                                page_data.append(pa.scalar({'text': pa.scalar(text),
                                                            'curve': baseline,
                                                            'bbox': bbox}, line_struct))
                            except Exception:
                                continue
                        # render PDF to image
                        im = pdf_page.render(scale=4).to_pil()
                        fp = io.BytesIO()
                        im.save(fp, format='png')
                        num_lines += len(page_data)
                        max_lines_in_page = max(len(page_data), max_lines_in_page)
                        # flush into arrow file
                        if len(page_data):
                            ar = pa.array([pa.scalar({'im': fp.getvalue(), 'lines': page_data}, page_struct)], page_struct)
                            writer.write(pa.RecordBatch.from_arrays([ar], schema=schema))

    print(f'Writing temporary file {tmpfile.name} to {output_file}')
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
