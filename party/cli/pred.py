#
# Copyright 2022 Benjamin Kiessling
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
party.cli.pred
~~~~~~~~~~~~~~

Command line driver for recognition inference.
"""
import logging

import click

from pathlib import Path
from .util import _expand_gt, _validate_manifests, message, to_ptl_device

logging.captureWarnings(True)
logger = logging.getLogger('party')


@click.command('ocr')
@click.pass_context
@click.option('-i', '--input',
              type=(click.Path(exists=True, dir_okay=False, path_type=Path),  # type: ignore
                    click.Path(writable=True, dir_okay=False, path_type=Path)),
              multiple=True,
              help='Input-output file pairs. Each input file (first argument) is mapped to one '
                   'output file (second argument), e.g. `-i input.alto output.alto`')
@click.option('-I', '--batch-input', multiple=True, help='Glob expression to add multiple files at once.')
@click.option('-o', '--suffix', default='', show_default=True,
              help='Suffix for output files from batch and PDF inputs.')
@click.option('-h', '--hocr', 'serializer',
              help='Switch between hOCR, ALTO, abbyyXML, PageXML or "native" '
              'output. Native are plain image files for image, JSON for '
              'segmentation, and text for transcription output.',
              flag_value='hocr')
@click.option('-a', '--alto', 'serializer', flag_value='alto')
@click.option('-y', '--abbyy', 'serializer', flag_value='abbyyxml')
@click.option('-x', '--pagexml', 'serializer', flag_value='pagexml')
@click.option('-n', '--native', 'serializer', flag_value='native', default=True,
              show_default=True)
@click.option('-m', '--model',
              default='mittagessen/llama_party',
              show_default=True,
              help="Huggingface hub identifier of the party model")
@click.option('--compile/--no-compile', help='Switch to enable/disable torch.compile() on model')
@click.option('-b', '--batch-size', default=2, help='Set batch size in generator')
def ocr(ctx, input, batch_input, suffix, serializer, model, compile, batch_size):
    """
    Runs text recognition on pre-segmented images in XML format.
    """
    # try importing kraken as we need it for inference
    try:
        from kraken.lib.xml import XMLPage
        from kraken.serialization import serialize
    except ImportError:
        raise click.UsageError('Inference requires the kraken package')

    import torch
    import pathlib
    from rich.progress import Progress
    from PIL import Image
    from lightning.fabric import Fabric

    from party.dataset import get_default_transforms, _to_curve
    from party.fusion import PartyModel

    try:
        accelerator, device = to_ptl_device(ctx.meta['device'])
    except Exception as e:
        raise click.BadOptionUsage('device', str(e))

    # parse input files
    input = list(input)
    # expand batch inputs
    if batch_input and suffix:
        for batch_expr in batch_input:
            for in_file in glob.glob(os.path.expanduser(batch_expr), recursive=True):
                input.append((in_file, '{}{}'.format(os.path.splitext(in_file)[0], suffix)))

    fabric = Fabric(accelerator=accelerator,
                    devices=device,
                    precision=ctx.meta['precision'])

    with fabric.init_module():
        model = PartyModel.from_huggingface(pretrained=model)

    if compile:
        model = torch.compile(model)

    # load image transforms
    im_transforms = get_default_transforms()

    # prepare model for generation
    model.prepare_for_generation(batch_size=batch_size)
    model = model.eval()

    m_dtype = next(model.parameters()).dtype

    with Progress() as progress:
        file_prog = progress.add_task('Files', len(input))
        for input_file, output_file in input:
            progress.update(file_prog, advance=1)
            input_file = pathlib.Path(input_file)
            output_file = pathlib.Path(output_file)

            doc = XMLPage(input_file)
            im = Image.open(doc.imagename)
            bounds = doc.to_container()
            rec_prog = progress.add_task(f'Processing {input_file}', total=len(bounds.lines))
            image_input = fabric.to_device(im_transforms(im)).to(m_dtype).unsqueeze(0)
            curves = fabric.to_device(torch.tensor([_to_curve(line.baseline, im.size).as_py() for line in bounds.lines], dtype=m_dtype))
            curves = curves.view(-1, 4, 2)
            preds = []
            for pred in model.predict_string(encoder_input=image_input,
                                             curves=curves):
                preds.append(pred)
                progress.update(rec_prog, advance=1)
            print(preds)
        progress.update(file_prog, advance=1)

if __name__ == '__main__':
    cli()
