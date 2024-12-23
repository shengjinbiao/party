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
import uuid
import click
import logging

from lxml import etree
from pathlib import Path

from .util import _expand_gt, _validate_manifests, message, to_ptl_device

logging.captureWarnings(True)
logger = logging.getLogger('party')



def _repl_alto(fname, preds):
    with open(fname, 'rb') as fp:
        doc = etree.parse(fp)
        lines = doc.findall('.//{*}TextLine')
        char_idx = 0
        for line, pred in zip(lines, preds):
            idx = 0
            # strip out previous recognition results
            for el in line:
                if el.tag.endswith('Shape'):
                    continue
                elif el.tag.endswith('SP'):
                    line.remove(el)
                elif el.tag.endswith('String'):
                    line.remove(el)
            pred_el = etree.SubElement(line, 'String')
            pred_el.set('CONTENT', pred)
            pred_el.set('ID', str(uuid.uuid4()))
    return etree.tostring(doc, encoding='UTF-8', xml_declaration=True)


def _repl_page(fname, preds):
    with open(fname, 'rb') as fp:
        doc = etree.parse(fp)
        lines = doc.findall('.//{*}TextLine')
        for line, pred in zip(lines, preds):
            # strip out previous recognition results
            for el in line:
                if el.tag.endswith('TextEquiv') or el.tag.endswith('Word'):
                    line.remove(el)
            pred_el = etree.SubElement(etree.SubElement(line, 'TextEquiv'), 'Unicode')
            pred_el.text = pred
    return etree.tostring(doc, encoding='UTF-8', xml_declaration=True)



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
@click.option('-m', '--model',
              default='mittagessen/llama_party',
              show_default=True,
              help="Huggingface hub identifier of the party model")
@click.option('--compile/--no-compile', help='Switch to enable/disable torch.compile() on model')
@click.option('--quantize/--no-quantize', help='Switch to enable/disable PTQ')
@click.option('-b', '--batch-size', default=2, help='Set batch size in generator')
def ocr(ctx, input, batch_input, suffix, model, compile, quantize, batch_size):
    """
    Runs text recognition on pre-segmented images in XML format.
    """
    # try importing kraken as we need it for inference
    try:
        from kraken.lib.xml import XMLPage
        from kraken.lib.progress import KrakenProgressBar
    except ImportError:
        raise click.UsageError('Inference requires the kraken package')

    import os
    import glob
    import torch
    import pathlib
    from PIL import Image
    from lightning.fabric import Fabric

    from threadpoolctl import threadpool_limits

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

    # torchao expects bf16 weights
    if quantize:
        ctx.meta['precision'] = 'bf16-true'

    fabric = Fabric(accelerator=accelerator,
                    devices=device,
                    precision=ctx.meta['precision'])

    with torch.inference_mode(), threadpool_limits(limits=ctx.meta['threads']), fabric.init_tensor(), fabric.init_module():

        model = PartyModel.from_huggingface(pretrained=model)

        if compile:
            click.echo('Compiling model ', nl=False)
            try:
                model = torch.compile(model)
                click.secho('\u2713', fg='green')
            except Exception:
                click.secho('\u2717', fg='red')

        if quantize:
            click.echo('Quantizing model ', nl=False)
            try:
                from optimum.quanto import quantize, qint8
                quantize(model, weights=qint8, activtions=qint8)
                click.secho('\u2713', fg='green')
            except Exception:
                click.secho('\u2717', fg='red')

        # load image transforms
        im_transforms = get_default_transforms()

        # prepare model for generation
        model.prepare_for_generation(batch_size=batch_size, device=fabric.device)
        model = model.eval()

        fabric.to_device(model)
        m_dtype = next(model.parameters()).dtype

        with KrakenProgressBar() as progress:
            file_prog = progress.add_task('Files', total=len(input))
            for input_file, output_file in input:
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
                    logger.info(f'pred: {pred}')
                    progress.update(rec_prog, advance=1)
                with open(output_file, 'wb') as fo:
                    if doc.filetype == 'alto':
                        out_xml = _repl_alto(input_file, preds)
                    elif doc.filetype == 'page':
                        out_xml = _repl_page(input_file, preds)
                    else:
                        raise ValueError(f'{input_file} has unknown XML format {doc.filetype} (not in [alto|page]).')
                    fo.write(out_xml)
                progress.update(file_prog, advance=1)

if __name__ == '__main__':
    cli()
