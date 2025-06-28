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
from collections import defaultdict

from .util import to_ptl_device

from party.tokenizer import ISO_TO_IDX, LANG_TO_ISO

logging.captureWarnings(True)
logger = logging.getLogger('party')


def _repl_alto(fname, preds):
    from itertools import chain
    langs = set(chain.from_iterable(pred.language if pred.language else [] for pred in preds))
    with open(fname, 'rb') as fp:
        doc = etree.parse(fp)
        if langs:
            page = doc.find('.//{*}Page')
            page.set('LANG', langs.pop())
            if langs:
                page.set('OTHERLANGS', ' '.join(langs))
        lines = doc.findall('.//{*}TextLine')
        for line, pred in zip(lines, preds):
            # strip out previous recognition results
            for el in line:
                if el.tag.endswith('Shape'):
                    continue
                elif el.tag.endswith('SP'):
                    line.remove(el)
                elif el.tag.endswith('String'):
                    line.remove(el)
            pred_el = etree.SubElement(line, 'String')
            pred_el.set('CONTENT', pred.prediction)
            pred_el.set('ID', f'_{uuid.uuid4()}')
            if pred.language:
                pred_el.set('LANG', pred.language[0])
    return etree.tostring(doc, encoding='UTF-8', xml_declaration=True)


def _parse_page_custom(s):
    o = defaultdict(list)
    s = s.strip()
    l_chunks = [l_chunk for l_chunk in s.split('}') if l_chunk.strip()]
    if l_chunks:
        for chunk in l_chunks:
            tag, vals = chunk.split('{')
            tag_vals = {}
            vals = [val.strip() for val in vals.split(';') if val.strip()]
            for val in vals:
                key, *val = val.split(':')
                tag_vals[key] = ":".join(val)
            o[tag.strip()].append(tag_vals)
    return dict(o)


def dict_to_page_custom(d) -> str:
    return ' '.join((' '.join(f'{k} {{' + ';'.join(f'{m.strip()}: {n.strip()}' for m, n in i.items()) + ';}' for i in v)) for k, v in d.items())


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
            pred_el.text = pred.prediction
            # add language(s) to custom string
            if pred.language:
                custom_str = line.get('custom', '')
                cs = _parse_page_custom(custom_str)
                cs['language'] = [{'type': lang} for lang in pred.language]
                line.set('custom', dict_to_page_custom(cs))
    return etree.tostring(doc, encoding='UTF-8', xml_declaration=True)


@click.command('ocr',
               epilog=f"""
                       Language codes known to party:
                       {', '.join(' -> '.join((k, v)) for k, v in LANG_TO_ISO.items())}
                       """)
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
@click.option('-m', '--load-from-repo',
              default=None,
              show_default=True,
              help="HTRMoPo identifier of the party model to evaluate")
@click.option('-mi', '--load-from-file',
              default=None,
              show_default=True,
              help="Path to the party model to evaluate")
@click.option('--curves/--boxes', help='Encode line prompts as bounding boxes or curves', default=None, show_default=True)
@click.option('-l', '--language', multiple=True, default=None,
              show_default=True, type=click.Choice(list(ISO_TO_IDX.keys())),
              help='ISO639-3 code(s) for language(s) in input images.')
@click.option('--compile/--no-compile', help='Switch to enable/disable torch.compile() on model', default=True, show_default=True)
@click.option('--quantize/--no-quantize', help='Switch to enable/disable PTQ', default=False, show_default=True)
@click.option('-b', '--batch-size', default=2, help='Set batch size in generator')
def ocr(ctx, input, batch_input, suffix, load_from_repo, load_from_file,
        curves, language, compile, quantize, batch_size):
    """
    Runs text recognition on pre-segmented images in XML format.
    """
    # try importing kraken as we need it for inference
    try:
        from kraken.lib.xml import XMLPage
        from kraken.lib.progress import KrakenProgressBar, KrakenDownloadProgressBar
    except ImportError:
        raise click.UsageError('Inference requires the kraken package')

    if load_from_file and load_from_repo:
        raise click.BadOptionUsage('load_from_file', 'load_from_* options are mutually exclusive.')
    elif load_from_file is None and load_from_repo is None:
        load_from_repo = '10.5281/zenodo.14616981'

    import os
    import glob
    import torch

    from PIL import Image
    from pathlib import Path
    from lightning.fabric import Fabric

    from htrmopo import get_model

    from threadpoolctl import threadpool_limits

    from party.pred import batched_pred
    from party.fusion import PartyModel

    try:
        accelerator, device = to_ptl_device(ctx.meta['device'])
    except Exception as e:
        raise click.BadOptionUsage('device', str(e))

    if load_from_repo:
        with KrakenDownloadProgressBar() as progress:
            download_task = progress.add_task(f'Downloading {load_from_repo}', total=0, visible=True)
            load_from_file = get_model(load_from_repo,
                                       callback=lambda total, advance: progress.update(download_task, total=total, advance=advance)) / 'model.safetensors'

    if curves is True:
        curves = 'curves'
    elif curves is False:
        curves = 'boxes'

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

        model = PartyModel.from_safetensors(load_from_file)

        if compile:
            click.echo('Compiling model ', nl=False)
            try:
                model = torch.compile(model, mode='max-autotune')
                click.secho('\u2713', fg='green')
            except Exception:
                click.secho('\u2717', fg='red')

        if quantize:
            click.echo('Quantizing model ', nl=False)
            try:
                import torchao
                torchao.quantization.utils.recommended_inductor_config_setter()
                model = torchao.autoquant(model)
                click.secho('\u2713', fg='green')
            except Exception:
                click.secho('\u2717', fg='red')

        with KrakenProgressBar() as progress:
            file_prog = progress.add_task('Files', total=len(input))
            for input_file, output_file in input:
                input_file = Path(input_file)
                output_file = Path(output_file)

                doc = XMLPage(input_file)
                im = Image.open(doc.imagename)
                bounds = doc.to_container()
                bounds.language = language
                rec_prog = progress.add_task(f'Processing {input_file}', total=len(bounds.lines))
                predictor = batched_pred(model=model,
                                         im=im,
                                         bounds=bounds,
                                         fabric=fabric,
                                         prompt_mode=curves,
                                         batch_size=batch_size)

                preds = []
                for pred in predictor:
                    logger.info(f'pred (lang: {pred.language}): {pred}')
                    preds.append(pred)
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
