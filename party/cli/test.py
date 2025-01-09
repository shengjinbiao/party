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
party.cli.test
~~~~~~~~~~~~~~

Command line driver for recognition training.
"""
import click
import logging

from typing import List

from .util import _expand_gt, _validate_manifests, message, to_ptl_device

from party.default_specs import RECOGNITION_HYPER_PARAMS

logging.captureWarnings(True)
logger = logging.getLogger('party')

# suppress worker seeding message
logging.getLogger("lightning.fabric.utilities.seed").setLevel(logging.ERROR)


@click.command('test')
@click.pass_context
@click.option('-B', '--batch-size', show_default=True, type=click.INT,
              default=RECOGNITION_HYPER_PARAMS['batch_size'], help='Batch sample size')
@click.option('-m', '--load-from-repo',
              default='10.5281/zenodo.14616981',
              show_default=True,
              help="HTRMoPo identifier of the party model to evaluate")
@click.option('-i', '--load-from-file',
              default=None,
              show_default=True,
              help="Path to the party model to evaluate")
@click.option('-e', '--evaluation-files', show_default=True, default=None, multiple=True,
              callback=_validate_manifests, type=click.File(mode='r', lazy=True),
              help='File(s) with paths to evaluation data.')
@click.option('-d', '--device', show_default=True, default='cpu', help='Select device to use (cpu, cuda:0, cuda:1, ...)')
@click.option('--workers', show_default=True, default=1,
              type=click.IntRange(0),
              help='Number of worker processes when running on CPU.')
@click.option('--threads', show_default=True, default=1,
              type=click.IntRange(1),
              help='Max size of thread pools for OpenMP/BLAS operations.')
@click.option('-u', '--normalization', show_default=True, type=click.Choice(['NFD', 'NFKD', 'NFC', 'NFKC']),
              default=None, help='Ground truth normalization')
@click.option('-n', '--normalize-whitespace/--no-normalize-whitespace',
              show_default=True, default=True, help='Normalizes unicode whitespace')
@click.option('--curves/--boxes', help='Encode line prompts as bounding boxes or curves', default=None, show_default=True)
@click.option('--compile/--no-compile', help='Switch to enable/disable torch.compile() on model', default=True, show_default=True)
@click.option('--quantize/--no-quantize', help='Switch to enable/disable PTQ', default=False, show_default=True)
@click.argument('test_set', nargs=-1, callback=_expand_gt, type=click.Path(exists=False, dir_okay=False))
def test(ctx, batch_size, load_from_repo, load_from_file, evaluation_files,
         device, workers, threads, normalization, normalize_whitespace, curves,
         compile, quantize, test_set):
    """
    Tests a model on XML input data.
    """
    if load_from_file and load_from_repo:
        raise click.BadOptionsUsage('load_from_file', 'load_from_* options are mutually exclusive.')

    try:
        accelerator, device = to_ptl_device(ctx.meta['device'])
    except Exception as e:
        raise click.BadOptionUsage('device', str(e))

    import uuid
    import torch

    from PIL import Image
    from pathlib import Path
    from htrmopo import get_model
    from platformdirs import user_data_dir
    from threadpoolctl import threadpool_limits
    from lightning.fabric import Fabric

    try:
        from kraken.lib.xml import XMLPage
        from kraken.serialization import render_report
        from kraken.lib.dataset import compute_confusions, global_align
        from kraken.lib.progress import KrakenProgressBar, KrakenDownloadProgressBar
    except ImportError:
        raise click.UsageError('Inference requires the kraken package')

    from torchmetrics.text import CharErrorRate, WordErrorRate

    from party.pred import batched_pred
    from party.fusion import PartyModel

    torch.set_float32_matmul_precision('medium')

    if load_from_repo:
        path = Path(user_data_dir('htrmopo')) / str(uuid.uuid5(uuid.NAMESPACE_DNS, load_from_repo))
        try:
            with KrakenDownloadProgressBar() as progress:
                download_task = progress.add_task(f'Downloading {load_from_repo}', total=0, visible=True)
                get_model(load_from_repo,
                          path=path,
                          callback=lambda total, advance: progress.update(download_task, total=total, advance=advance),
                          abort_if_exists=True)
        except ValueError:
            print(f'Model {load_from_repo} already downloaded.')
        load_from_file = path / 'model.safetensors'

    if curves is True:
        curves = 'curves'
    elif curves is False:
        curves = 'boxes'

    # torchao expects bf16 weights
    if quantize:
        ctx.meta['precision'] = 'bf16-true'

    logger.info('Building test set from {} line images'.format(len(test_set) + len(evaluation_files)))

    test_set = list(test_set)

    if evaluation_files:
        test_set.extend(evaluation_files)

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
                from optimum.quanto import quantize, qint8
                quantize(model, weights=qint8, activtions=qint8)
                click.secho('\u2713', fg='green')
            except Exception:
                click.secho('\u2717', fg='red')

        algn_gt: List[str] = []
        algn_pred: List[str] = []
        chars = 0
        error = 0

        test_cer = CharErrorRate()
        test_wer = WordErrorRate()

        with KrakenProgressBar() as progress:
            file_prog = progress.add_task('Files', total=len(test_set))
            for input_file in test_set:
                input_file = Path(input_file)

                doc = XMLPage(input_file)
                im = Image.open(doc.imagename)
                bounds = doc.to_container()
                rec_prog = progress.add_task(f'Processing {input_file}', total=len(bounds.lines))
                predictor = batched_pred(model=model,
                                         im=im,
                                         bounds=bounds,
                                         fabric=fabric,
                                         prompt_mode=curves,
                                         batch_size=batch_size)

                for pred, line in zip(predictor, bounds.lines):
                    x = pred.prediction
                    y = line.text
                    logger.info(f'pred: {x}')
                    chars += len(y)
                    c, algn1, algn2 = global_align(y, x)
                    algn_gt.extend(algn1)
                    algn_pred.extend(algn2)
                    error += c
                    test_cer.update(x, y)
                    test_wer.update(x, y)
                    progress.update(rec_prog, advance=1)
                progress.update(file_prog, advance=1)

            confusions, scripts, ins, dels, subs = compute_confusions(algn_gt, algn_pred)
            rep = render_report(load_from_file,
                                chars,
                                error,
                                1.0 - test_cer.compute(),
                                1.0 - test_wer.compute(),
                                confusions,
                                scripts,
                                ins,
                                dels,
                                subs)
            logger.info(rep)
            message(rep)
