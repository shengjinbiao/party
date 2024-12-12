party
=====

party is **PA**\ ge-wise\ **R**\ ecognition of **T**\ ext-\ **y**. It is a
replacement for conventional text recognizers in ATR system using the
baseline+bounding polygon line data model where it eliminates the need for
bounding polygons.

Party consists of a Swin vision transformer encoder, baseline positional
embeddings, and a `tiny Llama decoder
<https://github.com/mittagessen/bytellama>`_ trained on octet tokenization.

Installation
------------

::

        $ pip install .


Training
--------

Party needs to be trained on datasets precompiled from PageXML or ALTO files
containing line-wise transcriptions and baseline information for each line. The
binary dataset format is **NOT** compatible with kraken but the process of
compilation is fairly similar:

::

        $ party compile -o dataset.arrow --no-reorder *.xml

It is recommended to disable BiDi reordering as the pretrained model has been
trained to recognize RTL text in logical order.

To fine-tune the pretrained base model from one or more dataset files on all
available GPUs:

::

        $ party -d cuda --precision bf16-true fine-tune --workers 32 *.arrow

Training from scratch is also supported through the `party train` command.

Inference
---------

To recognize text in pre-segmented page images in PageXML or ALTO with the
pretrained model run:

::

        $ party ocr -i input_file.xml output_file.xml

It is recommended to adjust the `--compile/--no-compile`,
`--quantize/--no-quantize`, and `--batch-size` arguments to optimize inference
speed for your inference environment.
