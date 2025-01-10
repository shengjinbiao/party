party
=====

party is **PA**\ ge-wise **R**\ ecognition of **T**\ ext-\ **y**. It is a
replacement for conventional text recognizers in ATR system using conventional
baseline+bounding polygon (where it eliminates the need for bounding polygons)
and bounding box line data models. 

Party consists of a Swin vision transformer encoder, baseline positional
embeddings, and a `tiny Llama decoder
<https://github.com/mittagessen/bytellama>`_ trained on octet tokenization.

Installation
------------

::

        $ pip install .


Fine Tuning
-----------

Party needs to be trained on datasets precompiled from PageXML or ALTO files
containing line-wise transcriptions and baseline information for each line. The
binary dataset format is **NOT** compatible with kraken but the process of
compilation is fairly similar:

::

        $ party compile -o dataset.arrow *.xml

It is recommended not to enable BiDi reordering as the pretrained language
model and the base model have been trained to recognize RTL text in logical
order.

It is recommended to disable BiDi reordering as the pretrained model has been
trained to recognize RTL text in logical order.

To fine-tune the pretrained base model dataset files in listed in manifest
files on all available GPUs:

::

        $ party train --load-from-repo 10.5281/zenodo.14616981 --workers 32 -f train.lst -e val.lst

With the default parameters both baseline and bounding box prompts are randomly
sampled from the training data. It is suggested that you fine-tune the model
with uni-modal line embeddings by only selecting the line format that your
segmentation method produces, i.e.:

::

        $ party train --load-from-repo 10.5281/zenodo.14616981 -f train.lst -e val.lst --prompt-mode curves

or:

::

        $ party train --load-from-repo 10.5281/zenodo.14616981 -f train.lst -e val.lst --prompt-mode boxes

To continue training from an existing checkpoint 

::
        
        $ party train --load-from-checkpoint checkpoint_03-0.0640.ckpt -f train.lst -e val.lst


Checkpoint conversion
---------------------

Checkpoints need to be converted into a safetensors format before being usable
for inference and testing.

::

        $  party convert -o model.safetensors checkpoint.ckpt

Inference
---------

To recognize text in pre-segmented page images in PageXML or ALTO with the
pretrained model run:

::

        $ party -d cuda:0 ocr -i in.xml out.xml --load-from-repo 10.5281/zenodo.14616981

The paths to the image file(s) is automatically extracted from the XML input
file(s).

When the recognizer supports both curves and box prompts, curves are selected
by default. To select a prompt type explicitly you can use the `--curves` and
`--boxes` switches:

::

        $ party -d cuda:0 ocr -i in.xml out.xml --curves --compile
        $ party -d cuda:0 ocr -i in.xml out.xml --boxes --compile

Inference from a converted checkpoint:

::

        $ party -d cuda:0 ocr -i in.xml out.xml --curves --load-from-file model.safetensors

Testing
-------

Testing for now only works from XML files. As with for inference curve prompts
are selected if the model supports both, but an explicit line prompt type can
be selected.

::

        $  party -d cuda:0 test --curves --load-from-file arabic.safetensors  */*.xml
        $  party -d cuda:0 test --boxes --load-from-file arabic.safetensors  */*.xml
        $  party -d cuda:0 test --curves --load-from-repo 10.5281/zenodo.14616981 */*.xml
        $  party -d cuda:0 test --boxes --load-from-repo 10.5281/zenodo.14616981 */*.xml

Performance
-----------

Training and inference resource consumption is highly dependent on various
optimizations being enabled. Torch compilation which is required for various
attention optimizations is enabled per default but lower precision training
which isn't supported on CPU needs to be configured manually with `party
--precision bf16-true ...`.

Moderate speedups on CPU are possible with intra-op parallelism (`party
--threads 4 ocr ...`).

Quantization isn't yet supported.
