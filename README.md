# party

party is **PA**ge-wise **R**ecognition of **T**ext-**y**. It is a replacement for conventional text recognizers in ATR system using conventional baseline+bounding polygon (where it eliminates the need for bounding polygons) and bounding box line data models. 

Party consists of a Swin vision transformer encoder, baseline positional embeddings, and a [tiny Llama decoder](https://github.com/mittagessen/bytellama) trained on octet tokenization.

## Metrics

The base model has been pretrained on a very diverse collection of datasets in a dozen writing systems and even more languages, in addition to the language model decoder being trained on all 151 languages in the OSCAR corpus. No attempts have been made to adjust the frequency of particular data so character accuracy is fairly uneven across the corpus. 

The current base model's character accuracies on the validation set with curve and bounding box prompts (sorted by ascending curve error rate):

| Script    | Code Points | %Right (curves) | %Right (boxes) |
| :-------- | :---------- | :-------------- | :------------- |
| Han       | 107416      | 98.90%          | 98.88%         |  
| Hiragana  | 1868        | 97.11%          | 97.11%         |
| Cyrillic  | 22239       | 92.70%          | 92.34%         |
| Greek     | 1036        | 92.28%          | 91.31%         |
| Katakana  | 390         | 90.00%          | 90.00%         |
| Latin     | 199703      | 88.02%          | 86.98%         |
| Common    | 85863       | 80.24%          | 79.28%         |
| Arabic    | 18061       | 79.22%          | 79.64%         |
| Hebrew    | 40182       | 73.98%          | 73.97%         |
| Inherited | 2886        | 61.61%          | 60.95%         |
| Unknown   | 202         | 58.42%          | 57.43%         |

The script types are determined from the Unicode script property of each individual code point.

The base model has been trained on Georgian, Syriac, Newa, Malayalam, and Devanagari, albeit with fairly small datasets. No pages with these scripts are contained in the validation sample.

While the model performs quite well on languages and scripts that are commonly found in the training data, **it is generally expected that it requires fine-tuning for practical use, in particular to ensure alignment with desired transcription guidelines.**

## Installation

    $ pip install .

## Fine Tuning

Party needs to be trained on datasets precompiled from PageXML or ALTO files containing line-wise transcriptions and baseline information for each line. The binary dataset format is **NOT** compatible with kraken but the process of compilation is fairly similar:

        $ party compile -o dataset.arrow *.xml

To fine-tune the pretrained base model dataset files in listed in manifest files on all available GPUs:

        $ party train --load-from-repo 10.5281/zenodo.15073482 --workers 32 -t train.lst -e val.lst

With the default parameters both baseline and bounding box prompts are randomly sampled from the training data. It is suggested that you fine-tune the model with uni-modal line embeddings by only selecting the line format that your segmentation method produces, i.e.:

        $ party train --load-from-repo 10.5281/zenodo.15073482 -t train.lst -e val.lst --prompt-mode curves

or:

        $ party train --load-from-repo 10.5281/zenodo.15073482 -t train.lst -e val.lst --prompt-mode boxes

To continue training from an existing checkpoint:

        $ party train --load-from-checkpoint checkpoint_03-0.0640.ckpt -t train.lst -e val.lst


## Checkpoint conversion

Checkpoints need to be converted into a safetensors format before being usable for inference and testing.

        $  party convert -o model.safetensors checkpoint.ckpt

## Inference

Inference and teseting requires a working [kraken](https://kraken.re) installation.

To recognize text in pre-segmented page images in PageXML or ALTO with the pretrained model run:

        $ party -d cuda:0 ocr -i in.xml out.xml --load-from-repo 10.5281/zenodo.15073482

The paths to the image file(s) is automatically extracted from the XML input file(s).

When the recognizer supports both curves and box prompts, curves are selected by default. To select a prompt type explicitly you can use the `--curves` and `--boxes` switches:

        $ party -d cuda:0 ocr -i in.xml out.xml --curves --compile
        $ party -d cuda:0 ocr -i in.xml out.xml --boxes --compile

Inference from a converted checkpoint:

        $ party -d cuda:0 ocr -i in.xml out.xml --curves --load-from-file model.safetensors

## Testing

Testing for now only works from XML files. As with for inference curve prompts are selected if the model supports both, but an explicit line prompt type can be selected.

        $  party -d cuda:0 test --curves --load-from-file arabic.safetensors  */*.xml
        $  party -d cuda:0 test --boxes --load-from-file arabic.safetensors  */*.xml
        $  party -d cuda:0 test --curves --load-from-repo 10.5281/zenodo.15073482 */*.xml
        $  party -d cuda:0 test --boxes --load-from-repo 10.5281/zenodo.15073482 */*.xml

## Performance

Training and inference resource consumption is highly dependent on various optimizations being enabled. Torch compilation which is required for various attention optimizations is enabled per default but lower precision training which isn't supported on CPU needs to be configured manually with `party --precision bf16-mixed ...`. It is possible to reduce training memory requirements substantially by freezing the visual encoder with the `--freeze-encoder` option.

Moderate speedups on CPU are possible with intra-op parallelism (`party --threads 4 ocr ...`).

Quantization isn't yet supported.
