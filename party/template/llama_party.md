---
# Example metadata to be added to a model card.
id: {id_0}  # Example: https://doi.org/10.5281/zenodo.14399779
summary: Pretrained multilingual Party model 
authors:
  - name: Benjamin Kiessling
    affiliation: École Pratique des Hautes Études, PSL University
license: Apache-2.0
software_name: party
software_hints:
- segmentation=both
language:
- ang
- ara
- cat
- cmn
- cos
- ces
- chu
- deu
- eng
- fas
- fin
- fra
- grc
- hbo
- ita
- jpn
- kat
- lat
- mal
- new
- nld
- ota
- por
- rus
- san
- spa
- syr
- urd
- ukr
- yid
script:
- Arab
- Aran
- Cyrl
- Cyrs
- Deva
- Geor
- Glag
- Grek
- Hant
- Hebr
- Latn
- Latf
- Mlym
- Newa
- Syrc
tags:
- automatic-text-recognition
- multilingual
- multiscriptal
- multimodal
model_type:
- recognition
metrics:
  cer: 0.00
  wer: 0.00
base_model:
- https://huggingface.co/timm/swin_base_patch4_window12_384.ms_in22k
- https://github.com/mittagessen/bytellama
---
# Llama Party

Party is *pa*ge-wise *r*ecognition of *t*ext-*y*. It is a replacement for
conventional text recognizers in automatic text recognition pipelines that
utilize either bounding box or baseline+bounding polygon segmentation methods
for layout analysis.

Llama party is a full-page generative text recognizer that has been pretrained
on a large corpus of multilingual historical, contemporary, and born-digital
document page images, both handwritten and machine-printed.

## Architecture

The recognizer is a deep fusion multimodal model consisting of a Swin vision
encoder and a tiny Llama (100M parameters) decoder trained with octet
tokenization. The network is prompted with the line positions through
positional embeddings added to the encoder hidden state. 

During training the encoder weights were initialized with a ImageNet-22k
pretrained Swin-base from pytorch-image-models, the decoder weights came from a
custom [Llama 3.2](https://github.com/mittagessen/bytellama) pretrained on a
subset of [OSCAR 2301](https://huggingface.co/datasets/oscar-corpus/OSCAR-2301)
tokenized with a ByT5-style octet tokenizer.

The pre-initialized model was then pre-trained on a collection of public and
private training historical document page datasets augmented with born-digital
data crafted from [PubLayNet](https://github.com/ibm-aur-nlp/PubLayNet).

## Uses

Llama party is a recognition foundation model primarily targeted at automatic
text recognition for the humanities. While it produces fairly accurate output
on an impressive range of material it is intended to be fine-tuned on some
target dataset to ensure compliance with desired transcription guidelines.

## Transcription guidelines, Normalization, and Transformations 

No attempts have been made to normalize the datasets or to only use data
adhering to common transcription guidelines. While some subsets of the corpus
are internally consistent, only a very small subset of the languages in the
training data only contain datasets from a single source.

## Bias, Risks, and Limitations

The training corpus is heavily skewed towards a couple of languages (Chinese,
English, French, German, and Portuguese) and frequently incorporates datasets
of esoteric material transcribed for specific purposes. Especially
machine-printed and born-digital material lack diversity, so error rates will
most likely vary considerably across languages and document type.

Some additional limitations are to be expected:

- Some of the training data are Latin transliterations of Yiddish and Ottoman
  Turkish. Arabic and Hebrew script recognition is likely to require
  fine-tuning.
- Some transcriptions resolved abbreviations while others did not. Inconsistent
  output is to be expected, in particular for European manuscripts in Latin
  script.
- As the model predicts 8-bit UTF-8 code units directly the lack of consistent
  Unicode normalization can cause slightly different code point streams during
  prediction.

## How to Get Started with the Model

Install the `party` package from [github](https://github.com/mittagessen/party) and follow the instructions.

## Training Details

### Training Data

The model has been pretrained on the vast majority of publicly available ATR
datasets, in addition to a decent number of restricted datasets. For English
exclusively we converted the PubLayNet dataset for layout analysis on
born-digital documents into an ATR dataset with
[PDFMiner](https://pdfminersix.readthedocs.io/en/latest/) and some basic
baseline heuristic based on the line bounding box.

|Language|Pages|Lines|Datasets|
|---|---|---|---|
|Arabic|   |   |   |[RASAM 1](https://github.com/calfa-co/rasam-dataset)<br>[TariMa](https://github.com/calfa-co/tarima)<br>[OpenITI Arabic MS Data](https://github.com/OpenITI/arabic_ms_data)<br>[OpenITI Arabic Print Data](https://github.com/OpenITI/arabic_print_data)
|Catalan|   |   |   |[FONDUE-CA-PRINT-20](https://github.com/FoNDUE-HTR/FONDUE-CA-PRINT-20)
|Chinese|   |   |   |1 large private dataset
|Corsican|   |   |   |[HN2021-OCR-Poesie-Corse](https://github.com/PSL-Chartes-HTR-Students/HN2021-OCR-Poesie-Corse)
|Czech|   |   |   |[Padeřov-Bible-handwriting-ground-truth](https://zenodo.org/records/7467034)
|Dutch|   |   |   |[ATR_TrainingSet_NLF_Newseye_GT_SV_M2+]()<br>4 private manuscript datasets<br>[VOC dataset](https://zenodo.org/records/11209325)
|English|   |   |   |[FONDUE-EN-PRINT-20](https://github.com/FoNDUE-HTR/FONDUE-EN-PRINT-20)<br>[PubLayNet](https://github.com/ibm-aur-nlp/PubLayNet)<br>[University of Denver Collections](https://zenodo.org/records/4243023)<br>[Joseph Hooker HTR](https://github.com/jschaefer738b/JosephHookerHTR)<br>[CCCC MS 41]()
|Finnish|   |   |   |[NewsEye/READ OCR Finnish Newspapers](https://zenodo.org/records/4599472)
|French|   |   |   |[NewsEye READ AS French Newspapers](https://zenodo.org/records/5654841)<br>[Boccace](https://github.com/PSL-Chartes-HTR-Students/HN2021-Boccace)<br>[Fabliaux](https://github.com/CIHAM-HTR/Fabliaux)<br>[Liber](https://github.com/CIHAM-HTR/Liber)<br>[Cremma Medieval](https://github.com/HTR-United/cremma-medieval)<br>[DecameronFR](https://github.com/PSL-Chartes-HTR-Students/TNAH-2021-DecameronFR)<br>[FONDUE-FR-MSS-18](https://github.com/FoNDUE-HTR/FONDUE-FR-MSS-18)<br>[FONDUE-FR-MSS-19](https://github.com/FoNDUE-HTR/FONDUE-FR-MSS-19)<br>[FONDUE-FR-PRINT-16](https://github.com/FoNDUE-HTR/FONDUE-FR-PRINT-16)<br>[FONDUE-FR-PRINT-17](https://github.com/FoNDUE-HTR/FONDUE-FR-PRINT-17)<br>[FONDUE-FR-PRINT-20](https://github.com/FoNDUE-HTR/FONDUE-FR-PRINT-20)<br>[Données imprimés gothiques du 16e siècle](https://github.com/Gallicorpora/HTR-imprime-16e-siecle)<br>[Données HTR incunables du 15e siècle](https://github.com/Gallicorpora/HTR-incunable-15e-siecle)<br>[Données HTR manuscrits du 15e siècle](https://github.com/Gallicorpora/HTR-MSS-15e-Siecle)<br>["Tables Décennales" French Civil Registry](https://github.com/jpmjpmjpm/genauto-td-htr.git)<br>[Données imprimés du 16e siècle](https://github.com/Gallicorpora/HTR-imprime-16e-siecle)<br>[Données imprimés du 17e siècle](https://github.com/Gallicorpora/HTR-imprime-17e-siecle)<br>[Données imprimés du 18e siècle](https://github.com/Gallicorpora/HTR-imprime-18e-siecle)<br>[Incunable français du 15e siècle](https://github.com/Gallicorpora/HTR-incunable-15e-siecle)<br>[HTRomance](https://github.com/HTRomance-Project/medieval-french)<br>[HTR-SETAF-Jean-Michel](https://github.com/SETAFDH/HTR-SETAF-Jean-Michel)<br>[HTR-SETAF-LesFaictzJCH](https://github.com/SETAFDH/HTR-SETAF-LesFaictzJCH)<br>[HTR-SETAF-Pierre-de-Vingle](https://github.com/SETAFDH/HTR-SETAF-Pierre-de-Vingle)<br>[La Correspondance Jacques Doucet - René Jean](https://gitlab.inha.fr/snr/LaCorrespondanceDoucetReneJean)<br>[OCR17+](https://github.com/e-ditiones/OCR17plus)<br>[Tapus Corpus](https://github.com/HTR-United/tapuscorpus)<br>[TIMEUS Corpus](https://github.com/HTR-United/timeuscorpus)<br>[Recensement Valaisan](https://github.com/PonteIneptique/valais-recensement)<br>3 private handwritten and print datasets
|Georgian|   |   |   |1 private dataset
|German|   |   |   |[Charlottenburger Amtsschrifttum](https://github.com/UB-Mannheim/charlottenburger-amtsschrifttum.git)<br>[DACH GT](https://github.com/UB-Mannheim/dach-gt.git)<br>[DigiTue GT](https://github.com/UB-Mannheim/digitue-gt.git)<br>[Fibeln](https://github.com/UB-Mannheim/Fibeln.git)<br>[FONDUE-DE-MSS-18](https://github.com/FoNDUE-HTR/FONDUE-DE-MSS-18)<br>[FoNDUE_Wolfflin_Fotosammlung](https://github.com/FoNDUE-HTR/FoNDUE_Wolfflin_Fotosammlung)<br>[HKB GT](https://github.com/UB-Mannheim/hkb-gt.git)<br>[Ground truth for Neue Zürcher Zeitung black letter](https://zenodo.org/records/3333627)<br>[Reichsanzeiger GT](https://github.com/UB-Mannheim/reichsanzeiger-gt.git)<br>[StABS Ratsbücher O10](https://zenodo.org/records/5153263)<br>[NewsEye / READ OCR Austrian Newspapers](https://zenodo.org/records/3387369)<br>[Weisthuemer](https://github.com/UB-Mannheim/Weisthuemer.git)<br>3 private manuscript datasets
|Greek|   |   |   |[EPARCHOS](https://zenodo.org/records/4095301)<br>[HTR CPgr23](https://gitlab.huma-num.fr/ecrinum/anthologia/htr_cpgr23)<br>[Handwritten Paleographic Greek Text Recognition](https://github.com/vivianpl/HPGTR.git)<br>[ΧΦ114](https://zenodo.org/records/5578251)<br>[XΦ79](https://zenodo.org/records/5578136)<br>[ΧΦ53](https://zenodo.org/records/5595669)<br>10 small private manuscript datasets
|Hebrew|   |   |   |[Tikkoun Sofrim](https://dataverse.nl/dataset.xhtml?persistentId=hdl:10411/RTTB3C)<br>[BiblIA](https://zenodo.org/records/5167263)
|Italian|   |   |   |[episearch-htr](https://github.com/vedph/episearch-htr)<br>[FONDUE-IT-PRINT-20](https://github.com/FoNDUE-HTR/FONDUE-IT-PRINT-20)<br>[HTRomance Italian](https://github.com/HTRomance-Project/medieval-italian)<br>1 private print dataset
|Japanese|   |   |   |[mm-ocr-dataset-v1]()
|Latin|   |   |   |[Caroline Minuscule](https://github.com/rescribe/carolineminuscule-groundtruth)<br>[CREMMA-Medieval-LAT](https://github.com/HTR-United/CREMMA-Medieval-LAT)<br>[HTRomance Latin](https://github.com/HTRomance-Project/medieval-latin)<br>[DIVA-HisDB](https://diuf.unifr.ch/main/hisdoc/diva-hisdb.html)<br>[Eutyches](https://github.com/malamatenia/Eutyches)<br>[FONDUE-LA-MSS-MA](https://github.com/FoNDUE-HTR/FONDUE-LA-MSS-MA)<br>[FONDUE-LA-PRINT-16](https://github.com/FoNDUE-HTR/FONDUE-LA-MSS-16)<br>[Lateinische Gedichte](https://zenodo.org/records/4780947)<br>[Wien ÖNB Cod 2160](https://zenodo.org/records/7537204)<br>2 private manuscript datasets
|Multilingual|   |   |   |[FONDUE-MLT-ART](https://github.com/FoNDUE-HTR/FONDUE-MLT-ART)<br>[[FONDUE-MLT-CAT](https://github.com/FoNDUE-HTR/FONDUE-MLT-CAT)<br>[FONDUE-MLT-PRINT-TEST](https://github.com/FoNDUE-HTR/FONDUE-MLT-PRINT-TEST)<br>gt_structure_text](https://github.com/OCR-D/gt_structure_text)
|Ottoman Turkish|   |   |   |[OpenITI Arabic MS Data](https://github.com/OpenITI/arabic_ms_data)<br>[OpenITI Arabic Print Data](https://github.com/OpenITI/arabic_print_data)<br>1 private manuscript dataset
|Farsi|   |   |   |[OpenITI Arabic MS Data](https://github.com/OpenITI/arabic_ms_data)<br>[OpenITI Arabic Print Data](https://github.com/OpenITI/arabic_print_data)
|Portuguese|   |   |   |[Portuguese Handwriting 16th-19th c.](https://zenodo.org/records/13986218)
|Russian|  |  |  |  |1 private manuscript dataset
|Spanish|   |   |   |[FONDUE-ES-PRINT-19](https://github.com/FoNDUE-HTR/FONDUE-ES-PRINT-19)<br>[FoNDUE-Spanish-chapbooks-Dataset](https://github.com/FoNDUE-HTR/FoNDUE-Spanish-chapbooks-Dataset)<br>[HTR Araucania](https://zenodo.org/records/7075186)<br>[HTRomance Spa](https://github.com/HTRomance-Project/middle-ages-in-spain)<br>3 private manuscript datasets
|Syriac|   |   |   |2 private print and manuscript datasets
|Urdu|   |   |   |[OpenITI Arabic MS Data](https://github.com/OpenITI/arabic_ms_data)<br>[OpenITI Arabic Print Data](https://github.com/OpenITI/arabic_print_data)
|Yiddish|   |   |   |1 private print datasets

### Training Procedure and Hyperparameters

- **Training regime:**: 6 * A40 GPU, BF16 precision, Mars-AdamW optimizer with caution, batch size: 32, gradient accumulation: 4, effective batch size: 768, 5 epochs with 5000 iteration warmup and cosine decay, max LR 5e-4, min LR 5e-6 at end of epoch 5, weight decay 1e-5, gradient clipping 1.0, augmentation, random sampling of bbox and curve batches

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

<!-- This should link to a Dataset Card if possible. -->

{{ testing_data | default("[More Information Needed]", true)}}

#### Factors

<!-- These are the things the evaluation is disaggregating by, e.g., subpopulations or domains. -->

{{ testing_factors | default("[More Information Needed]", true)}}

#### Metrics

CER:
WER: 

#### Summary

{{ results_summary | default("", true) }}

## Citation [optional]

**BibTeX:**

{{ citation_bibtex | default("[More Information Needed]", true)}}
