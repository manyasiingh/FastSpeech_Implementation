# FastSpeech 2 Text-to-Speech: English (Svarah) and Telugu (IndicTTS)

Two separate FastSpeech 2 models, one for English and one for Telugu, built on the public PyTorch implementation [ming024/FastSpeech2](https://github.com/ming024/FastSpeech2). Each model was trained on about 6 h of audio (about 5 h train, 1 h validation) for 30,000 steps. The repository contains preprocessing, training and inference code, configs, results (loss plots, test audio, variance-adaptor plots) and the technical report.

Full details of the approach, data, analysis and limitations are in `Report/Technical_Report.pdf`.

## Repository structure

```
.
├── English/
│   ├── preprocess.py, train.py, inference.py, config.yaml
│   └── checkpoints/
├── Telugu/
│   ├── preprocess.py, train.py, inference.py, config.yaml
│   └── checkpoints/
├── Results/
│   ├── English/  training_loss.png, validation_loss.png,
│   │             test_generated_audios/, variance_adaptor_plots/{duration,pitch,energy}/
│   └── Telugu/   (same layout)
├── Report/Technical_Report.pdf
├── README.md
└── requirements.txt
```

Each language folder is self-contained: the modified FastSpeech 2 source files (model, text frontend, utilities) sit alongside the scripts above.

## Datasets

| Language | Dataset | Selection |
|---|---|---|
| English | [ai4bharat/Svarah](https://huggingface.co/datasets/ai4bharat/Svarah) | Filtered by length and characters, shuffled (seed 42), up to 6 h; 3,984 utterances, 5.65 h used. Speaker labels are demographic proxies. |
| Telugu | [SPRINGLab/IndicTTS_Telugu](https://huggingface.co/datasets/SPRINGLab/IndicTTS_Telugu) | Female speaker only, filtered by length and script, shuffled (seed 42), 6.00 h; 1,864 utterances. |

Train and validation sets are split by utterance with no overlap (validation: 664 English, 311 Telugu utterances). Exact steps are in Report Section 3.

## Setup

1. Python environment: `pip install -r requirements.txt` (minimum versions: numpy 2.x, PyTorch 2.6 or newer; the original FastSpeech2 pins are not compatible with the modified code).
2. Montreal Forced Aligner (MFA), installed in its own environment, for phoneme alignment:
   - English: `mfa model download acoustic english_us_arpa` and `mfa model download dictionary english_us_arpa`
   - Telugu: the IndicMFA Telugu acoustic model and dictionary from [AI4Bharat/IndicMFA](https://github.com/AI4Bharat/IndicMFA), placed in `mfa_models/`
3. Vocoder: download the universal HiFi-GAN weights `generator_universal.pth.tar` from the [FastSpeech2 README](https://github.com/ming024/FastSpeech2) and place them in `hifigan/` (they are not stored in this repository).

## English: preprocess, train, synthesise

Run inside `English/`:

```bash
python prepare_data.py
mfa align raw_data/English english_us_arpa english_us_arpa preprocessed_data/English/TextGrid --clean -j 4
python make_config.py && python preprocess.py config.yaml
python train.py -c config.yaml
python inference.py -c config.yaml --restore_step 30000 --speaker_id <ID> --out ../Results/English
python plot_losses.py output/log/English ../Results/English
```

## Telugu: preprocess, train, synthesise

Run inside `Telugu/`:

```bash
python prepare_data.py && python build_dict.py
mfa align raw_data/Telugu mfa_models/dict_extended.txt mfa_models/Telugu_Acoustic_Model.zip \
    preprocessed_data/Telugu/TextGrid --clean --beam 100 --retry_beam 400 -j 4
python build_symbols.py && python make_config.py && python preprocess.py config.yaml
python train.py -c config.yaml
python inference.py -c config.yaml --restore_step 30000 --out ../Results/Telugu
python plot_losses.py output/log/Telugu ../Results/Telugu
```

`inference.py` synthesises the five test sentences per language (E1, E2a, E2b, E3, E4 and T1, T2a, T2b, T3, T4), writes the WAV files to `test_generated_audios/`, and saves predicted duration, pitch (F0) and energy plots to `variance_adaptor_plots/`.

## Checkpoints

Trained checkpoints are in `English/checkpoints/` and `Telugu/checkpoints/` (Git LFS, see `.gitattributes`).


## Main changes to the reference implementation

- Compatibility fixes for current numpy, PyTorch and librosa versions.
- Silence handling in the preprocessor for MFA 3 output.
- One config per language; new data preparation, config generation, loss plotting and inference scripts.
- English: speaker proxies from Svarah metadata, LibriSpeech lexicon plus g2p_en frontend.
- Telugu: Telugu phone set added to the symbol table, IndicMFA alignment, a generated word-level dictionary and a grapheme-based frontend.

## Results at a glance

| | English | Telugu |
|---|---|---|
| Final validation loss (step 30,000) | 3.3666 | 4.8623 |
| Lowest validation loss | 3.1089 (step 3,500) | 4.2522 (step 1,000) |
| Training time (Kaggle T4) | 3 h 55 min | 6 h 06 min |

English speech is clear at the start of each sentence but hard to follow without the transcript. Telugu speech sounds clear. Limitations, listening notes and variance-adaptor analysis are in the report.

## Acknowledgements and references

- FastSpeech 2: Ren et al., ICLR 2021, and the [ming024/FastSpeech2](https://github.com/ming024/FastSpeech2) implementation (see its repository for its licence).
- HiFi-GAN vocoder: Kong et al., NeurIPS 2020.
- Montreal Forced Aligner: McAuliffe et al., Interspeech 2017; IndicMFA Telugu by AI4Bharat.
- Datasets: Svarah (AI4Bharat) and IndicTTS Telugu.
