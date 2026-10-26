# Tunisian Derja ASR: Whisper-small fine-tune + low-latency inference

Speech recognition for **Tunisian Arabic (Derja)**, including the French and English words mixed into everyday speech.

- [`openai/whisper-small`](https://huggingface.co/openai/whisper-small), fully fine-tuned on the LinTO Tunisian datasets in one 12-hour Kaggle session (2× T4).
- Evaluated on a leakage-safe held-out test set.
- Exported to CTranslate2 for fast GPU/CPU inference and live microphone transcription.

| | |
|---|---|
| 🤗 Transformers model | [TheBiggyBang/whisper-small-tunisian-derja](https://huggingface.co/TheBiggyBang/whisper-small-tunisian-derja) |
| 🤗 faster-whisper / CTranslate2 model | [TheBiggyBang/whisper-small-tunisian-derja-ct2](https://huggingface.co/TheBiggyBang/whisper-small-tunisian-derja-ct2) |
| Training + evaluation notebook | [`asr-i-hope-final.ipynb`](asr-i-hope-final.ipynb) (outputs included) |
| Detailed notes (data, citations, every metric) | [`note.md`](note.md) |

## Results

Held-out test set: 482 clips (36.6 min) from all six LinTO test splits. It shares no recordings with training data or checkpoint selection.

| Model / decoder | WER | normalized WER (95 % CI) | CER | RTF |
|---|---|---|---|---|
| whisper-small, no fine-tuning (CT2, beam 3) | 79.1 | 76.3 (73.8–79.0) | 44.5 | 0.078 |
| **fine-tuned (CT2 int8, beam 3)** | **30.4** | **29.5** (27.5–31.6) | **12.0** | **0.044** |
| fine-tuned (Transformers fp16, greedy) | 32.8 | 31.9 (29.9–34.2) | 13.4 | 0.025 |
| fine-tuned (CT2 int8, CPU 4 threads, greedy, 80-clip subset) | 31.0 | 30.3 | 13.2 | 0.889 |
| [LinTO Kaldi/Vosk](https://huggingface.co/linagora/linto-asr-ar-tn-0.1) (4-gram LM, 400-clip subset) | 25.4 | 25.0 (22.0–28.1) | 15.7 | 0.150 |

- **What the numbers say:**
  - Fine-tuning cuts normalized WER from 76.3 to 29.5 (−61 % relative).
  - Fine-tuning cuts CER from 44.5 to 12.0 (−73 % relative).
  - Against the Kaldi/Vosk model, this one makes fewer character errors but more word errors.
- **Normalized WER** removes spelling variants that Derja has no rules for (أ/إ/آ→ا, ى→ي, ة→ه, digits, punctuation, Latin case, standalone «و»). It is applied identically to references and hypotheses.
- **CIs** come from a cluster bootstrap over recordings. **RTF** = processing time / audio length.

## Quick start

```bash
pip install faster-whisper
```

```python
from faster_whisper import WhisperModel

model = WhisperModel("TheBiggyBang/whisper-small-tunisian-derja-ct2",
                     device="cuda", compute_type="int8_float16")   # CPU: device="cpu", compute_type="int8"
segments, _ = model.transcribe("audio.wav", language="ar", beam_size=3,
                               condition_on_previous_text=False, vad_filter=True)
print(" ".join(s.text.strip() for s in segments))
```

### Live microphone transcription

```bash
pip install faster-whisper "av<19" sounddevice soundfile numpy
hf download TheBiggyBang/whisper-small-tunisian-derja-ct2 --local-dir whisper-tn-ct2
python live_asr.py --model whisper-tn-ct2                                   # microphone (GPU)
python live_asr.py --model whisper-tn-ct2 --file talk.wav                   # replay a file in real time
python live_asr.py --model whisper-tn-ct2 --device cpu --compute-type int8 --no-partials
```

[`live_asr.py`](live_asr.py) is turn-based pseudo-streaming:
- Silero VAD finds speech.
- Partial text is re-decoded every 0.6 s, and words that two decodes agree on are fixed.
- The turn is finalized with beam search after 500 ms of silence.

On a T4 GPU, first words appear after about 1–1.4 s and the final text 0.4–0.8 s after you stop talking. On a 4-thread CPU, use `--no-partials`.

## How it was built

The notebook runs end-to-end on Kaggle. Each step is a section in [`asr-i-hope-final.ipynb`](asr-i-hope-final.ipynb):

1. **Data.** The LinTO original (40 %) and augmented (60 %) training sets are streamed from the Hugging Face Hub (no full download). Each segment is validated: timestamps inside the audio, 0.6–29 s long, a plausible characters-per-second rate, and not silent. At most 4 segments per recording are kept.
2. **Leakage-safe evaluation.** All original test splits are pooled and split **by recording** into dev (323 clips) and test (482 clips). Every evaluation sentence is block-listed from training. The test set is used exactly once, after training.
3. **Pilot.** Zero-shot comparison of `whisper-base` and `whisper-small` on dev.
4. **Budgeted training.** A throughput benchmark measures seconds per step, then the step count is set to fit the 12 h session. Full fine-tuning: lr 1e-5, cosine schedule, effective batch 32, fp16, SpecAugment, gain/noise augmentation. Checkpoints are kept by dev normalized WER.
5. **Export.** CTranslate2 `int8_float16` (250 MB).
6. **Test evaluation.** Base vs fine-tuned, HF vs CT2, greedy vs beam, a CPU check, and the Kaldi/Vosk baseline. Includes bootstrap CIs and error analysis.
7. **Streaming benchmark.** Latency of `live_asr.py` against a virtual real-time clock.

### Reproduce

1. Upload the notebook to Kaggle. Set accelerator **GPU T4 × 2** and Internet **on**. Optionally add an `HF_TOKEN` secret.
2. Keep `RUN_MODE = "full"`, or use `"smoke"` for a ~30-minute pipeline check.
3. **Save Version → Run All.** The full run takes about 11 h.

## Repository layout

```
asr-i-hope-final.ipynb   training + evaluation notebook (with outputs)
live_asr.py              low-latency streaming engine and CLI
note.md                  detailed notes: data sources, citations, all metrics
```

Model weights are on the Hugging Face Hub, not in this repo.

## Limitations

- **Speaker overlap is possible.** The test set is recording-disjoint but not speaker-disjoint, because the data has no speaker IDs.
- **Spelling.** Derja has no standard orthography. Near-miss spelling is the largest error category, so report CER alongside WER.
- **Code-switching.** French/English words may come out in a different script than the annotator used.
- **Small test set.** 36.6 min, with only 12–15 clips for some sub-corpora.
- **Not a true streaming model.** Each partial is a full 30 s Whisper encoder pass.
- **Limited data coverage.** Training was time-limited (4,071 steps) and saw only about 131 k segments because of the 4-per-recording cap. There is room to improve.

## Data and attribution

Training and evaluation use the [LinTO DataSet Audio for Arabic Tunisian](https://huggingface.co/datasets/linagora/linto-dataset-audio-ar-tn) and its [augmented version](https://huggingface.co/datasets/linagora/linto-dataset-audio-ar-tn-augmented) by LINAGORA, licensed under [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/). These include TunSwitch (Abdallah et al., 2023), MASC (Al-Fetyani et al., 2021) and other community and web sources listed on the dataset cards. No audio or transcripts are redistributed in this repository. Base model: [openai/whisper-small](https://huggingface.co/openai/whisper-small) (Apache-2.0).

## Citation

If you use this work, please cite the datasets and base model:

```bibtex
@misc{linagora2024Linto-tn,
  title = {LinTO Audio and Textual Datasets to Train and Evaluate Automatic Speech Recognition in Tunisian Arabic Dialect},
  author = {Hedi Naouara and Jérôme Louradour and Jean-Pierre Lorré},
  year = {2025},
  month = {March},
  eprint={2504.02604},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  note={Good Data Workshop, AAAI 2025},
  url={arxiv.org/abs/2504.02604},
}

@misc{abdallah2023leveraging,
      title={Leveraging Data Collection and Unsupervised Learning for Code-switched Tunisian Arabic Automatic Speech Recognition},
      author={Ahmed Amine Ben Abdallah and Ata Kabboudi and Amir Kanoun and Salah Zaiem},
      year={2023},
      eprint={2309.11327},
      archivePrefix={arXiv},
      primaryClass={eess.AS}
}

@data{e1qb-jv46-21,
doi = {10.21227/e1qb-jv46},
url = {https://dx.doi.org/10.21227/e1qb-jv46},
author = {Al-Fetyani, Mohammad and Al-Barham, Muhammad and Abandah, Gheith and Alsharkawi, Adham and Dawas, Maha},
publisher = {IEEE Dataport},
title = {MASC: Massive Arabic Speech Corpus},
year = {2021} }

@misc{radford2022whisper,
  doi = {10.48550/ARXIV.2212.04356},
  url = {https://arxiv.org/abs/2212.04356},
  author = {Radford, Alec and Kim, Jong Wook and Xu, Tao and Brockman, Greg and McLeavey, Christine and Sutskever, Ilya},
  title = {Robust Speech Recognition via Large-Scale Weak Supervision},
  publisher = {arXiv},
  year = {2022},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```

## License

Code and model weights: Apache-2.0. Training data: CC-BY-4.0 (LINAGORA). See the attribution above.
