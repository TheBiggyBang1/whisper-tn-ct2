# Tunisian Derja ASR v3: project notes

Fine-tuned `openai/whisper-small` for Tunisian Arabic (Derja) with French/English code-switching.
Every number below comes from the saved outputs of `asr-i-hope-final.ipynb` (Kaggle, full run).
The notebook was not re-run.

---

## 1. At a glance

| | |
|---|---|
| **Base model** | `openai/whisper-small` (241.7 M params), full fine-tuning |
| **Headline result** (held-out test, 482 clips / 36.6 min) | **WER 30.4 %**, normalized WER **29.5 %** (95 % CI 27.5–31.6), **CER 12.0 %** |
| Same decoder, base model (no fine-tuning) | WER 79.1 %, normalized WER 76.3 %, CER 44.5 % |
| Deployed decoder | CTranslate2 `int8_float16`, beam 3 (faster-whisper) |
| Exports | Transformers fp32 safetensors (967 MB) + CTranslate2 int8 (250 MB) |
| Training | 2× Tesla T4 (Kaggle), 4,071 steps × 32 segments, 10.34 h |
| Licenses involved | Whisper: Apache-2.0 · LinTO datasets: **CC-BY-4.0 (attribution required)** |

---

## 2. Data sources and citations

### 2.1 What was used, and what each source asks for

The notebook streams the **`default` config** of both LinTO datasets, which is the union of **all** their sub-corpora.
So every sub-corpus below was part of training.

| Source | Used for | License | Citation requested? |
|---|---|---|---|
| [linagora/linto-dataset-audio-ar-tn](https://huggingface.co/datasets/linagora/linto-dataset-audio-ar-tn) | Training (`default/train`, 40 % sampling weight) **and** evaluation (all 6 `test` splits) | CC-BY-4.0 | **Yes.** Cite Naouara et al. 2025. Card also asks you to "consider citing" TunSwitch and MASC. |
| [linagora/linto-dataset-audio-ar-tn-augmented](https://huggingface.co/datasets/linagora/linto-dataset-audio-ar-tn-augmented) | Training (`default/train`, 60 % sampling weight) | CC-BY-4.0 | **Yes.** Same as above. |
| [openai/whisper-small](https://huggingface.co/openai/whisper-small) | Base model | Apache-2.0 | **Yes.** Card gives BibTeX for Radford et al. |
| [linagora/linto-asr-ar-tn-0.1](https://huggingface.co/linagora/linto-asr-ar-tn-0.1) (Kaldi/Vosk) | Comparison baseline only | Apache-2.0 | **Yes.** Same paper as the datasets (Naouara et al.). |

**Sub-corpora inside the LinTO datasets:**

| Sub-corpus | Origin | License / status | Citation |
|---|---|---|---|
| TunSwitchCS, TunSwitchTO | [Zenodo 8370566](https://zenodo.org/records/8370566) | CC-BY-4.0 | **Yes.** The Zenodo record asks you to cite Abdallah et al. 2023 (arXiv:2309.11327). |
| MASC | [IEEE DataPort](https://ieee-dataport.org/open-access/masc-massive-arabic-speech-corpus), doi:10.21227/e1qb-jv46 | Open access (login required) | LinTO asks to consider citing it, so cite it. |
| AmenyKH | HF `amenIKh/dataset1` (**no longer on the Hub**) | not stated | None requested; credit by name. |
| Wav2Vec-tunisian-Darja | HF `medaminekonyali/Value-Wav2Vec-tunisian-Darja-Augmented` (**no longer on the Hub**) | not stated | None requested; credit by name. |
| Tunisian_dataset_STT-TTS15s_filtred1.0 | HF [Arbi-Houssem/…](https://huggingface.co/datasets/Arbi-Houssem/Tunisian_dataset_STT-TTS15s_filtred1.0) | card has no license | None requested; credit by name. |
| ApprendreLeTunisien | apprendreletunisien.com (website) | not stated | None requested. |
| OneStory | onestory-media.org (website) | not stated | None requested. |
| Youtube_* (11 subsets) | YouTube channels | rights belong to the channel owners | None requested. |

LinTO re-licensed the whole collection under CC-BY-4.0. For the rows marked "not stated", that is the only license you can point to.

### 2.2 What CC-BY-4.0 requires from you

- Give credit to LinTO (and ideally the original corpora), link the license, and say that you changed the material (you trained a model on it).
- Commercial use and derivatives are allowed.
- Weights released under **Apache-2.0** (same as Whisper) plus an attribution section is the usual pattern for this kind of model.

### 2.3 Attribution text (ready to paste into README / model card)

> This model was fine-tuned from [openai/whisper-small](https://huggingface.co/openai/whisper-small) (Apache-2.0) on the
> [LinTO DataSet Audio for Arabic Tunisian](https://huggingface.co/datasets/linagora/linto-dataset-audio-ar-tn) and its
> [augmented version](https://huggingface.co/datasets/linagora/linto-dataset-audio-ar-tn-augmented) by LINAGORA, licensed under
> [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/). These datasets include TunSwitch (Abdallah et al., 2023), MASC
> (Al-Fetyani et al., 2021), and other community and web sources listed on the dataset cards. The audio and transcripts were
> segmented, filtered and used for training; no data is redistributed with this model.

### 2.4 BibTeX

Required: the LinTO datasets and the LinTO Vosk baseline both ask for this paper.
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
```

Required: the TunSwitch Zenodo record asks for it, and LinTO recommends it.
```bibtex
@misc{abdallah2023leveraging,
      title={Leveraging Data Collection and Unsupervised Learning for Code-switched Tunisian Arabic Automatic Speech Recognition},
      author={Ahmed Amine Ben Abdallah and Ata Kabboudi and Amir Kanoun and Salah Zaiem},
      year={2023},
      eprint={2309.11327},
      archivePrefix={arXiv},
      primaryClass={eess.AS}
}
```

Recommended by LinTO: MASC.
```bibtex
@data{e1qb-jv46-21,
doi = {10.21227/e1qb-jv46},
url = {https://dx.doi.org/10.21227/e1qb-jv46},
author = {Al-Fetyani, Mohammad and Al-Barham, Muhammad and Abandah, Gheith and Alsharkawi, Adham and Dawas, Maha},
publisher = {IEEE Dataport},
title = {MASC: Massive Arabic Speech Corpus},
year = {2021} }
```

Required by the Whisper model card: the base model.
```bibtex
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

Optional: Silero VAD (MIT). `live_asr.py` uses it through faster-whisper.
Its repo shows a citation; copy the exact block from github.com/snakers4/silero-vad and put the commit you used in its `commit` field.
```bibtex
@misc{SileroVAD,
  author = {Silero Team},
  title = {Silero VAD: pre-trained enterprise-grade Voice Activity Detector (VAD), Number Detector and Language Classifier},
  year = {2024},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/snakers4/silero-vad}}
}
```

### 2.5 Software used (no citation requested)

| Package | Version on Kaggle | License |
|---|---|---|
| PyTorch | 2.11.0+cu128 | BSD-3 |
| transformers | 4.57.6 | Apache-2.0 |
| datasets | 4.8.5 | Apache-2.0 |
| CTranslate2 / faster-whisper | ≥4.4 / ≥1.1.1 | MIT |
| jiwer | ≥3.0 | Apache-2.0 |
| vosk (baseline only) | latest at run time | Apache-2.0 |

---

## 3. Model

| Property | Value |
|---|---|
| Architecture | Whisper encoder–decoder, 12 + 12 layers, d_model 768, 12 heads, FFN 3072 |
| Input | 80 log-Mel bins, 16 kHz, ≤ 30 s window |
| Vocabulary | 51,865 (Whisper multilingual tokenizer, unchanged) |
| Prompt | `language=arabic`, `task=transcribe`, no timestamps |
| Output text style | Dialect spelling kept, French/English words kept (Latin or Arabic script, as annotated), punctuation kept, diacritics removed |
| Output projection | Tied to decoder embeddings (verified: `Tied output projection: True`). The `missing keys ['proj_out.weight']` warning in the training log is therefore harmless. |

### Exported artifacts

| Zip | Files | Size | Load with |
|---|---|---|---|
| `whisper-tn-v3-hf.zip` | `model.safetensors`, `config.json`, `generation_config.json`, tokenizer files, `preprocessor_config.json`, `run_plan.json`, `training_args.bin` | 967 MB weights | `transformers` (`WhisperForConditionalGeneration`) |
| `whisper-tn-v3-ct2.zip` | `model.bin`, `config.json`, `vocabulary.json`, `tokenizer.json`, `preprocessor_config.json`, `live_asr.py` | 250 MB | `faster-whisper` (`WhisperModel`) |

`training_args.bin` is a Python pickle. It is not needed for inference; leave it out of public uploads.

---

## 4. Training

### 4.1 Setup

| Setting | Value |
|---|---|
| Hardware | Kaggle, 2× Tesla T4 (DataParallel), 4 vCPU, 3 data workers |
| Data mix | 60 % augmented LinTO + 40 % original LinTO, streamed (no full download) |
| Effective batch | 32 = micro-batch 8 × grad-accum 2 × 2 GPUs |
| Optimizer | AdamW (fused), lr 1e-5, cosine schedule, 300 warm-up steps, weight decay 0.01, grad-clip 1.0 |
| Precision | fp16 mixed precision, no gradient checkpointing (peak 13.3 GB on GPU 0 in the benchmark) |
| Augmentation | SpecAugment (time mask p = 0.05, length 10); on 30 % of clips a random gain of ±6 dB, and on half of those also coloured noise at SNR 15–30 dB |
| Segment filter | 0.6–29 s, 2–45 chars/s, ≤ 220 label tokens, timestamps inside the audio, non-silent |
| Per-recording cap | ≤ 4 segments per recording (decorrelates long recordings) |
| Seed | 42 |

### 4.2 Run

- Planned **4,272 steps** from a measured 8.44 s/step. The time budget stopped training at **step 4,071**.
- Training wall time **10.34 h**; whole session **11.12 h** of Kaggle's 12 h.
- Segments consumed ≈ 4,071 × 32 = **130,272** (stream accepted 131,204).
- **Best checkpoint: step 3,416**, selected by normalized WER on dev.

### 4.3 Dev curve (323 clips, normalized metrics)

| Step | Dev norm WER | Dev norm CER | Dev loss |
|---|---|---|---|
| 427 | 47.77 | 19.94 | 0.699 |
| 854 | 51.24 | 24.09 | 0.617 |
| 1281 | 43.03 | 17.74 | 0.555 |
| 1708 | 38.61 | 15.35 | 0.532 |
| 2135 | 45.57 | 19.67 | 0.502 |
| 2562 | 50.15 | 29.02 | 0.514 |
| 2989 | 42.70 | 22.06 | 0.505 |
| **3416** | **35.08** | **14.44** | 0.458 |
| 3843 | 35.58 | 14.77 | 0.459 |
| 4071 | 35.31 | 14.60 | 0.457 |

Loss falls steadily, but dev WER swings between 35 and 51 until step 3,416. The most likely cause is a few looping outputs in greedy dev decoding (no temperature fallback during training-time evaluation), not unstable training.

### 4.4 Training-stream statistics

- 81,638 recordings read, 4,557,380 segments seen, **131,204 accepted (2.9 %)**.
- Almost all rejections (4,411,619) come from the **≤ 4 segments per recording cap**: long YouTube recordings contain thousands of segments each.
- So the model saw only a small slice of the available labeled speech. This is the biggest lever for a v4.
- Other rejected segments: inside misaligned rows (timestamps past the audio end) 8,407 (76 rows) · duration 3,860 · chars/s 1,766 · inside rows overlapping eval sentences 463 (97 rows) · empty text 40 · silent/corrupt 10 · too many tokens 4.

### 4.5 Leakage controls

- Every eval sentence with ≥ 3 words (679 sentences) is block-listed. A training row containing one is dropped whole (97 rows dropped).
- Text audit of the augmented TunSwitch sets: TunSwitchCS_VCA, 14 of 37,639 rows contain an eval sentence; TunSwitchTO_VCA, 7 of 15,365. Each is a single distinct sentence, excluded by the block-list.
- The augmented sets are 7 voice-converted copies of the **train** splits only: 37,639 = 7 × 5,377 (TunSwitchCS train), 15,365 = 7 × 2,195 (TunSwitchTO train), 6,146 = 7 × 878 (ApprendreLeTunisien train). So augmentation does not leak test audio.

---

## 5. Evaluation protocol

- **Pool:** every original `test` split that exists: AmenyKH, ApprendreLeTunisien, OneStory, TunSwitchCS, TunSwitchTO, Youtube_TNScrapped_V1. 799 recordings → 805 validated segments, capped at 4 per recording.
- **Split by recording group** (stable hash, 40 % dev / 60 % test), so dev and test share a distribution but no recordings. A code assertion checks this.

| Set | Clips | Groups | Audio | Composition |
|---|---|---|---|---|
| Dev (checkpoint selection) | 323 | 320 | 23.4 min | TunSwitchTO 152 · TunSwitchCS 114 · ApprendreLeTunisien 39 · AmenyKH 14 · Youtube_TNScrapped_V1 4 |
| **Test (used once)** | **482** | 459 | **36.6 min** | TunSwitchCS 185 · TunSwitchTO 171 · ApprendreLeTunisien 69 · AmenyKH 30 · Youtube_TNScrapped_V1 15 · OneStory 12 |

- **Metrics** (corpus-level = total edits / total reference units):
  - **WER / CER**: light clean-up only (NFKC, remove diacritics/tatweel/zero-width chars and `<…>` tags).
  - **Normalized WER / CER**: also lowercases Latin, maps Arabic-Indic digits to 0–9, أ/إ/آ/ٱ→ا, ى→ي, ة→ه, removes punctuation, and attaches standalone «و» to the next word. Applied identically to reference and hypothesis.
  - **No-space CER**: normalized CER with spaces deleted, which isolates word-segmentation errors.
  - **95 % CI**: cluster bootstrap over recording groups, 1,000 resamples.
- **RTF** = processing time / audio duration (lower is faster; < 1 is faster than real time).

---

## 6. Results on the held-out test set

| Model / decoder | n | WER | norm WER (95 % CI) | CER | norm CER | no-space CER | RTF |
|---|---|---|---|---|---|---|---|
| whisper-small base, HF fp16 greedy | 482 | 96.3 | 94.4 (85.4–103.6) | 63.7 | 62.4 | 63.3 | 0.037 |
| fine-tuned, HF fp16 greedy | 482 | 32.8 | 31.9 (29.9–34.2) | 13.4 | 12.9 | 13.1 | 0.025 |
| whisper-small base, CT2 int8_fp16 GPU beam 1 | 482 | 80.9 | 78.1 (75.5–80.9) | 46.5 | 44.9 | 46.1 | 0.092 |
| whisper-small base, CT2 int8_fp16 GPU beam 3 | 482 | 79.1 | 76.3 (73.8–79.0) | 44.5 | 42.9 | 44.0 | 0.078 |
| fine-tuned, CT2 int8_fp16 GPU beam 1 | 482 | 34.6 | 33.8 (30.5–38.6) | 15.0 | 14.6 | 14.8 | 0.042 |
| **fine-tuned, CT2 int8_fp16 GPU beam 3 (deployed)** | **482** | **30.4** | **29.5 (27.5–31.6)** | **12.0** | **11.5** | **11.8** | **0.044** |
| fine-tuned, CT2 int8 CPU beam 1 (subset, 4 threads) | 80 | 31.0 | 30.3 | 13.2 | 12.9 | n/a | 0.889 |
| LinTO Kaldi/Vosk (native streaming) | 400 | 25.4 | 25.0 (22.0–28.1) | 15.7 | 15.4 | 15.9 | 0.150 |

### How to read it

- **Fair base-vs-fine-tuned comparison (same CT2 beam-3 decoder):** normalized WER 76.3 → 29.5 (−46.8 points, −61 % relative); CER 44.5 → 12.0 (−73 % relative).
- The HF greedy rows have no temperature fallback, so looping outputs are not retried. The base model had 12 runaway outputs there, which is why its WER is above 90 %. Compare models on the CT2 rows.
- Beam 3 vs greedy (fine-tuned CT2): −4.2 WER points (−4.3 normalized) at almost the same speed on GPU.
- The cost of int8 quantization is not measured in isolation. CT2 int8 greedy (34.6) vs HF fp16 greedy (32.8) also differ in decoder (CT2 has temperature fallback), and their CIs overlap. The CPU int8 run (31.0 WER) used a different 80-clip subset, so it is indicative only.
- Runaway (looping) outputs, fine-tuned: 0 with beam 3, 1 with greedy CT2.

### Comparison with the LinTO Kaldi/Vosk model

- Vosk has **lower WER** (25.4 vs 30.4) but **higher CER** (15.7 vs 12.0).
- Vosk gets more whole words exactly right, likely helped by its 4-gram language model. Whisper gets more characters right but often misspells words in dialect-specific ways, which WER counts as full errors.
- This is not perfectly like-for-like:
  - Vosk was scored on the **first 400** test clips, not all 482.
  - It uses an external LM trained on 4.5 M lines of Tunisian text.
  - Its training also included TARIC.
- Say this openly in the model card.

### Per-sub-corpus results

These were computed (`per_config_norm_wer`) but only written to `evaluation_report.json` on Kaggle. That file is **not** inside the two zips. Download it from the Kaggle notebook's Output tab if you want per-corpus numbers in the model card. With only 12 OneStory and 15 Youtube_TNScrapped_V1 clips, those two per-corpus numbers would be very noisy anyway.

---

## 7. Error analysis (deployed decoder, test set)

| Error category | Word count |
|---|---|
| Near-miss spelling (≥ 60 % characters right) | 439 |
| Lexical (dialect vocabulary / wrong word) | 371 |
| Deleted words | 160 |
| Word segmentation (merge/split) | 147 |
| Inserted words | 129 |
| Code-switch: Latin word wrong | 105 |
| Code-switch: script mismatch (Latin vs Arabic) | 78 |

- **The largest category is spelling.** Derja has no standard orthography, so many "errors" are valid alternative spellings.
  Most frequent substitutions: لي↔الي (8 + 6), ساره→صاره, موش→مش, والا→ولا, fait→faite, heures→heure.
- **Code-switching** accounts for 183 errors, including 78 where the model wrote a French/English word in the other script than the annotator did.
- **Segmentation** has little effect on CER: no-space CER (11.8) is not lower than normalized CER (11.5). Deleting spaces also shrinks the reference length, so it can even rise slightly.

---

## 8. Low-latency inference (`live_asr.py`)

**Approach:** pseudo-streaming on top of faster-whisper. Whisper is not a native streaming model.
- Silero VAD detects speech onset and end of turn.
- The growing turn is re-decoded greedily every `step_s` seconds to produce **partials**. Stable words are fixed with LocalAgreement-2.
- After 500 ms of silence the turn is **finalized** with beam search. Turns longer than 25 s are cut at a VAD gap.
- Hallucination guards: no decode without speech, a minimum speech length, no-speech/log-prob filters, and collapsing of repeated n-grams.

**Benchmark** (test audio replayed against a virtual real-time clock):
- **TTFP**: time from speech onset until the first partial reaches the user.
- **EoT**: time from end of speech until the final transcript reaches the user.

| Device / mode | Bucket | TTFP p50 / p95 | EoT p50 / p95 | RTF |
|---|---|---|---|---|
| GPU int8_fp16 | short < 4 s | 1.39 / 2.28 s | 0.39 / 0.98 s | 0.128 |
| GPU int8_fp16 | medium 4–12 s | 0.99 / 1.87 s | 0.79 / 1.01 s | 0.220 |
| GPU int8_fp16 | long 12–30 s | 1.01 / 1.89 s | −1.32 / 1.23 s | 0.303 |
| GPU int8_fp16 | multi-turn (~6 utts) | 1.24 / 2.38 s | 0.39 / 1.00 s | 0.217 |
| CPU int8, with partials | short | 4.69 / 5.48 s | 3.37 / 5.06 s | 1.054 |
| CPU int8, with partials | medium | 5.22 / 6.90 s | 4.28 / 7.81 s | 1.139 |
| CPU int8, with partials | long | 5.18 / 7.10 s | 1.59 / 9.33 s | 1.161 |
| CPU int8, final only | short | n/a | 3.12 / 4.18 s | 0.800 |
| CPU int8, final only | medium | n/a | 3.90 / 4.63 s | 0.527 |
| CPU int8, final only | long | n/a | −1.53 / 6.19 s | 0.485 |

- **GPU:** comfortably real time. First words appear in about 1–1.4 s, and the final transcript arrives 0.4–0.8 s after you stop talking.
- **CPU (4 threads):** with partials, RTF > 1, so it falls behind real time. Final-only mode keeps up (RTF 0.5–0.8) but users wait 3–4 s for each turn.
- **Negative EoT** on long clips means the 25 s turn limit split the utterance, so a final was emitted before the speech ended. It is not a real "negative latency".
- **10 s of pure noise produced no text** on every configuration.

**CLI defaults:** `--step 0.6`, `--min-silence-ms 500`, `--final-beam 3`. Use `--no-partials` for CPU.
```
pip install faster-whisper "av<19" sounddevice soundfile numpy
python live_asr.py --model ./whisper-tn-ct2                         # microphone
python live_asr.py --model ./whisper-tn-ct2 --file talk.wav          # replay a file in real time
python live_asr.py --model ./whisper-tn-ct2 --device cpu --compute-type int8 --no-partials
```

---

## 9. Limitations (put these in the model card)

1. **Speaker overlap is possible.** The LinTO datasets have no speaker IDs. TunSwitch-TO has only 89 speakers, so the same voices are likely in both train and test. The split is recording-disjoint, not speaker-disjoint.
2. **Small test set.** 482 clips / 36.6 min. The 95 % CI on normalized WER is about ±2 points. Some sub-corpora have only 12–15 test clips.
3. **Noisy checkpoint selection.** Dev WER jumped between 35 and 51 % across checkpoints, so the selected checkpoint is partly luck.
4. **Training was time-limited.** It stopped at step 4,071 of 4,272. Because of the 4-segments-per-recording cap, the model saw only about 131 k segments (2.9 % of those available).
5. **No standard orthography.** Even normalized WER penalizes valid alternative spellings. Report CER alongside WER.
6. **Code-switching script ambiguity.** References write French/English words sometimes in Latin and sometimes in Arabic script, and the model cannot know which convention an annotator used.
7. **Pseudo-streaming only.** Each partial is a full 30 s encoder pass. End-of-turn latency is at least `min_silence_ms`.
8. **Hardware-specific numbers.** Speeds were measured on Kaggle T4s and a 4-vCPU Kaggle CPU. Re-measure on your own deployment hardware.
9. **Inherited Whisper risks:** hallucinations and repetition on hard audio. Do not use on recordings made without consent or for high-stakes decisions (see the Whisper model card).
10. **YouTube-sourced training audio:** rights belong to the original channels. LinTO redistributes it under CC-BY-4.0.

---

## 10. Reproducing the run

1. Kaggle: GPU **T4 × 2**, Internet **on**, optional secret `HF_TOKEN`.
2. In the configuration cell, set `RUN_MODE = "full"`. It is already set to "full", although the header text says "smoke" is the default.
3. Run All, or Save Version → Run All. The run uses about 11.1 h of the 12 h limit.
4. Outputs: `whisper-tn-v3-hf.zip`, `whisper-tn-v3-ct2.zip`, `evaluation_report.json`, `evaluation_table.md`, `live_asr.py`.

---

## 11. How to cite this model (template)

```bibtex
@misc{whisper-small-tunisian-derja-2026,
  author = {TheBiggyBang1},
  title = {Whisper-small fine-tuned for Tunisian Derja ASR with code-switching},
  year = {2026},
  publisher = {Hugging Face},
  howpublished = {\url{https://huggingface.co/TheBiggyBang/whisper-small-tunisian-derja}}
}
```
