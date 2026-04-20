# CSL-7770 Speech Understanding

**Name:** Shushant Kumar Tiwari  
**Roll No:** M25DE1071  
**Course:** CSL-7770 Speech Understanding  
**Target Language:** Bhojpuri  
**Python:** 3.10  
**PyTorch:** 2.1.0

---

## Project Summary

This project is my submission for Assignment 2. The goal is to process a code-switched (Hindi-English) lecture, identify languages, transcribe, convert to IPA, translate to Bhojpuri, and synthesize speech using voice cloning. I also implemented adversarial and anti-spoofing evaluation.

---

## Directory Structure

```
code/
├── assets/
│   └── syllabus_terms.txt
├── checkpoints/
│   ├── lid_weights.pt
│   ├── speaker_embed.pt
│   └── cm_weights.pt
├── configs/
│   └── config.yaml
├── data/
│   ├── corpus/
│   │   └── bhojpuri_technical_corpus.csv
│   ├── processed/
│   └── raw/
├── pipeline.py
├── requirements.txt
├── results/
│   ├── ablation_mcd.json
│   ├── adversarial_clip.wav
│   ├── adversarial_results.json
│   ├── antispoofing_results.json
│   ├── det_curve.png
│   ├── epsilon_snr_plot.png
│   ├── evaluation_report.json
│   ├── lid_confusion_matrix.png
│   ├── lid_predictions.json
│   ├── output_LRL_cloned.wav
│   ├── output_flat_synthesis.wav
│   ├── pipeline_run.log
│   ├── prosody_f0_plot.png
│   ├── prosody_warped_ref.wav
│   ├── speaker_embed_info.json`
│   ├── transcript_bhojpuri.txt
│   ├── transcript_ipa.txt
│   ├── transcript_raw.txt
│   └── tts_input_bhojpuri.txt
├── src/
│   ├── antispoofing.py
│   ├── audio_preprocessor.py
│   ├── bhojpuri_translator.py
│   ├── constrained_asr.py
│   ├── evaluate_pipeline.py
│   ├── fgsm_attack.py
│   ├── hinglish_ipa.py
│   ├── lid_model.py
│   ├── prosody_dtw.py
│   ├── tts_synthesizer.py
│   └── voice_embedding.py
├── reports/
│   └── (IEEE style report, Implementation report)
```

---

## How to Run

1. **Set up environment:**
	- Python 3.10 (required for TTS)
	- Install dependencies:
	  ```bash
	  pip install -r requirements.txt
	  ```
	- Install ffmpeg (for audio processing)

2. **Prepare data:**
	- Place the provided audio and corpus files in the correct folders (see above).
	- If needed, download the lecture audio as described in `data/raw/LECTURE_VIDEO_README.txt`.

3. **Run pipeline:**
	- The main script is `pipeline.py` (or run individual scripts in `src/` for each task).

4. **Check results:**
	- Outputs and evaluation metrics are in the `results/` folder.

---

## Notes

- If you have issues with TTS, make sure you are using Python 3.10 and have all dependencies installed.
- For any missing files (like the main lecture audio), see the README in the raw data folder.

```
---

## Environment Setup

```bash
# Python 3.10 is required (Coqui TTS does not support Python 3.12)
conda create -n pa2_env python=3.10
conda activate pa2_env

# PyTorch (CUDA 11.8)
pip install torch==2.1.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# All other dependencies
pip install -r requirements.txt

# System dependency
sudo apt install ffmpeg
```

---

## Run Instructions

```bash
cd M25DE1071_PA2

# Run full pipeline end-to-end
python pipeline.py --stage all

# Or run each stage individually
python pipeline.py --stage preprocess   # Task 1.3: denoise + crop
python pipeline.py --stage lid          # Task 1.1: train LID model
python pipeline.py --stage asr          # Task 1.2: transcribe with bias
python pipeline.py --stage ipa          # Task 2.1: IPA conversion
python pipeline.py --stage translate    # Task 2.2: Bhojpuri translation
python pipeline.py --stage embed        # Task 3.1: speaker embedding
python pipeline.py --stage dtw          # Task 3.2: prosody warping
python pipeline.py --stage tts          # Task 3.3: TTS synthesis
python pipeline.py --stage cm           # Task 4.1: anti-spoofing
python pipeline.py --stage fgsm         # Task 4.2: FGSM attack
python pipeline.py --stage eval         # full evaluation report
```

---

## Audio Manifest

| File | Location | Duration | Sample Rate |
|------|----------|----------|-------------|
| `original_segment.wav`   | `data/processed/` | 10 min 0 s | 16000 Hz |
| `student_voice_ref.wav`  | `data/processed/` | 1 min 0 s  | 16000 Hz |
| `output_LRL_cloned.wav`  | `results/`        | 10 min 0 s | 22050 Hz |

---

## Evaluation Results

| Metric | Target | Achieved | Status |
|--------|--------|----------|--------|
| LID F1 (macro) | ≥ 0.85 | **0.9260** | PASS |
| WER English | < 15% | **11.9%** | PASS |
| WER Hindi | < 25% | **21.4%** | PASS |
| MCD (warped TTS) | < 8.0 dB | **6.74 dB** | PASS |
| Switch Precision | ≤ 200 ms | **20 ms** | PASS |
| Anti-Spoofing EER | < 10% | **0.00%** | PASS |
| Min Adversarial ε | reported | **0.002212** | REPORTED |

---

## Notes

- Python 3.10 is strictly required — Coqui TTS (`pip install TTS>=0.22.0`) does not support Python 3.12.
- Run `pipeline.py --stage preprocess` first; it generates `data/processed/` files used by all later stages.
- If DeepFilterNet is unavailable, spectral subtraction activates automatically.
- If IndicTrans2 cannot connect, the 244-entry Bhojpuri corpus fallback activates automatically.
- All GPU/CPU detection is automatic; CPU-only machines are fully supported.
- Run all commands from inside the `M25DE1071_PA2/` directory.

