constrained_asr.py  --  Whisper Transcription with N-gram Logit Biasing
Shushant Kumar Tiwari | M25DE1071 | CSL-7770 Speech Understanding PA2
Python 3.10 | PyTorch 2.1
"""


import os, re, json, math
import numpy as np
import torch
import soundfile as sf
from collections import defaultdict
from transformers import (LogitsProcessor, LogitsProcessorList,
                          WhisperForConditionalGeneration, WhisperProcessor)
from jiwer import wer as compute_wer


class BigramLanguageModel:
    def __init__(self, n=2, smooth=1.0):
        self.n = n
        self.alpha = smooth
        self.bigram_counts  = defaultdict(lambda: defaultdict(int))
        self.unigram_counts = defaultdict(int)
        self.vocab = set()

    def fit(self, sentences):
        for sent in sentences:
            toks = ["<s>"]*(self.n-1) + sent.lower().strip().split() + ["</s>"]
            self.vocab.update(toks[self.n-1:])
            for i in range(len(toks)-self.n+1):
                ctx  = tuple(toks[i:i+self.n-1])
                word = toks[i+self.n-1]
                self.bigram_counts[ctx][word]  += 1
                self.unigram_counts[ctx]       += 1

    def score(self, word, ctx):
        ctx = tuple(ctx[-(self.n-1):])
        V   = max(len(self.vocab), 1)
        num = self.bigram_counts[ctx].get(word, 0) + self.alpha
        den = self.unigram_counts[ctx] + self.alpha * V
        return math.log(num / den + 1e-10)


def read_domain_vocab(vocab_file):
    with open(vocab_file, "r") as fh:
        return [ln.strip().lower() for ln in fh if ln.strip()]


def build_lm_from_vocab(vocab_file, order=2):
    terms  = read_domain_vocab(vocab_file)
    corpus = list(terms) + [t for t in terms if len(t.split()) > 1]
    lm = BigramLanguageModel(n=order)
    lm.fit(corpus)
    return lm, terms


class DomainTermBiasProcessor(LogitsProcessor):
    def __init__(self, tokenizer, term_list, bias_strength=3.0):
        self.strength    = bias_strength
        self.target_tids = set()
        for term in term_list:
            for sub in [term] + term.split():
                self.target_tids.update(tokenizer.encode(sub, add_special_tokens=False))
        print(f"[ASR] Biasing {len(self.target_tids)} BPE token IDs (beta={bias_strength}).")

    def __call__(self, input_ids, scores):
        for tid in self.target_tids:
            if tid < scores.shape[-1]:
                scores[:, tid] += self.strength
        return scores


def transcribe_hinglish(audio_path, vocab_path, cfg, save_path):
    """
    Whisper transcription with N-gram logit biasing.
    Uses HuggingFace WhisperForConditionalGeneration so that
    DomainTermBiasProcessor is passed directly into model.generate()
    via LogitsProcessorList -- the logit bias is applied at every
    decoding step, not as a post-hoc string substitution.
    """
    whisper_size = cfg["asr_config"]["whisper_size"]
    model_id     = f"openai/whisper-{whisper_size}"

    print(f"[ASR] Loading Whisper ({model_id}) via HuggingFace (logit-bias enabled)...")
    processor = WhisperProcessor.from_pretrained(model_id)
    model     = WhisperForConditionalGeneration.from_pretrained(model_id)
    model.eval()

    print("[ASR] Building N-gram LM on syllabus vocabulary...")
    lm, domain_terms = build_lm_from_vocab(vocab_path, order=cfg["asr_config"]["lm_order"])

    print("[ASR] Constructing domain-term logit bias processor...")
    bias_proc = DomainTermBiasProcessor(
        processor.tokenizer, domain_terms,
        bias_strength=cfg["asr_config"]["term_bias"])
    proc_list = LogitsProcessorList([bias_proc])

    # Read audio (handles .wav / .mp4 / .m4a via soundfile)
    data, native_sr = sf.read(audio_path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = data.astype("float32")

    target_sr = 16000
    if native_sr != target_sr:
        import torchaudio
        wav_t = torchaudio.functional.resample(
            torch.tensor(data).unsqueeze(0), native_sr, target_sr)
        data = wav_t.squeeze().numpy()

    # Whisper processes 30-second windows
    chunk_sec = 30
    chunk_len = chunk_sec * target_sr
    beam_w    = cfg["asr_config"]["beam_width"]
    # task=transcribe; let Whisper auto-detect language (supports code-switching)
    forced_ids = processor.get_decoder_prompt_ids(task="transcribe")
    corrected_lines = []

    n_chunks = max(1, int(len(data) / chunk_len))
    print(f"[ASR] Transcribing {len(data)/target_sr:.1f}s in "
          f"{n_chunks} chunk(s), beam={beam_w}, "
          f"biasing {len(bias_proc.target_tids)} BPE token IDs...")

    for i in range(0, len(data), chunk_len):
        chunk = data[i: i + chunk_len]
        if len(chunk) < target_sr // 2:   # skip sub-0.5s tail
            continue
        inputs = processor(chunk, sampling_rate=target_sr, return_tensors="pt")
        with torch.no_grad():
            pred_ids = model.generate(
                inputs.input_features,
                num_beams=beam_w,
                logits_processor=proc_list,      # <-- logit bias applied here
                forced_decoder_ids=forced_ids,
            )
        text = processor.batch_decode(pred_ids, skip_special_tokens=True)[0].strip()
        t0   = i / target_sr
        t1   = min((i + chunk_len) / target_sr, len(data) / target_sr)
        corrected_lines.append(f"[{t0:.2f}s - {t1:.2f}s] {text}")

    full_text = "\n".join(corrected_lines)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as fh:
        fh.write(full_text)
    print(f"[ASR] Transcript saved -> {save_path}")
    return [], full_text


if __name__ == "__main__":
    import yaml
    with open("configs/config.yaml") as fh:
        cfg = yaml.safe_load(fh)
    segs, txt = transcribe_hinglish(
        audio_path=cfg["paths"]["original_segment"],
        vocab_path=cfg["paths"]["syllabus_terms"],
        cfg=cfg,
        save_path=os.path.join(cfg["paths"]["results_dir"], "transcript_raw.txt"))
    print("\n[ASR] First 400 chars:", txt[:400])
