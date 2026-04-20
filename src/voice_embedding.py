voice_embedding.py  --  Speaker Embedding Extraction (192-d x-vector)
Shushant Kumar Tiwari | M25DE1071 | CSL-7770 Speech Understanding PA2
Python 3.10 | PyTorch 2.1

ECAPA-TDNN via SpeechBrain; fallback: global statistics pooling over
MFCC+delta features (fully offline, no internet required).
"""


import os, json
import numpy as np
import torch
import torchaudio
import soundfile as sf


def load_wav_sf(filepath, target_sr=16000, max_dur_sec=60):
    data, sr = sf.read(filepath)
    if data.ndim > 1:
        data = data.mean(axis=1)
    wav = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    if wav.shape[1] > max_dur_sec * target_sr:
        wav = wav[:, :max_dur_sec * target_sr]
    return wav, target_sr


class VoicePrintExtractor:
    def __init__(self, device="cpu"):
        self.device = device
        self._ecapa = None
        try:
            from speechbrain.inference import EncoderClassifier
            self._ecapa = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                run_opts={"device": device},
                savedir="checkpoints/ecapa_tdnn")
            print("[Embed] ECAPA-TDNN loaded.")
        except Exception as e:
            print(f"[Embed] ECAPA unavailable ({e}). Using offline MFCC-stats embedding.")

    def extract(self, waveform, sr=16000):
        if self._ecapa is not None:
            w = waveform.squeeze()
            if w.dim() == 1:
                w = w.unsqueeze(0)
            with torch.no_grad():
                emb = self._ecapa.encode_batch(w.to(self.device))
            return emb.squeeze().cpu()
        return self._offline_embed(waveform, sr)

    def _offline_embed(self, waveform, sr=16000):
        import librosa
        sig = waveform.squeeze().numpy()
        mfcc  = librosa.feature.mfcc(y=sig, sr=sr, n_mfcc=64)
        d1    = librosa.feature.delta(mfcc)
        d2    = librosa.feature.delta(mfcc, order=2)
        feat  = np.concatenate([mfcc, d1, d2], axis=0)    # (192, T)
        stats = np.concatenate([feat.mean(axis=1), feat.std(axis=1)])  # (384,)
        rng   = np.random.RandomState(7)
        W     = rng.randn(192, 384).astype(np.float32)
        W    /= (np.linalg.norm(W, axis=1, keepdims=True) + 1e-8)
        emb   = W @ stats.astype(np.float32)
        emb  /= (np.linalg.norm(emb) + 1e-8)
        return torch.tensor(emb, dtype=torch.float32)


def cosine_sim(a, b):
    return torch.dot(a / (a.norm()+1e-8), b / (b.norm()+1e-8)).item()


def segment_consistency(waveform, sr, extractor, seg_sec=5):
    seg_len = seg_sec * sr
    n_segs  = waveform.shape[1] // seg_len
    embeddings = [extractor.extract(waveform[:, i*seg_len:(i+1)*seg_len], sr)
                  for i in range(n_segs)]
    sims = [cosine_sim(embeddings[i], embeddings[i+1])
            for i in range(len(embeddings)-1)]
    mean_s = float(np.mean(sims)) if sims else 1.0
    print(f"[Embed] Mean intra-speaker cosine similarity: {mean_s:.4f}")
    return mean_s


if __name__ == "__main__":
    import yaml
    with open("configs/config.yaml") as fh:
        cfg = yaml.safe_load(fh)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Embed] Device: {device}")

    os.makedirs(cfg["paths"]["checkpoints_dir"], exist_ok=True)
    os.makedirs(cfg["paths"]["results_dir"],     exist_ok=True)

    wav_path  = cfg["paths"]["student_voice_ref"]
    save_path = os.path.join(cfg["paths"]["checkpoints_dir"], "speaker_embed.pt")

    wav, sr = load_wav_sf(wav_path, max_dur_sec=cfg["audio"]["voice_ref_duration"])
    print(f"[Embed] Voice ref: {wav.shape[1]/sr:.1f}s at {sr}Hz")

    ext = VoicePrintExtractor(device=device)
    emb = ext.extract(wav, sr)
    print(f"[Embed] Embedding shape: {emb.shape}")
    torch.save(emb, save_path)

    mean_sim = segment_consistency(wav, sr, ext)
    meta = {"embedding_dim": emb.shape[0], "voice_duration_sec": wav.shape[1]/sr,
            "mean_cosine_similarity": mean_sim}
    with open(os.path.join(cfg["paths"]["results_dir"], "speaker_embed_info.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"[Embed] Saved -> {save_path}")
