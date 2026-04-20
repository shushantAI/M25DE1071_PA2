


import os
import json
import numpy as np
import torch
import torch.nn as nn
import torchaudio
import soundfile as sf
from transformers import Wav2Vec2Model, Wav2Vec2Processor
from sklearn.metrics import f1_score, confusion_matrix
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def load_wav_sf(filepath, target_sr=16000):
    data, native_sr = sf.read(filepath)
    if data.ndim > 1:
        data = data.mean(axis=1)
    tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
    if native_sr != target_sr:
        tensor = torchaudio.functional.resample(tensor, native_sr, target_sr)
    return tensor, target_sr



class CodeSwitchLIDNet(nn.Module):
    def __init__(self, feat_dim=768, hidden_units=256, rnn_layers=2, n_classes=2):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=feat_dim,
            hidden_size=hidden_units,
            num_layers=rnn_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        rnn_out_dim = hidden_units * 2
        self.local_attn  = nn.MultiheadAttention(embed_dim=rnn_out_dim, num_heads=4, batch_first=True)
        self.global_attn = nn.MultiheadAttention(embed_dim=rnn_out_dim, num_heads=4, batch_first=True)
        self.ln_local  = nn.LayerNorm(rnn_out_dim)
        self.ln_global = nn.LayerNorm(rnn_out_dim)
        self.head = nn.Sequential(
            nn.Linear(rnn_out_dim, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        h, _ = self.rnn(x)
        a1, _ = self.local_attn(h, h, h)
        a1 = self.ln_local(h + a1)
        a2, _ = self.global_attn(a1, a1, a1)
        a2 = self.ln_global(a1 + a2)
        return self.head(a2)


# Wav2Vec 2.0 feature extractor

class W2VFeatureExtractor:
    def __init__(self, model_id="facebook/wav2vec2-base", device="cpu"):
        self.device = device
        self.processor = Wav2Vec2Processor.from_pretrained(model_id)
        self.encoder   = Wav2Vec2Model.from_pretrained(model_id).to(device)
        self.encoder.eval()

    @torch.no_grad()
    def get_features(self, waveform, sr=16000):
        if isinstance(waveform, np.ndarray):
            waveform = torch.tensor(waveform, dtype=torch.float32)
        inp = self.processor(waveform.squeeze().numpy(),
                             sampling_rate=sr, return_tensors="pt", padding=True)
        out = self.encoder(inp.input_values.to(self.device))
        return out.last_hidden_state.squeeze(0).cpu()


# Pseudo-label generator using Whisper word timestamps + Unicode detection

def build_frame_labels(audio_path, hop_ms=20, sr=16000):
    import whisper
    print("[LID] Generating frame-level pseudo-labels via Whisper alignment...")
    asr = whisper.load_model("base")
    result = asr.transcribe(audio_path, word_timestamps=True, language=None)

    wav, _ = load_wav_sf(audio_path, sr)
    n_frames = wav.shape[1] // int(sr * hop_ms / 1000)
    labels = np.zeros(n_frames, dtype=np.int64)

    devanagari = set("अआइईउऊएऐओऔकखगघचछजझटठडढणतथदधनपफबभमयरलवशषसह")
    for seg in result.get("segments", []):
        for wi in seg.get("words", []):
            word = wi.get("word", "").strip()
            s_fr = int(wi.get("start", 0.0) * 1000 / hop_ms)
            e_fr = int(wi.get("end",   0.0) * 1000 / hop_ms)
            labels[s_fr:e_fr] = 1 if any(c in devanagari for c in word) else 0
    return labels


# Sliding-window dataset

class SwitchingFrameDataset(Dataset):
    def __init__(self, feats, labs, window=32):
        self.feats  = feats
        self.labs   = labs
        self.window = window

    def __len__(self):
        return len(self.feats) - self.window

    def __getitem__(self, idx):
        x = torch.tensor(self.feats[idx: idx + self.window], dtype=torch.float32)
        y = torch.tensor(self.labs[idx:  idx + self.window], dtype=torch.long)
        return x, y


# Training routine

def fit_lid_model(feats, labs, cfg, ckpt_path, device="cpu"):
    ds = SwitchingFrameDataset(feats, labs, window=cfg["ctx_frames"])
    n_tr = int(0.8 * len(ds))
    tr_ds, va_ds = torch.utils.data.random_split(ds, [n_tr, len(ds) - n_tr])
    tr_dl = DataLoader(tr_ds, batch_size=cfg["batch_size"], shuffle=True)
    va_dl = DataLoader(va_ds, batch_size=cfg["batch_size"])

    net = CodeSwitchLIDNet(feat_dim=feats.shape[-1],
                           hidden_units=cfg["hidden_units"],
                           rnn_layers=cfg["rnn_layers"],
                           n_classes=cfg["n_classes"]).to(device)
    opt  = torch.optim.Adam(net.parameters(), lr=cfg["learning_rate"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3)
    loss_fn = nn.CrossEntropyLoss()
    best_f1 = 0.0

    for ep in range(cfg["num_epochs"]):
        net.train()
        ep_loss = 0.0
        for xb, yb in tr_dl:
            xb, yb = xb.to(device), yb.to(device)
            logits = net(xb)
            loss = loss_fn(logits.view(-1, cfg["n_classes"]), yb.view(-1))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item()

        net.eval()
        preds_all, tgts_all = [], []
        with torch.no_grad():
            for xb, yb in va_dl:
                logits = net(xb.to(device))
                preds_all.extend(logits.argmax(-1).cpu().numpy().flatten())
                tgts_all.extend(yb.numpy().flatten())

        val_f1 = f1_score(tgts_all, preds_all, average="macro")
        avg_loss = ep_loss / len(tr_dl)
        sched.step(avg_loss)
        print(f"[LID] Epoch {ep+1:02d}/{cfg['num_epochs']} | loss={avg_loss:.4f} | val_F1={val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(net.state_dict(), ckpt_path)
            print(f"[LID] Checkpoint saved (F1={val_f1:.4f})")

    print(f"[LID] Training done. Best F1={best_f1:.4f}")
    return best_f1


# Inference

def run_inference(audio_path, ckpt_path, cfg, device="cpu", sr=16000):
    extractor = W2VFeatureExtractor(model_id=cfg["backbone"], device=device)
    wav, _ = load_wav_sf(audio_path, sr)
    feats = extractor.get_features(wav, sr).numpy()

    net = CodeSwitchLIDNet(feat_dim=feats.shape[-1],
                           hidden_units=cfg["hidden_units"],
                           rnn_layers=cfg["rnn_layers"],
                           n_classes=cfg["n_classes"]).to(device)
    net.load_state_dict(torch.load(ckpt_path, map_location=device))
    net.eval()

    ctx = cfg.get("ctx_frames", 32)
    out = []
    with torch.no_grad():
        for i in range(0, len(feats) - ctx, ctx):
            chunk = torch.tensor(feats[i:i+ctx], dtype=torch.float32).unsqueeze(0).to(device)
            out.extend(net(chunk).argmax(-1).squeeze().cpu().numpy().tolist())
    return np.array(out)


# Switch-point timestamp extraction

def get_switch_timestamps(preds, hop_ms=20):
    events = []
    prev = preds[0]
    for i, lbl in enumerate(preds[1:], 1):
        if lbl != prev:
            events.append({"frame": i, "time_sec": round(i * hop_ms / 1000.0, 3),
                           "to_lang": "EN" if lbl == 0 else "HI"})
            prev = lbl
    return events


# Confusion matrix visualisation

def save_confusion_matrix(true_seq, pred_seq, out_path):
    cm = confusion_matrix(true_seq, pred_seq)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlOrBr",
                xticklabels=["English", "Hindi"],
                yticklabels=["English", "Hindi"], ax=ax, annot_kws={"size": 13})
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("Ground-Truth Label", fontsize=11)
    ax.set_title("Frame-Level LID: Confusion Matrix\n(Code-Switching Boundaries)", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[LID] Confusion matrix saved → {out_path}")


# CLI entry-point

if __name__ == "__main__":
    import yaml
    with open("configs/config.yaml") as fh:
        cfg = yaml.safe_load(fh)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[LID] Device: {device}")

    seg_path  = cfg["paths"]["original_segment"]
    ckpt_path = os.path.join(cfg["paths"]["checkpoints_dir"], "lid_weights.pt")
    res_dir   = cfg["paths"]["results_dir"]
    os.makedirs(cfg["paths"]["checkpoints_dir"], exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)

    extractor = W2VFeatureExtractor(model_id=cfg["lid_config"]["backbone"], device=device)
    wav, sr   = load_wav_sf(seg_path)
    feats     = extractor.get_features(wav, sr).numpy()
    labels    = build_frame_labels(seg_path, hop_ms=cfg["lid_config"]["hop_ms"], sr=sr)

    n = min(len(feats), len(labels))
    feats, labels = feats[:n], labels[:n]
    print(f"[LID] Features: {feats.shape} | Labels: {labels.shape}")

    best_f1 = fit_lid_model(feats, labels, cfg["lid_config"], ckpt_path, device=device)
    preds   = run_inference(seg_path, ckpt_path, cfg["lid_config"], device=device)
    events  = get_switch_timestamps(preds, cfg["lid_config"]["hop_ms"])

    out_json = os.path.join(res_dir, "lid_predictions.json")
    with open(out_json, "w") as fh:
        json.dump({"predictions_sample": preds[:200].tolist(),
                   "switch_timestamps": events[:50], "best_f1": best_f1}, fh, indent=2)

    save_confusion_matrix(labels[:len(preds)], preds,
                          os.path.join(res_dir, "lid_confusion_matrix.png"))
    print(f"[LID] Done. Best F1 = {best_f1:.4f}")
