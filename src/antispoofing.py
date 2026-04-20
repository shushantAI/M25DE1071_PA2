antispoofing.py  --  LFCC-CNN Anti-Spoofing Countermeasure
Shushant Kumar Tiwari | M25DE1071 | CSL-7770 Speech Understanding PA2
Python 3.10 | PyTorch 2.1

Feature  : LFCC (linear filterbank -- superior to MFCC for detecting
           WORLD-vocoder artifacts above 4 kHz)
Model    : 1-D CNN with 3 conv blocks + adaptive pooling + 2-class head
Metric   : Equal Error Rate (EER) from ROC curve
"""


import os, json
import numpy as np
import torch
import torch.nn as nn
import torchaudio
import soundfile as sf
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_curve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_wav_sf(path, target_sr=16000):
    data, sr = sf.read(path)
    if data.ndim > 1: data = data.mean(axis=1)
    wav = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav, target_sr


class LinearCepstralExtractor:
    def __init__(self, sr=16000, n_filters=70, n_ceps=40, win=512, hop=160):
        self.sr = sr; self.n_f = n_filters; self.n_c = n_ceps
        self.win = win; self.hop = hop

    def compute(self, waveform):
        if isinstance(waveform, np.ndarray):
            waveform = torch.tensor(waveform, dtype=torch.float32)
        waveform = waveform.squeeze()
        spec = torch.stft(waveform, n_fft=self.win, hop_length=self.hop,
                          win_length=self.win,
                          window=torch.hann_window(self.win), return_complex=True)
        mag = spec.abs()
        fb  = self._lin_filterbank(mag.shape[0])
        log_fb = torch.log(torch.matmul(fb, mag) + 1e-8)
        lfcc   = torch.matmul(self._dct(self.n_f, self.n_c), log_fb)
        d1     = self._delta(lfcc)
        d2     = self._delta(d1)
        return torch.cat([lfcc, d1, d2], dim=0)   # (3*n_ceps, T)

    def _lin_filterbank(self, n_bins):
        fb  = torch.zeros(self.n_f, n_bins)
        pts = torch.linspace(0, n_bins-1, self.n_f+2).long()
        for i in range(self.n_f):
            s, c, e = pts[i], pts[i+1], pts[i+2]
            if c > s: fb[i, s:c] = (torch.arange(s,c)-s).float()/(c-s)
            if e > c: fb[i, c:e] = (e-torch.arange(c,e)).float()/(e-c)
        return fb

    def _dct(self, n_in, n_out):
        n = torch.arange(n_in).float()
        k = torch.arange(n_out).float()
        return torch.cos(np.pi/n_in * k.unsqueeze(1)*(n.unsqueeze(0)+0.5)) / n_in

    def _delta(self, feat, N=2):
        T = feat.shape[1]; d = torch.zeros_like(feat)
        for t in range(T):
            num = sum(n*(feat[:, min(t+n,T-1)]-feat[:, max(t-n,0)]) for n in range(1,N+1))
            d[:, t] = num / (2*sum(n**2 for n in range(1,N+1)))
        return d


def segment_audio(wav_path, sr=16000, seg_sec=3.0):
    wav, _ = read_wav_sf(wav_path, sr)
    wav = wav.squeeze()
    seg = int(seg_sec * sr)
    return [wav[i:i+seg] for i in range(0, len(wav)-seg, seg)]


class AudioAuthenticityDataset(Dataset):
    def __init__(self, real_segs, spoof_segs, extractor):
        self.items, self.labs = [], []
        for s in real_segs:
            self.items.append(extractor.compute(s)); self.labs.append(0)
        for s in spoof_segs:
            self.items.append(extractor.compute(s)); self.labs.append(1)

    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        return self.items[i], torch.tensor(self.labs[i], dtype=torch.long)


class AudioAuthenticityNet(nn.Module):
    def __init__(self, in_ch=120):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_ch, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 3, padding=1),   nn.BatchNorm1d(128),nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(128, 64, 3, padding=1),   nn.BatchNorm1d(64), nn.ReLU(), nn.AdaptiveAvgPool1d(1))
        self.clf = nn.Sequential(
            nn.Flatten(), nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32, 2))

    def forward(self, x): return self.clf(self.encoder(x))


def train_cm(dataset, cfg, save_path, device="cpu"):
    n_tr = int(0.8 * len(dataset))
    tr, va = torch.utils.data.random_split(dataset, [n_tr, len(dataset)-n_tr])
    tr_dl = DataLoader(tr, batch_size=cfg["batch_size"], shuffle=True)
    va_dl = DataLoader(va, batch_size=cfg["batch_size"])
    in_ch = dataset[0][0].shape[0]
    net = AudioAuthenticityNet(in_ch=in_ch).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=cfg["learning_rate"])
    loss_fn = nn.CrossEntropyLoss()
    best_vloss = float("inf")
    for ep in range(cfg["num_epochs"]):
        net.train()
        for x, y in tr_dl:
            x,y = x.to(device), y.to(device)
            loss = loss_fn(net(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval(); vl = 0.0
        with torch.no_grad():
            for x,y in va_dl:
                vl += loss_fn(net(x.to(device)), y.to(device)).item()
        vl /= len(va_dl)
        if (ep+1) % 10 == 0:
            print(f"[CM] Epoch {ep+1}/{cfg['num_epochs']} | val_loss={vl:.4f}")
        if vl < best_vloss:
            best_vloss = vl; torch.save(net.state_dict(), save_path)
    print(f"[CM] Training done. Model saved -> {save_path}")
    return net


def evaluate_eer(net, dataset, device="cpu"):
    dl = DataLoader(dataset, batch_size=16, shuffle=False)
    net.eval(); scores, labs = [], []
    with torch.no_grad():
        for x,y in dl:
            s = torch.softmax(net(x.to(device)), dim=1)[:,1].cpu().numpy()
            scores.extend(s); labs.extend(y.numpy())
    fpr, tpr, _ = roc_curve(labs, scores)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = float((fpr[idx] + fnr[idx]) / 2)
    print(f"[CM] EER = {eer*100:.2f}%  (target < 10%)")
    return eer, fpr, tpr


def plot_det(fpr, fnr, out_path):
    plt.figure(figsize=(6, 5))
    plt.plot(fpr*100, fnr*100, color="#3a3a8c", lw=2)
    plt.xlabel("False Alarm Rate (%)"); plt.ylabel("Miss Rate (%)")
    plt.title("DET Curve - Anti-Spoofing Countermeasure (LFCC-CNN)")
    plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(out_path, dpi=150); plt.close()
    print(f"[CM] DET curve saved -> {out_path}")


if __name__ == "__main__":
    import yaml
    with open("configs/config.yaml") as fh:
        cfg = yaml.safe_load(fh)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    res  = cfg["paths"]["results_dir"]
    ckpt = cfg["paths"]["checkpoints_dir"]
    sr   = cfg["cm_config"]["sr"]

    real_segs  = segment_audio(cfg["paths"]["student_voice_ref"], sr)
    spoof_segs = segment_audio(os.path.join(res, "output_LRL_cloned.wav"), sr)
    print(f"[CM] Bona-fide: {len(real_segs)}  Spoof: {len(spoof_segs)} segments")

    ext = LinearCepstralExtractor(sr=sr, n_filters=cfg["cm_config"]["n_linear_filters"],
                                  n_ceps=cfg["cm_config"]["n_cepstral_coeffs"])

    # Proper 80/20 train / held-out test split — EER is reported on test segments only
    n_rt = max(1, int(0.8 * len(real_segs)))
    n_st = max(1, int(0.8 * len(spoof_segs)))
    train_ds = AudioAuthenticityDataset(real_segs[:n_rt],  spoof_segs[:n_st],  ext)
    test_ds  = AudioAuthenticityDataset(real_segs[n_rt:],  spoof_segs[n_st:],  ext)
    print(f"[CM] Train samples: {len(train_ds)}  Test samples: {len(test_ds)}")

    cm_path = os.path.join(ckpt, "cm_weights.pt")
    net = train_cm(train_ds, cfg["cm_config"], cm_path, device=device)
    net.load_state_dict(torch.load(cm_path, map_location=device))
    eer, fpr, tpr = evaluate_eer(net, test_ds, device=device)   # held-out test only
    fnr = 1 - tpr
    plot_det(fpr, fnr, os.path.join(res, "det_curve.png"))

    out = {"eer_percent": round(eer*100,2), "eer_target": 10.0,
           "passes": eer*100 < 10.0, "n_real": len(real_segs), "n_spoof": len(spoof_segs)}
    with open(os.path.join(res, "antispoofing_results.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(out)
