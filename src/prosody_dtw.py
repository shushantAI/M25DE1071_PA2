prosody_dtw.py  --  DTW-Based Prosody Transfer (F0 + Energy)
Shushant Kumar Tiwari | M25DE1071 | CSL-7770 Speech Understanding PA2
Python 3.10 | PyTorch 2.1

Extracts F0 and energy contours from professor's lecture via WORLD vocoder.
Applies Dynamic Time Warping in log-F0 domain to warp student's prosody
onto the professor's temporal pattern, preserving teaching intonation style.
"""


import os
import numpy as np
import torch
import torchaudio
import soundfile as sf
import pyworld as pw
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


def read_wav(filepath, target_sr=16000, max_sec=None):
    data, sr = sf.read(filepath)
    if data.ndim > 1:
        data = data.mean(axis=1)
    wav = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    if max_sec:
        wav = wav[:, :max_sec*target_sr]
    return wav.squeeze().numpy().astype(np.float64), target_sr


def extract_f0_world(signal, sr, frame_ms=5.0):
    _f0, times = pw.dio(signal, sr, frame_period=frame_ms)
    f0 = pw.stonemask(signal, _f0, times, sr)
    return f0, times


def extract_energy(signal, win=512, hop=160):
    return np.array([np.sqrt(np.mean(signal[i:i+win]**2))
                     for i in range(0, len(signal)-win, hop)])


def log_f0(f0):
    voiced = f0 > 0
    lf0 = np.zeros_like(f0)
    lf0[voiced] = np.log(f0[voiced] + 1e-8)
    return lf0, voiced


def exp_f0(lf0, voiced):
    f0 = np.zeros_like(lf0)
    f0[voiced] = np.exp(lf0[voiced])
    return f0


def dtw_align_sequences(src, ref):
    """
    Minimal DTW (pure numpy, no external library dependency).
    Warps src onto the time-axis of ref using the Sakoe-Chiba algorithm.
    Applied in log-F0 domain for perceptually uniform pitch distance.
    """
    M, N = len(src), len(ref)
    D = np.full((M, N), np.inf)
    D[0, 0] = abs(src[0] - ref[0])
    for i in range(1, M):
        D[i, 0] = D[i-1, 0] + abs(src[i] - ref[0])
    for j in range(1, N):
        D[0, j] = D[0, j-1] + abs(src[0] - ref[j])
    for i in range(1, M):
        for j in range(1, N):
            D[i, j] = min(D[i-1,j], D[i,j-1], D[i-1,j-1]) + abs(src[i]-ref[j])
    i, j = M-1, N-1
    pi, pj = [i], [j]
    while i > 0 or j > 0:
        opts = []
        if i > 0: opts.append((D[i-1,j], i-1, j))
        if j > 0: opts.append((D[i,j-1], i,   j-1))
        if i > 0 and j > 0: opts.append((D[i-1,j-1], i-1, j-1))
        _, i, j = min(opts)
        pi.append(i); pj.append(j)
    pi.reverse(); pj.reverse()
    warped = np.zeros(N)
    for a, b in zip(pi, pj):
        warped[b] = src[min(a, M-1)]
    return warped


def warp_prosody(prof_wav_path, student_wav_path, out_wav_path, cfg):
    sr = cfg["audio"]["sampling_rate"]

    print("[DTW] Loading professor lecture (60s clip)...")
    prof_sig, _   = read_wav(prof_wav_path,    target_sr=sr, max_sec=60)
    print("[DTW] Loading student voice reference...")
    student_sig, _ = read_wav(student_wav_path, target_sr=sr)

    print("[DTW] Extracting F0 and energy contours...")
    prof_f0,    prof_t   = extract_f0_world(prof_sig,    sr)
    student_f0, student_t = extract_f0_world(student_sig, sr)
    prof_nrg    = extract_energy(prof_sig)
    student_nrg = extract_energy(student_sig)

    print("[DTW] Log-F0 computation + DTW alignment...")
    pf0, pv = log_f0(prof_f0)
    sf0, _  = log_f0(student_f0)
    N = min(len(pf0), len(sf0), 800)
    warped_lf0 = dtw_align_sequences(sf0[:N], pf0[:N])
    warped_f0  = exp_f0(warped_lf0, pv[:N])

    Ne = min(len(prof_nrg), len(student_nrg), 5000)
    warped_nrg = dtw_align_sequences(student_nrg[:Ne], prof_nrg[:Ne])

    print("[DTW] WORLD resynthesis with transferred prosody...")
    sp = pw.cheaptrick(student_sig, student_f0, student_t, sr)
    ap = pw.d4c(student_sig,        student_f0, student_t, sr)
    n_fr = sp.shape[0]

    wf0 = np.pad(warped_f0, (0, max(0, n_fr - len(warped_f0))))[:n_fr]
    nrg_fn = interp1d(np.linspace(0, 1, len(warped_nrg)), warped_nrg,
                      fill_value="extrapolate")
    nrg_sc = np.clip(nrg_fn(np.linspace(0, 1, n_fr)), 0, None)
    sp_sc  = sp * nrg_sc[:, np.newaxis]

    synth = pw.synthesize(wf0, sp_sc, ap, sr)
    synth = (synth / (np.max(np.abs(synth)) + 1e-8)).astype(np.float32)
    os.makedirs(os.path.dirname(out_wav_path) or ".", exist_ok=True)
    sf.write(out_wav_path, synth, sr)
    print(f"[DTW] Prosody-warped audio saved -> {out_wav_path}")
    return prof_f0, student_f0, warped_f0, prof_nrg, warped_nrg


def plot_f0_comparison(pf0, sf0, wf0, out_path):
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=False)
    N = min(500, len(pf0), len(sf0), len(wf0))
    axes[0].plot(pf0[:N], color="#1b7837", lw=1.2)
    axes[0].set_title("Professor F0 Contour (Reference)"); axes[0].set_ylabel("F0 (Hz)")
    axes[1].plot(sf0[:N], color="#2166ac", lw=1.2)
    axes[1].set_title("Student F0 Contour (Pre-Warping)"); axes[1].set_ylabel("F0 (Hz)")
    axes[2].plot(wf0[:N], color="#d6604d", lw=1.2)
    axes[2].set_title("Warped Student F0 (Post DTW Transfer)"); axes[2].set_ylabel("F0 (Hz)")
    for ax in axes:
        ax.set_xlabel("Frame Index"); ax.grid(True, alpha=0.3)
    plt.suptitle("Prosody Warping: Log-F0 DTW Transfer\n(Teaching Style Preserved)", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[DTW] Prosody plot saved -> {out_path}")


if __name__ == "__main__":
    import yaml
    with open("configs/config.yaml") as fh:
        cfg = yaml.safe_load(fh)
    res = cfg["paths"]["results_dir"]
    pf0, sf0, wf0, pe, we = warp_prosody(
        cfg["paths"]["original_segment"],
        cfg["paths"]["student_voice_ref"],
        os.path.join(res, "prosody_warped_ref.wav"), cfg)
    plot_f0_comparison(pf0, sf0, wf0, os.path.join(res, "prosody_f0_plot.png"))
