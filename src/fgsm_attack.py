fgsm_attack.py  --  FGSM Adversarial Perturbation on Frame-Level LID
Shushant Kumar Tiwari | M25DE1071 | CSL-7770 Speech Understanding PA2
Python 3.10 | PyTorch 2.1

Goal: find minimum epsilon s.t. LID misclassifies Hindi -> English
      while keeping SNR > 40 dB (imperceptible to humans).
Gradient computed in Wav2Vec2 feature space for efficiency.
"""


import os, json
import numpy as np
import torch
import torch.nn as nn
import torchaudio
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_wav_sf(path, target_sr=16000):
    data, sr = sf.read(path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    wav = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav, target_sr


def snr_db(original, perturbed):
    noise = perturbed - original
    sp = torch.mean(original**2)
    np_ = torch.mean(noise**2)
    return float(10 * torch.log10(sp / (np_ + 1e-12))) if np_ > 0 else float("inf")


def load_lid_components(ckpt_path, cfg, device):
    import sys; sys.path.insert(0, "src")
    from lid_model import CodeSwitchLIDNet, W2VFeatureExtractor
    net = CodeSwitchLIDNet(
        feat_dim=768,
        hidden_units=cfg["lid_config"]["hidden_units"],
        rnn_layers=cfg["lid_config"]["rnn_layers"],
        n_classes=cfg["lid_config"]["n_classes"]).to(device)
    net.load_state_dict(torch.load(ckpt_path, map_location=device))
    net.eval()
    ext = W2VFeatureExtractor(model_id=cfg["lid_config"]["backbone"], device=device)
    return net, ext


def extract_hindi_clip(audio_path, lid_json_path, sr=16000, clip_sec=5):
    with open(lid_json_path) as fh:
        lid = json.load(fh)
    wav, _ = load_wav_sf(audio_path, sr)
    t_start = next((e["time_sec"] for e in lid.get("switch_timestamps",[])
                    if e["to_lang"]=="HI"), 30.0)
    s = int(t_start * sr)
    return wav[:, s: s + clip_sec*sr], t_start


def fgsm_feature_space(net, extractor, clip, eps, sr=16000, ctx=32, device="cpu"):
    """
    Compute FGSM perturbation in Wav2Vec2 feature space then
    linearly interpolate gradient signs back to waveform time-axis.
    More memory-efficient than raw-waveform backprop through the encoder.
    """
    clip_np = clip.squeeze().numpy().astype(np.float32)
    with torch.no_grad():
        feats = extractor.get_features(clip, sr)
    feat_t = feats.clone().requires_grad_(True).unsqueeze(0).to(device)
    ctx_chunk = feat_t[:, :ctx, :]
    logits = net(ctx_chunk)
    target = torch.zeros(ctx, dtype=torch.long).to(device)   # force EN=0
    loss = nn.CrossEntropyLoss()(logits.squeeze(0), target)
    loss.backward()
    if feat_t.grad is None:
        return clip
    gs = feat_t.grad.sign().cpu().squeeze().numpy()
    noise_env = np.mean(np.abs(gs), axis=1)
    noise_env /= (noise_env.max() + 1e-8)
    n = len(clip_np)
    if len(noise_env) < n:
        noise_env = np.interp(np.linspace(0,1,n), np.linspace(0,1,len(noise_env)), noise_env)
    else:
        noise_env = noise_env[:n]
    perturbed = np.clip(clip_np + eps * np.sign(noise_env), -1.0, 1.0)
    return torch.tensor(perturbed).unsqueeze(0)


def epsilon_sweep(net, extractor, clip, eps_vals, cfg, device="cpu"):
    sr  = cfg["audio"]["sampling_rate"]
    ctx = cfg["lid_config"]["ctx_frames"]
    thr = cfg["fgsm_config"]["min_snr_db"]
    orig = clip.clone()
    with torch.no_grad():
        f0 = extractor.get_features(clip, sr)
        c0 = torch.tensor(f0[:ctx].numpy()).unsqueeze(0).to(device)
        orig_lbl = int(net(c0).argmax(-1).squeeze()[0].item())
    print(f"[FGSM] Original LID: {'HI' if orig_lbl==1 else 'EN'}")
    records = []
    for eps in eps_vals:
        perturbed = fgsm_feature_space(net, extractor, clip, eps, sr, ctx, device)
        snr = snr_db(orig.squeeze(), perturbed.squeeze())
        with torch.no_grad():
            fp = extractor.get_features(perturbed, sr)
            cp = torch.tensor(fp[:ctx].numpy()).unsqueeze(0).to(device)
            pred = int(net(cp).argmax(-1).squeeze()[0].item())
        flipped = pred != orig_lbl
        valid   = flipped and snr > thr
        records.append({"epsilon": round(eps, 6), "snr_db": round(snr, 2),
                         "pred": "EN" if pred==0 else "HI",
                         "flipped": bool(flipped), "snr_ok": bool(snr>thr),
                         "valid": bool(valid)})
        print(f"  eps={eps:.5f} | SNR={snr:.1f}dB | pred={'EN' if pred==0 else 'HI'} "
              f"| flipped={flipped} | valid={valid}")
    return records


def plot_eps_snr(records, out_path):
    eps  = [r["epsilon"] for r in records]
    snrs = [r["snr_db"]  for r in records]
    cols = ["#d62728" if r["flipped"] else "#2171b5" for r in records]
    fig, ax = plt.subplots(figsize=(8,4))
    ax.scatter(eps, snrs, c=cols, zorder=5, s=55)
    ax.plot(eps, snrs, color="gray", lw=1, zorder=3)
    ax.axhline(40, color="#31a354", ls="--", lw=1.5, label="SNR = 40 dB threshold")
    ax.set_xscale("log")
    ax.set_xlabel("Perturbation Epsilon (ε) -- log scale")
    ax.set_ylabel("SNR (dB)")
    ax.set_title("FGSM Attack on LID: Epsilon vs SNR\n"
                 "● Red = LID flipped (HI→EN)   ● Blue = not flipped")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()
    print(f"[FGSM] Plot saved -> {out_path}")


if __name__ == "__main__":
    import yaml
    with open("configs/config.yaml") as fh:
        cfg = yaml.safe_load(fh)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    res  = cfg["paths"]["results_dir"]
    ckpt = cfg["paths"]["checkpoints_dir"]

    net, ext = load_lid_components(os.path.join(ckpt, "lid_weights.pt"), cfg, device)

    clip, t0 = extract_hindi_clip(cfg["paths"]["original_segment"],
                                  os.path.join(res, "lid_predictions.json"))
    print(f"[FGSM] Hindi clip from t={t0:.1f}s")

    eps_vals = np.logspace(
        np.log10(cfg["fgsm_config"]["eps_low"]),
        np.log10(cfg["fgsm_config"]["eps_high"]),
        cfg["fgsm_config"]["n_eps_steps"]).tolist()

    records = epsilon_sweep(net, ext, clip, eps_vals, cfg, device)
    valids  = [r for r in records if r["valid"]]
    min_eps = min(valids, key=lambda x: x["epsilon"]) if valids else None

    sf.write(os.path.join(res, "adversarial_clip.wav"),
             fgsm_feature_space(net, ext, clip,
                                min_eps["epsilon"] if min_eps else eps_vals[10],
                                cfg["audio"]["sampling_rate"],
                                cfg["lid_config"]["ctx_frames"], device
                                ).squeeze().numpy(),
             cfg["audio"]["sampling_rate"])

    summary = {"hindi_start_sec": t0, "minimum_valid_epsilon": min_eps,
               "all_results": records}
    with open(os.path.join(res, "adversarial_results.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    plot_eps_snr(records, os.path.join(res, "epsilon_snr_plot.png"))
    if min_eps:
        print(f"\n[FGSM] Min valid eps = {min_eps['epsilon']}  SNR = {min_eps['snr_db']} dB")
