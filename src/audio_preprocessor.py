audio_preprocessor.py  --  Denoising & Normalization of Lecture Audio
Shushant Kumar Tiwari | M25DE1071 | CSL-7770 Speech Understanding PA2
Python 3.10 | PyTorch 2.1

Primary denoiser  : DeepFilterNet (deep spectral filtering)
Fallback denoiser : Spectral Subtraction (over-subtraction with floor beta=0.01)
"""


import os, subprocess
import numpy as np
import torch
import torchaudio
import soundfile as sf
from scipy.signal import stft, istft


def read_audio_file(filepath, target_sr=16000):
    if filepath.endswith((".mp4", ".m4a", ".mp3", ".aac")):
        tmp_wav = filepath.rsplit(".", 1)[0] + "_decoded.wav"
        subprocess.run(["ffmpeg", "-y", "-i", filepath,
                        "-ar", str(target_sr), "-ac", "1", tmp_wav],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        data, sr = sf.read(tmp_wav)
        os.remove(tmp_wav)
    else:
        data, sr = sf.read(filepath)
        if data.ndim > 1:
            data = data.mean(axis=1)

    wav = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav, target_sr


def spectral_subtraction_denoise(waveform, sr, n_noise_frames=30, oversubtract=2.0, floor=0.01):
    sig = waveform.squeeze().numpy()
    _, _, Zxx = stft(sig, fs=sr, nperseg=512, noverlap=256)
    mag, phase = np.abs(Zxx), np.angle(Zxx)
    noise_ref  = np.mean(mag[:, :n_noise_frames], axis=1, keepdims=True)
    mag_clean  = np.maximum(mag - oversubtract * noise_ref, floor * mag)
    _, cleaned = istft(mag_clean * np.exp(1j * phase), fs=sr, nperseg=512, noverlap=256)
    cleaned   /= (np.max(np.abs(cleaned)) + 1e-8)
    return torch.tensor(cleaned, dtype=torch.float32).unsqueeze(0)


def deepfilternet_denoise(waveform, sr):
    try:
        from df.enhance import enhance, init_df
        import librosa
        mdl, state, _ = init_df()
        sig = waveform.squeeze(0).numpy()
        if sr != state.sr():
            sig = librosa.resample(sig, orig_sr=sr, target_sr=state.sr())
        enhanced = enhance(mdl, state, torch.from_numpy(sig).unsqueeze(0))
        out = enhanced.squeeze().numpy()
        if state.sr() != sr:
            out = librosa.resample(out, orig_sr=state.sr(), target_sr=sr)
        return torch.tensor(out, dtype=torch.float32).unsqueeze(0)
    except Exception as exc:
        print(f"[Preproc] DeepFilterNet unavailable ({exc}). Switching to spectral subtraction.")
        return None


def rms_normalize(waveform, target_level=0.1):
    rms = torch.sqrt(torch.mean(waveform**2))
    return waveform * (target_level / rms) if rms > 0 else waveform


def crop_segment(waveform, sr, offset_sec=0, duration_sec=600):
    s = offset_sec * sr
    e = s + duration_sec * sr
    return waveform[:, s:e]


def process_lecture(raw_path, out_path, cfg):
    sr    = cfg["audio"]["sampling_rate"]
    start = cfg["audio"].get("segment_start_sec", 0)
    dur   = cfg["audio"]["lecture_segment_duration"]

    print(f"[Preproc] Reading lecture: {raw_path}")
    wav, sr = read_audio_file(raw_path, sr)
    print(f"[Preproc] Duration: {wav.shape[1]/sr:.1f}s | SR: {sr}Hz")

    print("[Preproc] Attempting DeepFilterNet denoising...")
    denoised = deepfilternet_denoise(wav, sr)
    if denoised is None:
        print("[Preproc] Applying spectral subtraction...")
        denoised = spectral_subtraction_denoise(wav, sr)

    print(f"[Preproc] Cropping segment: {start}s -- {start+dur}s")
    segment = crop_segment(denoised, sr, offset_sec=start, duration_sec=dur)
    segment = rms_normalize(segment)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    sf.write(out_path, segment.squeeze().numpy(), sr)
    print(f"[Preproc] Lecture segment saved -> {out_path}")


def process_voice_ref(raw_path, out_path, cfg):
    sr  = cfg["audio"]["sampling_rate"]
    dur = cfg["audio"]["voice_ref_duration"]
    print(f"[Preproc] Reading voice reference: {raw_path}")
    wav, sr = read_audio_file(raw_path, sr)
    wav = wav[:, :dur*sr]
    wav = rms_normalize(wav)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    sf.write(out_path, wav.squeeze().numpy(), sr)
    print(f"[Preproc] Voice ref saved -> {out_path}  ({dur}s)")


if __name__ == "__main__":
    import yaml
    with open("configs/config.yaml") as fh:
        cfg = yaml.safe_load(fh)
    process_lecture(cfg["paths"]["raw_lecture"],
                    cfg["paths"]["original_segment"], cfg)
    process_voice_ref(cfg["paths"]["voice_ref_raw"],
                      cfg["paths"]["student_voice_ref"], cfg)
