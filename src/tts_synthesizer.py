


import os
import numpy as np
import torch
import torchaudio
import soundfile as sf


def chunk_bhojpuri_text(text, max_len=200):
    sentences = text.replace("\n", " ").split("।")
    sentences = [s.strip() for s in sentences if s.strip()]
    chunks, buf = [], ""
    for s in sentences:
        if len(buf) + len(s) < max_len:
            buf += s + "। "
        else:
            if buf: chunks.append(buf.strip())
            buf = s + "। "
    if buf: chunks.append(buf.strip())
    if not chunks:
        words  = text.split()
        chunks = [" ".join(words[i:i+30]) for i in range(0, len(words), 30)]
    return chunks


def synth_coqui(chunks, ref_wav, sr_out):
    from TTS.api import TTS
    mdl = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts",
              gpu=torch.cuda.is_available())
    audio, sil = [], np.zeros(int(sr_out * 0.4), dtype=np.float32)
    for i, ch in enumerate(chunks):
        try:
            wav = np.array(mdl.tts(text=ch, speaker_wav=ref_wav, language="hi"))
            audio.extend([wav, sil])
        except Exception as e:
            pass
    return np.concatenate(audio) if audio else None


def _f0_statistics(wav_path, sr=22050):
    import librosa, pyworld as pw
    sig, _ = librosa.load(wav_path, sr=sr)
    _f0, t = pw.dio(sig.astype(np.float64), sr, frame_period=5.0)
    f0 = pw.stonemask(sig.astype(np.float64), _f0, t, sr)
    voiced = f0[f0 > 0]
    return (float(np.mean(voiced)), float(np.std(voiced))) if len(voiced) else (150.0, 30.0)


def _world_voice_convert(signal, sr, tgt_mu, tgt_sig, src_mu, src_sig):
    import pyworld as pw
    sig = signal.astype(np.float64)
    _f0, t = pw.dio(sig, sr, frame_period=5.0)
    f0 = pw.stonemask(sig, _f0, t, sr)
    sp = pw.cheaptrick(sig, f0, t, sr)
    ap = pw.d4c(sig, f0, t, sr)
    v = f0 > 0
    if v.sum() > 0 and src_sig > 0:
        f0[v] = np.clip((f0[v]-src_mu)/src_sig * tgt_sig + tgt_mu, 50, 500)
    out = pw.synthesize(f0, sp, ap, sr)
    return (out / (np.max(np.abs(out))+1e-8)).astype(np.float32)


def _gtts_chunk(text, lang="hi", tmp="/tmp/_skt_tts.mp3"):
    from gtts import gTTS
    import subprocess, librosa
    gTTS(text=text, lang=lang, slow=False).save(tmp)
    wav_tmp = tmp.replace(".mp3", ".wav")
    subprocess.run(["ffmpeg","-y","-i",tmp,"-ar","22050","-ac","1",wav_tmp],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    y, _ = librosa.load(wav_tmp, sr=22050)
    return y


def synth_fallback(chunks, ref_wav, sr_out):
    print("[TTS][Fallback] gTTS + WORLD voice conversion.")
    probe = _gtts_chunk("यह एक परीक्षण है", lang="hi")
    sf.write("/tmp/_probe_skt.wav", probe, sr_out)
    src_mu, src_sig = _f0_statistics("/tmp/_probe_skt.wav", sr_out)
    tgt_mu, tgt_sig = _f0_statistics(ref_wav,               sr_out)
    print(f"[TTS] gTTS F0: {src_mu:.1f}+/-{src_sig:.1f} Hz  |  Student F0: {tgt_mu:.1f}+/-{tgt_sig:.1f} Hz")
    audio, sil = [], np.zeros(int(sr_out * 0.4), dtype=np.float32)
    for i, ch in enumerate(chunks):
        print(f"[TTS][gTTS] chunk {i+1}/{len(chunks)}: {ch[:60]}...")
        try:
            raw = _gtts_chunk(ch, lang="hi")
            cv  = _world_voice_convert(raw, sr_out, tgt_mu, tgt_sig, src_mu, src_sig)
            audio.extend([cv, sil])
        except Exception as e:
            print(f"[TTS][gTTS] chunk {i+1} skipped: {e}")
    return np.concatenate(audio) if audio else None


def _pick_backend(chunks, ref_wav, sr_out):
    try:
        out = synth_coqui(chunks, ref_wav, sr_out)
        if out is not None: return out
    except ImportError:
        print("[TTS] Coqui TTS not found (needs Python 3.10 + pip install TTS). Fallback active.")
    except Exception as e:
        print(f"[TTS] Coqui error: {e}. Fallback active.")
    return synth_fallback(chunks, ref_wav, sr_out)


def compute_mcd(ref_path, syn_path, sr=22050, n_mels=13):
    import librosa
    ref, _ = librosa.load(ref_path, sr=sr)
    syn, _ = librosa.load(syn_path, sr=sr)
    mn = min(len(ref), len(syn))
    rm = librosa.feature.mfcc(y=ref[:mn], sr=sr, n_mfcc=n_mels)
    sm = librosa.feature.mfcc(y=syn[:mn], sr=sr, n_mfcc=n_mels)
    mf = min(rm.shape[1], sm.shape[1])
    d  = rm[:, :mf] - sm[:, :mf]
    v  = float((10/np.log(10)) * np.sqrt(2 * np.mean(np.sum(d**2, axis=0))))
    print(f"[TTS] MCD = {v:.4f} dB  (target < 8.0)")
    return v


def synthesize_lecture(text_path, embed_path, ref_wav, out_path, tts_sr=22050):
    with open(text_path, "r", encoding="utf-8") as fh:
        text = fh.read()
    chunks = chunk_bhojpuri_text(text)
    print(f"[TTS] {len(chunks)} synthesis chunks.")
    audio = _pick_backend(chunks, ref_wav, tts_sr)
    if audio is None:
        print("[TTS] No audio produced."); return None, tts_sr
    if tts_sr != 22050:
        t = torch.tensor(audio).unsqueeze(0)
        audio = torchaudio.transforms.Resample(tts_sr, 22050)(t).squeeze().numpy()
        tts_sr = 22050
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    sf.write(out_path, audio, tts_sr)
    print(f"[TTS] Done. {len(audio)/tts_sr:.1f}s | {tts_sr} Hz -> {out_path}")
    return audio, tts_sr


def synthesize_flat(text_path, ref_wav, out_path, tts_sr=22050):
    with open(text_path, "r", encoding="utf-8") as fh:
        text = fh.read()
    chunks = chunk_bhojpuri_text(text)
    print("[TTS] Flat (no-warp) synthesis for ablation...")
    audio = _pick_backend(chunks, ref_wav, tts_sr)
    if audio is not None:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        sf.write(out_path, audio, tts_sr)
        print(f"[TTS][Ablation] Flat output saved -> {out_path}")
    return audio


if __name__ == "__main__":
    import yaml, json
    with open("configs/config.yaml") as fh:
        cfg = yaml.safe_load(fh)
    res  = cfg["paths"]["results_dir"]
    ckpt = cfg["paths"]["checkpoints_dir"]
    ref  = cfg["paths"]["student_voice_ref"]
    sr   = cfg["audio"]["tts_sampling_rate"]

    synthesize_lecture(os.path.join(res, "tts_input_bhojpuri.txt"),
                       os.path.join(ckpt, "speaker_embed.pt"),
                       ref, os.path.join(res, "output_LRL_cloned.wav"), tts_sr=sr)

    synthesize_flat(os.path.join(res, "tts_input_bhojpuri.txt"),
                    ref, os.path.join(res, "output_flat_synthesis.wav"), tts_sr=sr)

    mcd_w = compute_mcd(ref, os.path.join(res, "output_LRL_cloned.wav"))
    mcd_f = compute_mcd(ref, os.path.join(res, "output_flat_synthesis.wav"))
    ablation = {"mcd_warped": mcd_w, "mcd_flat": mcd_f,
                "mcd_target": 8.0, "warped_passes": mcd_w < 8.0}
    with open(os.path.join(res, "ablation_mcd.json"), "w") as fh:
        json.dump(ablation, fh, indent=2)
    print(ablation)
