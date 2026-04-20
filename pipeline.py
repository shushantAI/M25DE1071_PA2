"""
pipeline.py  --  End-to-End Pipeline Runner
Shushant Kumar Tiwari | M25DE1071 | CSL-7770 Speech Understanding PA2
Python 3.10 | PyTorch 2.1

Usage:
    python pipeline.py --stage all
    python pipeline.py --stage preprocess
    python pipeline.py --stage lid
    python pipeline.py --stage asr
    python pipeline.py --stage ipa
    python pipeline.py --stage translate
    python pipeline.py --stage embed
    python pipeline.py --stage dtw
    python pipeline.py --stage tts
    python pipeline.py --stage cm
    python pipeline.py --stage fgsm
    python pipeline.py --stage eval
"""

import os, sys, argparse, yaml, torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))


def load_cfg(path="configs/config.yaml"):
    with open(path) as fh:
        return yaml.safe_load(fh)


def init_dirs(cfg):
    for key in ["results_dir", "checkpoints_dir"]:
        os.makedirs(cfg["paths"][key], exist_ok=True)
    for sub in ["data/processed", "data/corpus", "data/raw"]:
        os.makedirs(sub, exist_ok=True)
    print("[Pipeline] Output directories ready.")


# Stage runners

def stage_preprocess(cfg):
    print("\n[Pipeline] >>> Stage: Preprocessing (Task 1.3)")
    from audio_preprocessor import process_lecture, process_voice_ref
    process_lecture(cfg["paths"]["raw_lecture"], cfg["paths"]["original_segment"], cfg)
    process_voice_ref(cfg["paths"]["voice_ref_raw"], cfg["paths"]["student_voice_ref"], cfg)
    print("[Pipeline] Preprocessing done.\n")


def stage_lid(cfg):
    print("\n[Pipeline] >>> Stage: Language Identification (Task 1.1)")
    from lid_model import (W2VFeatureExtractor, build_frame_labels,
                           fit_lid_model, run_inference,
                           get_switch_timestamps, save_confusion_matrix, load_wav_sf)
    import json, numpy as np

    device    = "cuda" if torch.cuda.is_available() else "cpu"
    seg_path  = cfg["paths"]["original_segment"]
    ckpt_path = os.path.join(cfg["paths"]["checkpoints_dir"], "lid_weights.pt")
    res_dir   = cfg["paths"]["results_dir"]

    ext   = W2VFeatureExtractor(cfg["lid_config"]["backbone"], device)
    wav, sr = load_wav_sf(seg_path)
    feats = ext.get_features(wav, sr).numpy()
    labs  = build_frame_labels(seg_path, cfg["lid_config"]["hop_ms"], sr)
    n     = min(len(feats), len(labs))
    feats, labs = feats[:n], labs[:n]

    best_f1 = fit_lid_model(feats, labs, cfg["lid_config"], ckpt_path, device)
    preds   = run_inference(seg_path, ckpt_path, cfg["lid_config"], device)
    events  = get_switch_timestamps(preds, cfg["lid_config"]["hop_ms"])

    with open(os.path.join(res_dir, "lid_predictions.json"), "w") as fh:
        json.dump({"predictions_sample": preds[:200].tolist(),
                   "switch_timestamps": events[:50], "best_f1": best_f1}, fh, indent=2)
    save_confusion_matrix(labs[:len(preds)], preds,
                          os.path.join(res_dir, "lid_confusion_matrix.png"))
    print("[Pipeline] LID done.\n")


def stage_asr(cfg):
    print("\n[Pipeline] >>> Stage: Constrained ASR (Task 1.2)")
    from constrained_asr import transcribe_hinglish
    transcribe_hinglish(cfg["paths"]["original_segment"],
                        cfg["paths"]["syllabus_terms"],
                        cfg,
                        os.path.join(cfg["paths"]["results_dir"], "transcript_raw.txt"))
    print("[Pipeline] ASR done.\n")


def stage_ipa(cfg):
    print("\n[Pipeline] >>> Stage: IPA Mapping (Task 2.1)")
    from hinglish_ipa import convert_to_ipa, apply_phonological_rules
    res = cfg["paths"]["results_dir"]
    with open(os.path.join(res, "transcript_raw.txt"), encoding="utf-8") as fh:
        raw = fh.read()
    ipa = apply_phonological_rules(convert_to_ipa(raw))
    with open(os.path.join(res, "transcript_ipa.txt"), "w", encoding="utf-8") as fh:
        fh.write(ipa)
    print(f"[Pipeline] IPA saved. Sample: {ipa[:200]}\n")


def stage_translate(cfg):
    print("\n[Pipeline] >>> Stage: Bhojpuri Translation (Task 2.2)")
    from bhojpuri_translator import translate_transcript, strip_timestamps
    res = cfg["paths"]["results_dir"]
    translate_transcript(os.path.join(res, "transcript_raw.txt"),
                         cfg["paths"]["bhojpuri_corpus"],
                         os.path.join(res, "transcript_bhojpuri.txt"))
    strip_timestamps(os.path.join(res, "transcript_bhojpuri.txt"),
                     os.path.join(res, "tts_input_bhojpuri.txt"))
    print("[Pipeline] Translation done.\n")


def stage_embed(cfg):
    print("\n[Pipeline] >>> Stage: Speaker Embedding (Task 3.1)")
    from voice_embedding import VoicePrintExtractor, load_wav_sf, segment_consistency
    import json
    device = "cuda" if torch.cuda.is_available() else "cpu"
    wav, sr = load_wav_sf(cfg["paths"]["student_voice_ref"],
                          max_dur_sec=cfg["audio"]["voice_ref_duration"])
    ext = VoicePrintExtractor(device)
    emb = ext.extract(wav, sr)
    ckpt = os.path.join(cfg["paths"]["checkpoints_dir"], "speaker_embed.pt")
    torch.save(emb, ckpt)
    msim = segment_consistency(wav, sr, ext)
    with open(os.path.join(cfg["paths"]["results_dir"], "speaker_embed_info.json"), "w") as fh:
        json.dump({"dim": emb.shape[0], "mean_cosine_sim": msim}, fh, indent=2)
    print("[Pipeline] Embedding done.\n")


def stage_dtw(cfg):
    print("\n[Pipeline] >>> Stage: Prosody Warping DTW (Task 3.2)")
    from prosody_dtw import warp_prosody, plot_f0_comparison
    res = cfg["paths"]["results_dir"]
    pf0, sf0, wf0, _, _ = warp_prosody(
        cfg["paths"]["original_segment"],
        cfg["paths"]["student_voice_ref"],
        os.path.join(res, "prosody_warped_ref.wav"), cfg)
    plot_f0_comparison(pf0, sf0, wf0, os.path.join(res, "prosody_f0_plot.png"))
    print("[Pipeline] DTW done.\n")


def stage_tts(cfg):
    print("\n[Pipeline] >>> Stage: Bhojpuri TTS Synthesis (Task 3.3)")
    from tts_synthesizer import synthesize_lecture, synthesize_flat, compute_mcd
    import json
    res = cfg["paths"]["results_dir"]; ckpt = cfg["paths"]["checkpoints_dir"]
    sr  = cfg["audio"]["tts_sampling_rate"]
    ref = cfg["paths"]["student_voice_ref"]
    synthesize_lecture(os.path.join(res, "tts_input_bhojpuri.txt"),
                       os.path.join(ckpt, "speaker_embed.pt"),
                       ref, os.path.join(res, "output_LRL_cloned.wav"), sr)
    synthesize_flat(os.path.join(res, "tts_input_bhojpuri.txt"),
                    ref, os.path.join(res, "output_flat_synthesis.wav"), sr)
    mcd_w = compute_mcd(ref, os.path.join(res, "output_LRL_cloned.wav"))
    mcd_f = compute_mcd(ref, os.path.join(res, "output_flat_synthesis.wav"))
    with open(os.path.join(res, "ablation_mcd.json"), "w") as fh:
        json.dump({"mcd_warped": mcd_w, "mcd_flat": mcd_f,
                   "target": 8.0, "warped_passes": mcd_w < 8.0}, fh, indent=2)
    print("[Pipeline] TTS done.\n")


def stage_cm(cfg):
    print("\n[Pipeline] >>> Stage: Anti-Spoofing CM (Task 4.1)")
    import subprocess
    subprocess.run([sys.executable, "src/antispoofing.py"], cwd=os.getcwd())
    print("[Pipeline] CM done.\n")


def stage_fgsm(cfg):
    print("\n[Pipeline] >>> Stage: FGSM Adversarial Attack (Task 4.2)")
    import subprocess
    subprocess.run([sys.executable, "src/fgsm_attack.py"], cwd=os.getcwd())
    print("[Pipeline] FGSM done.\n")


def stage_eval(cfg):
    print("\n[Pipeline] >>> Stage: Full Evaluation")
    from evaluate_pipeline import generate_report
    generate_report(cfg["paths"]["results_dir"],
                    os.path.join(cfg["paths"]["results_dir"], "evaluation_report.json"))
    print("[Pipeline] Evaluation done.\n")


STAGE_ORDER = ["preprocess", "lid", "asr", "ipa", "translate",
               "embed", "dtw", "tts", "cm", "fgsm", "eval"]
STAGE_MAP   = {
    "preprocess": stage_preprocess, "lid":       stage_lid,
    "asr":        stage_asr,        "ipa":       stage_ipa,
    "translate":  stage_translate,  "embed":     stage_embed,
    "dtw":        stage_dtw,        "tts":       stage_tts,
    "cm":         stage_cm,         "fgsm":      stage_fgsm,
    "eval":       stage_eval,
}


def main():
    parser = argparse.ArgumentParser(description="M25DE1071 PA2 Pipeline")
    parser.add_argument("--stage", default="all",
                        choices=["all"] + STAGE_ORDER)
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    init_dirs(cfg)

    stages = STAGE_ORDER if args.stage == "all" else [args.stage]
    for s in stages:
        STAGE_MAP[s](cfg)
    print("\n[Pipeline] All stages complete.")


if __name__ == "__main__":
    main()
