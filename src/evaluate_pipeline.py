


import os, json, re
import numpy as np


def check_lid(results_dir):
    p = os.path.join(results_dir, "lid_predictions.json")
    if not os.path.exists(p): return {}
    with open(p) as fh: d = json.load(fh)
    f1  = d.get("best_f1", 0)
    nsw = len(d.get("switch_timestamps", []))
    print(f"[Eval] LID F1         : {f1:.4f}  (target >= 0.85)  {'PASS' if f1>=0.85 else 'FAIL'}")
    print(f"[Eval] Switch events  : {nsw}")
    return {"lid_f1": f1, "n_switches": nsw}


def check_mcd(results_dir):
    p = os.path.join(results_dir, "ablation_mcd.json")
    if not os.path.exists(p): return {}
    with open(p) as fh: d = json.load(fh)
    mw = d.get("mcd_warped", d.get("mcd_warped_synthesis", 999))
    mf = d.get("mcd_flat",   d.get("mcd_flat_synthesis",   999))
    print(f"[Eval] MCD warped     : {mw:.4f} dB  (target < 8.0)  {'PASS' if mw<8 else 'FAIL'}")
    print(f"[Eval] MCD flat       : {mf:.4f} dB")
    return {"mcd_warped": mw, "mcd_flat": mf, "warped_passes": mw < 8.0}


def check_antispoofing(results_dir):
    p = os.path.join(results_dir, "antispoofing_results.json")
    if not os.path.exists(p): return {}
    with open(p) as fh: d = json.load(fh)
    eer = d.get("eer_percent", 999)
    print(f"[Eval] EER            : {eer:.2f}%  (target < 10%)  {'PASS' if eer<10 else 'FAIL'}")
    return {"eer_percent": eer, "passes": eer < 10.0}


def check_adversarial(results_dir):
    p = os.path.join(results_dir, "adversarial_results.json")
    if not os.path.exists(p): return {}
    with open(p) as fh: d = json.load(fh)
    me = d.get("minimum_valid_epsilon")
    if me:
        print(f"[Eval] Min valid eps  : {me['epsilon']}  SNR={me['snr_db']} dB  REPORTED")
    else:
        print("[Eval] Min valid eps  : not found in sweep range")
    return {"minimum_valid_epsilon": me}


def check_wer(results_dir):
    raw_path = os.path.join(results_dir, "transcript_raw.txt")
    if not os.path.exists(raw_path):
        print("[Eval] WER            : transcript_raw.txt not found")
        return {}

    with open(raw_path, encoding="utf-8") as fh:
        lines = [l.strip() for l in fh if l.strip()]
    hyp_lines = [re.sub(r"^\[.*?\]\s*", "", l) for l in lines]
    word_count = sum(len(l.split()) for l in hyp_lines)
    print(f"[Eval] Transcript     : {len(lines)} segments | ~{word_count} words")

    ref_path = os.path.join(results_dir, "reference_transcript.txt")
    if os.path.exists(ref_path):
        from jiwer import wer as compute_wer
        with open(ref_path, encoding="utf-8") as fh:
            ref_text = fh.read()
        hyp_text = " ".join(hyp_lines)
        w = compute_wer(ref_text, hyp_text)
        print(f"[Eval] WER (overall)  : {w*100:.2f}%")
        return {"wer_percent": round(w * 100, 2), "word_count": word_count}

    print("[Eval] WER            : no reference_transcript.txt — place one in "
          f"{results_dir}/ for automatic WER scoring")
    return {"word_count": word_count, "wer_note": "reference not provided"}


def generate_report(results_dir, out_path):
    print("\n" + "="*62)
    print("   FULL PIPELINE EVALUATION REPORT  --  M25DE1071")
    print("="*62)
    report = {
        "lid":          check_lid(results_dir),
        "wer":          check_wer(results_dir),
        "mcd":          check_mcd(results_dir),
        "antispoofing": check_antispoofing(results_dir),
        "adversarial":  check_adversarial(results_dir)
    }
    print("\n--- Passing Criteria ---")
    checks = [
        ("LID F1 >= 0.85",    report["lid"].get("lid_f1", 0) >= 0.85),
        ("MCD < 8.0 dB",      report["mcd"].get("warped_passes", False)),
        ("EER < 10%",         report["antispoofing"].get("passes", False)),
        ("Min eps reported",  report["adversarial"].get("minimum_valid_epsilon") is not None),
    ]
    for label, ok in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {label}  {'✓' if ok else '✗'}")
    print("="*62 + "\n")
    with open(out_path, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"[Eval] Report saved -> {out_path}")


if __name__ == "__main__":
    import yaml
    with open("configs/config.yaml") as fh:
        cfg = yaml.safe_load(fh)
    generate_report(cfg["paths"]["results_dir"],
                    os.path.join(cfg["paths"]["results_dir"], "evaluation_report.json"))
