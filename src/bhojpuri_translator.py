bhojpuri_translator.py  --  Code-Switched Hinglish -> Bhojpuri Translation
Shushant Kumar Tiwari | M25DE1071 | CSL-7770 Speech Understanding PA2
Python 3.10 | PyTorch 2.1

Primary  : IndicTrans2 (ai4bharat/indictrans2-indic-indic-dist-200M)
Fallback : Token-level lookup in 244-entry parallel technical corpus
"""


import os, re, csv


def load_parallel_corpus(csv_path):
    en2bho, hi2bho = {}, {}
    with open(csv_path, "r", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            en2bho[row["english"].lower().strip()] = row["bhojpuri"].strip()
            hi2bho[row["hindi"].strip()]            = row["bhojpuri"].strip()
    return en2bho, hi2bho


def indictrans2_translate(text, src="hin_Deva", tgt="bho_Deva"):
    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        mname = "ai4bharat/indictrans2-indic-indic-dist-200M"
        tok = AutoTokenizer.from_pretrained(mname, trust_remote_code=True)
        mdl = AutoModelForSeq2SeqLM.from_pretrained(mname, trust_remote_code=True)
        ids = tok(text, return_tensors="pt", padding=True,
                  truncation=True, max_length=256).input_ids
        out = mdl.generate(ids, max_new_tokens=256, num_beams=4)
        return tok.decode(out[0], skip_special_tokens=True)
    except Exception as exc:
        print(f"[Trans] IndicTrans2 unavailable: {exc}. Using corpus fallback.")
        return None


DEVANAGARI_CHARS = set("अआइईउऊएऐओऔकखगघचछजझटठडढणतथदधनपफबभमयरलवशषसह")

def is_hindi_token(w):
    return any(c in DEVANAGARI_CHARS for c in w)


def corpus_translate_line(line, en2bho, hi2bho):
    m = re.match(r"^(\[.*?\]\s*)", line)
    stamp = m.group(1) if m else ""
    text  = line[len(stamp):]
    out   = []
    for word in text.strip().split():
        key = re.sub(r"[^\u0900-\u097Fa-zA-Z\-]", "", word).lower()
        if key in en2bho:
            out.append(en2bho[key])
        elif key in hi2bho or word.strip() in hi2bho:
            out.append(hi2bho.get(key, hi2bho.get(word.strip(), word)))
        else:
            out.append(word)
    return stamp + " ".join(out)


def translate_transcript(src_path, corpus_path, dst_path):
    print("[Trans] Loading parallel corpus...")
    en2bho, hi2bho = load_parallel_corpus(corpus_path)
    print(f"[Trans] Corpus: {len(en2bho)} EN, {len(hi2bho)} HI entries")

    with open(src_path, "r", encoding="utf-8") as fh:
        lines = fh.readlines()

    output_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            output_lines.append(""); continue

        bho_line = corpus_translate_line(line, en2bho, hi2bho)
        plain    = re.sub(r"^\[.*?\]\s*", "", line)
        it2_out  = indictrans2_translate(plain)
        if it2_out:
            m = re.match(r"^(\[.*?\]\s*)", line)
            bho_line = (m.group(1) if m else "") + it2_out
        output_lines.append(bho_line)

    result = "\n".join(output_lines)
    os.makedirs(os.path.dirname(dst_path) or ".", exist_ok=True)
    with open(dst_path, "w", encoding="utf-8") as fh:
        fh.write(result)
    print(f"[Trans] Bhojpuri transcript saved -> {dst_path}")
    return result


def strip_timestamps(src_path, dst_path):
    with open(src_path, "r", encoding="utf-8") as fh:
        lines = fh.readlines()
    plain = "\n".join(re.sub(r"^\[.*?\]\s*", "", l.strip())
                      for l in lines if l.strip())
    with open(dst_path, "w", encoding="utf-8") as fh:
        fh.write(plain)
    print(f"[Trans] TTS input saved -> {dst_path}")
    return plain


if __name__ == "__main__":
    import yaml
    with open("configs/config.yaml") as fh:
        cfg = yaml.safe_load(fh)
    res = cfg["paths"]["results_dir"]
    translate_transcript(os.path.join(res, "transcript_raw.txt"),
                         cfg["paths"]["bhojpuri_corpus"],
                         os.path.join(res, "transcript_bhojpuri.txt"))
    strip_timestamps(os.path.join(res, "transcript_bhojpuri.txt"),
                     os.path.join(res, "tts_input_bhojpuri.txt"))
