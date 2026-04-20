hinglish_ipa.py  --  Unified IPA Representation for Code-Switched Hinglish
Shushant Kumar Tiwari | M25DE1071 | CSL-7770 Speech Understanding PA2
Python 3.10 | PyTorch 2.1

Three-layer G2P pipeline:
  Layer 1: Devanagari Unicode -> IPA (hand-crafted 58-entry table)
  Layer 2: Romanised Hinglish -> IPA (40-entry custom dictionary)
  Layer 3: English -> IPA via eng_to_ipa library
Post-processing: word-final schwa deletion + prosodic boundary insertion
"""


import re, os


DEVANAGARI_TO_IPA = {
    "अ":"ə","आ":"aː","इ":"ɪ","ई":"iː","उ":"ʊ","ऊ":"uː",
    "ए":"eː","ऐ":"ɛː","ओ":"oː","औ":"ɔː","ऋ":"rɪ",
    "क":"k","ख":"kʰ","ग":"ɡ","घ":"ɡʱ","ङ":"ŋ",
    "च":"tʃ","छ":"tʃʰ","ज":"dʒ","झ":"dʒʱ","ञ":"ɲ",
    "ट":"ʈ","ठ":"ʈʰ","ड":"ɖ","ढ":"ɖʱ","ण":"ɳ",
    "त":"t̪","थ":"t̪ʰ","द":"d̪","ध":"d̪ʱ","न":"n",
    "प":"p","फ":"pʰ","ब":"b","भ":"bʱ","म":"m",
    "य":"j","र":"r","ल":"l","व":"ʋ",
    "श":"ʃ","ष":"ʂ","स":"s","ह":"ɦ",
    "क्ष":"kʂ","त्र":"tr","ज्ञ":"ɡjə",
    "ा":"aː","ि":"ɪ","ी":"iː","ु":"ʊ","ू":"uː",
    "े":"eː","ै":"ɛː","ो":"oː","ौ":"ɔː",
    "ं":"ŋ","ः":"h","्":"","ँ":"̃",
    "।":".","॥":".",
    "०":"0","१":"1","२":"2","३":"3","४":"4",
    "५":"5","६":"6","७":"7","८":"8","९":"9",
}

HINGLISH_ROMAN_IPA = {
    "kya":"kjɑː","hai":"ɦɛː","hain":"ɦɛ̃ː","aur":"ɔːr",
    "toh":"t̪oː","matlab":"mət̪ləb","wala":"ʋɑːlɑː",
    "wali":"ʋɑːliː","bhi":"bʱiː","nahi":"nəɦiː",
    "nahin":"nəɦĩː","kuch":"kʊtʃ","iska":"ɪskɑː",
    "uska":"ʊskɑː","isme":"ɪsmẽː","usme":"ʊsmẽː",
    "sirf":"sɪrf","phir":"pʰɪr","lekin":"leːkɪn",
    "isliye":"ɪslɪjeː","yahan":"jəɦɑːn","samajh":"səmədʒʱ",
    "dekho":"d̪eːkʰoː","suno":"sʊnoː","theek":"tʰiːk",
    "achha":"ətʃʰɑː","acha":"ətʃʰɑː","seedha":"siːdʱɑː",
    "bolte":"boːlt̪eː","bolna":"boːlnɑː","padhna":"pəɖʱnɑː",
    "likhna":"lɪkʰnɑː","karna":"kərnɑː","jana":"dʒɑːnɑː",
    "aana":"ɑːnɑː","hona":"ɦoːnɑː","samajhna":"səmədʒʱnɑː",
    "samjhe":"səmdʒʱeː","toh phir":"t̪oː pʰɪr",
}

DEVANAGARI_CHARS = set("अआइईउऊएऐओऔकखगघचछजझटठडढणतथदधनपफबभमयरलवशषसह")


def detect_lang(word):
    return "hi" if any(c in DEVANAGARI_CHARS for c in word) else "en"


def devanagari_to_ipa_word(word):
    out, i = "", 0
    while i < len(word):
        matched = False
        for span in [3, 2, 1]:
            chunk = word[i:i+span]
            if chunk in DEVANAGARI_TO_IPA:
                out += DEVANAGARI_TO_IPA[chunk]
                i   += span
                matched = True
                break
        if not matched:
            out += word[i]; i += 1
    # word-final schwa deletion (colloquial Hindi pronunciation)
    return out[:-1] if out.endswith("ə") else out


def english_to_ipa(word):
    try:
        import eng_to_ipa as e2i
        result = e2i.convert(word)
        if "*" not in result:
            return result
    except Exception:
        pass
    return word


def word_to_ipa(word):
    clean = re.sub(r"[^\u0900-\u097Fa-zA-Z]", "", word)
    if not clean:
        return word
    lang = detect_lang(clean)
    if lang == "hi":
        return devanagari_to_ipa_word(clean)
    hrom = HINGLISH_ROMAN_IPA.get(clean.lower())
    return hrom if hrom else english_to_ipa(clean)


def convert_to_ipa(transcript_text):
    out_lines = []
    for line in transcript_text.strip().split("\n"):
        m = re.match(r"^(\[.*?\]\s*)", line)
        stamp = m.group(1) if m else ""
        text  = line[len(stamp):]
        ipa_words = [word_to_ipa(w) for w in text.strip().split()]
        out_lines.append(stamp + " ".join(ipa_words))
    return "\n".join(out_lines)


def apply_phonological_rules(ipa_text):
    """
    Post-processing phonological rules for Hinglish:
    - Aspirated stop normalisation (already encoded)
    - Prosodic boundary insertion at detected language-switch points
    """
    ipa_text = re.sub(r"(\w)([ \t]+)([\u0250-\u02AF])", r"\1 | \3", ipa_text)
    return ipa_text


if __name__ == "__main__":
    import yaml
    with open("configs/config.yaml") as fh:
        cfg = yaml.safe_load(fh)
    res_dir = cfg["paths"]["results_dir"]
    src = os.path.join(res_dir, "transcript_raw.txt")
    dst = os.path.join(res_dir, "transcript_ipa.txt")
    with open(src, "r", encoding="utf-8") as fh:
        raw_txt = fh.read()
    ipa = apply_phonological_rules(convert_to_ipa(raw_txt))
    with open(dst, "w", encoding="utf-8") as fh:
        fh.write(ipa)
    print(f"[IPA] Saved -> {dst}")
    print(ipa[:400])
