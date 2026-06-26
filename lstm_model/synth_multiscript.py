"""
Multi-script synthetic scene-text generator for the best-system track.

CORRECTNESS GUARANTEE (coverage-aware): a word is rendered ONLY with a font
whose cmap covers EVERY codepoint in that word. PIL+libraqm then shapes Indic
conjuncts/vowel-reordering correctly (libraqm is verified present). Words no
available font can fully cover (e.g. code-mixed / foreign-script label noise)
are SKIPPED, never rendered as .notdef tofu boxes. This makes every synthetic
image's pixels match its ground-truth label exactly.

For a given language:
  1. Collect fonts that cover the script (cmap), system + downloaded Noto.
  2. Lexicon from the language's REAL official-train labels (no invented words).
  3. Oversample words rich in RARE grapheme clusters (Unicode \X segmentation).
  4. Per word: choose among fonts that FULLY cover it, preferring script-specific
     fonts over the generic Free* fallbacks.
  5. Render with the existing degradation pipeline (render_text_image).
  6. Write florence2_splits_synth_<lang>_full.json (train = synthetic + real train).
"""
import os, sys, json, random, argparse, urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import regex  # \X extended grapheme clusters (script-agnostic)
from collections import Counter
from fontTools.ttLib import TTFont, TTLibError
from generate_synthetic import render_text_image

BASE = os.path.dirname(os.path.abspath(__file__))
FONT_CACHE = os.path.join(BASE, "fonts_synth")
SEED = 42

SCRIPT = {
    "bengali":   (0x0995, "Bengali"),
    "assamese":  (0x0995, "Bengali"),
    "tamil":     (0x0B95, "Tamil"),
    "telugu":    (0x0C15, "Telugu"),
    "kannada":   (0x0C95, "Kannada"),
    "malayalam": (0x0D15, "Malayalam"),
    "gujarati":  (0x0A95, "Gujarati"),
    "hindi":     (0x0915, "Devanagari"),
    "marathi":   (0x0915, "Devanagari"),
    "punjabi":   (0x0A15, "Gurmukhi"),
    "odia":      (0x0B15, "Oriya"),
}
SYS_FONT_DIRS = ["/usr/share/fonts", "/usr/local/share/fonts"]

def font_cmap(path):
    """Return the set of codepoints the font's best cmap maps, or empty set."""
    try:
        f = TTFont(path, fontNumber=0, lazy=True)
        cm = set(f.getBestCmap().keys())
        f.close()
        return cm
    except (TTLibError, Exception):
        return set()

def system_fonts():
    out = []
    for d in SYS_FONT_DIRS:
        if not os.path.isdir(d):
            continue
        for r, _, fs in os.walk(d):
            for fn in fs:
                if fn.lower().endswith((".ttf", ".otf")):  # skip .ttc collections
                    out.append(os.path.join(r, fn))
    return out

def download_noto(noto_name):
    """Download Noto Sans + Serif Regular static TTF for the script."""
    os.makedirs(FONT_CACHE, exist_ok=True)
    got = []
    for style in ("Sans", "Serif"):
        fam = f"Noto{style}{noto_name}"
        out = os.path.join(FONT_CACHE, f"{fam}-Regular.ttf")
        if os.path.exists(out) and os.path.getsize(out) > 1000:
            got.append(out); continue
        urls = [
            f"https://github.com/notofonts/notofonts.github.io/raw/main/fonts/{fam}/hinted/ttf/{fam}-Regular.ttf",
            f"https://github.com/notofonts/notofonts.github.io/raw/main/fonts/{fam}/unhinted/ttf/{fam}-Regular.ttf",
        ]
        for u in urls:
            try:
                urllib.request.urlretrieve(u, out)
                if os.path.getsize(out) > 1000:
                    got.append(out); break
            except Exception:
                if os.path.exists(out):
                    os.remove(out)
    return got

def collect_fonts(lang):
    """Return list of font paths whose cmap contains the script's base consonant."""
    cp, noto_name = SCRIPT[lang]
    fonts = [p for p in system_fonts() if cp in font_cmap(p)]
    for p in download_noto(noto_name):
        if cp in font_cmap(p) and p not in fonts:
            fonts.append(p)
    fonts = sorted(set(fonts))
    if not fonts:
        raise SystemExit(f"[{lang}] no fonts cover U+{cp:04X} — cannot synthesize")
    print(f"[{lang}] {len(fonts)} candidate fonts:")
    for p in fonts:
        print("   ", os.path.basename(p))
    return fonts

def real_words(split_file):
    s = json.load(open(split_file))
    words = []
    for r in s["train"]:
        t = (r.get("gt") or "").strip()
        if t and t != "###":
            words.append(t)
    return words, s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True)
    ap.add_argument("--num", type=int, default=8000)
    ap.add_argument("--splits_file", default=None)
    args = ap.parse_args()
    random.seed(SEED)

    split_file = args.splits_file or os.path.join(
        BASE, f"florence2_splits_bstd_{args.lang}_full.json")
    if not os.path.exists(split_file):
        raise SystemExit(f"missing {split_file} — run build_full_splits.py first")

    fonts = collect_fonts(args.lang)
    cmaps = {p: font_cmap(p) for p in fonts}
    # prefer script-specific fonts; Free* are generic Latin fonts w/ partial Indic
    good = [p for p in fonts if not os.path.basename(p).startswith("Free")]
    fallback = [p for p in fonts if os.path.basename(p).startswith("Free")]

    def fonts_for(word):
        cps = set(ord(c) for c in word)
        g = [p for p in good if cps <= cmaps[p]]
        if g:
            return g
        return [p for p in fallback if cps <= cmaps[p]]

    words, real = real_words(split_file)
    if not words:
        raise SystemExit(f"[{args.lang}] no real words in {split_file}")

    # rare-grapheme oversampling weights, restricted to RENDERABLE words
    gf = Counter()
    for w in words:
        gf.update(regex.findall(r"\X", w))
    uniq = sorted(set(words))
    renderable, rweights, skipped = [], [], 0
    for w in uniq:
        if fonts_for(w):
            renderable.append(w)
            rweights.append(sum(1.0 / (gf[g] + 1) for g in regex.findall(r"\X", w)))
        else:
            skipped += 1
    print(f"[{args.lang}] renderable unique words: {len(renderable)}/{len(uniq)} "
          f"(skipped {skipped} no-font-covers)")
    if not renderable:
        raise SystemExit(f"[{args.lang}] no renderable words")

    out_img = os.path.join(BASE, f"synth_{args.lang}", "images")
    os.makedirs(out_img, exist_ok=True)
    chosen = random.choices(renderable, weights=rweights, k=args.num)

    syn_entries, ok, fail = [], 0, 0
    for i, text in enumerate(chosen):
        try:
            fp = random.choice(fonts_for(text))
            img = render_text_image(text, fp)
            name = f"syn_{args.lang}_{i:06d}"
            p = os.path.join(out_img, name + ".jpg")
            img.save(p, "JPEG", quality=90)
            syn_entries.append({"image": p, "gt": text, "name": name,
                                "font": os.path.basename(fp)})
            ok += 1
        except Exception as e:
            fail += 1
            if fail <= 5:
                print(f"  warn: {e}")
        if (i + 1) % 2000 == 0:
            print(f"  {i+1}/{args.num} ({ok} ok, {fail} fail)")

    out = {
        "train": syn_entries + real["train"],
        "val": real["val"], "test": real["test"],
        "metadata": {**real.get("metadata", {}), "lang": args.lang,
                     "synthetic_added": ok, "real_train": len(real["train"]),
                     "n_fonts": len(fonts), "renderable_words": len(renderable),
                     "skipped_words": skipped, "coverage_aware": True,
                     "synth_source": "rare-grapheme-weighted"},
    }
    fn = os.path.join(BASE, f"florence2_splits_synth_{args.lang}_full.json")
    json.dump(out, open(fn, "w"), ensure_ascii=False)
    print(f"[{args.lang}] {ok} synthetic (+{fail} failed) -> {os.path.basename(fn)} "
          f"train={len(out['train'])} (synth {ok} + real {len(real['train'])})")

if __name__ == "__main__":
    main()
