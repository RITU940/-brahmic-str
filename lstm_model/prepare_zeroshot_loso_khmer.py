"""
prepare_zeroshot_loso_khmer.py -- Khmer out-of-benchmark LOSO builder (Amendment 4b).
============================================================================
Builds Rung A / Rung B data to read KHMER with ZERO real Khmer training images,
exactly mirroring prepare_zeroshot_loso.py but for the 10th (out-of-block) script.

Decisions (frozen in KHMER_BUILD_DECISIONS.md, recorded before this build):
  * Recognition unit = single-token (0-space) KhmerST regions -> word-level crops
    from the line-level polygons (their own geometry). 75% of regions qualify.
  * Image-level train/test split (seed 42, TEST_IMG_FRAC=0.5): no image gives crops
    to both sides. Synthetic Khmer (Rung B) is rendered ONLY from train-image tokens.
  * Sources = ALL 11 Brahmic source languages (Khmer holds out none of the 9 scripts).
  * Pivot: khmer_pivot_map.khmer_to_pivot (Khmer is out-of-ISCII); sources use to_pivot.
  * tok_cov (prereg input 89.02) is RECOMPUTED over the frozen test set; re-file if material.

CPU only. Run in the raqm env (Khmer coeng shaping REQUIRES libraqm):
  /c/ujjwalb/.conda/envs/ritu_scenetext/bin/python prepare_zeroshot_loso_khmer.py
"""
import os, sys, json, glob, random
from collections import Counter
from PIL import Image, features

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import prepare_zeroshot_loso as base
from build_shared_grapheme_space import to_pivot, segment_graphemes_indic, pivot_graphemes
from khmer_pivot_map import khmer_to_pivot

# Khmer complex shaping (coeng stacks) is silent-wrong without libraqm -> hard gate.
assert features.check("raqm"), "libraqm NOT active in this Python — Khmer will render unshaped. Use ritu_scenetext env."

SEED = 42
TAG = "khmer"
KHM_JSON = os.path.join(BASE, "khmerst_data", "json_v4")
KHM_IMG = os.path.join(BASE, "khmerst_data", "repo", "scene_img_data")
CROP_DIR = os.path.join(BASE, "khmer_test_crops")
SYNTH_DIR = os.path.join(BASE, "synth_zeroshot_loso_khmer")
TEST_IMG_FRAC = 0.5
SYNTH_TARGET = 3240          # matches the benchmark 1x budget (is_2x = 0)
PER_SOURCE_TRAIN = 700       # same as prepare_zeroshot_loso defaults
PER_SOURCE_VAL = 90
PREREG_TOKCOV = 89.02        # PROSPECTIVE_PREDICTION_KHMER.md input; recheck gate


def load_khmer_regions():
    """Every annotated region: {fn, ridx, text, bbox=(x0,y0,x1,y1)}."""
    rows = []
    for jf in sorted(glob.glob(os.path.join(KHM_JSON, "*.json"))):
        d = json.load(open(jf, encoding="utf-8"))
        for _viakey, meta in d.items():
            fn = meta.get("filename")
            if not fn or not os.path.exists(os.path.join(KHM_IMG, fn)):
                continue
            for i, r in enumerate(meta.get("regions", [])):
                ra = r.get("region_attributes", {})
                txt = next((v.strip() for v in ra.values()
                            if isinstance(v, str) and v.strip()), None)
                if not txt:
                    continue
                sa = r.get("shape_attributes", {})
                xs, ys = sa.get("all_points_x"), sa.get("all_points_y")
                if not xs or not ys:
                    continue
                rows.append({"fn": fn, "ridx": i, "text": txt,
                             "bbox": (min(xs), min(ys), max(xs), max(ys))})
    return rows


def crop_region(fn, bbox, out_path):
    """Axis-aligned polygon-bbox crop, clamped to image bounds. False if degenerate."""
    try:
        im = Image.open(os.path.join(KHM_IMG, fn)).convert("RGB")
    except Exception:
        return False
    W, H = im.size
    x0, y0, x1, y1 = bbox
    x0, y0 = max(0, int(x0)), max(0, int(y0))
    x1, y1 = min(W, int(x1)), min(H, int(y1))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return False
    im.crop((x0, y0, x1, y1)).save(out_path, quality=92)
    return True


def build_vocab_khmer(source_records, synth_records, tag, out_path):
    """Source graphemes via pivot_graphemes(orig) (blessed Brahmic path); Khmer synth
    graphemes via segmenting the Devanagari-space pivot gt (Khmer isn't Indic-segmentable)."""
    freq = Counter()
    for r in source_records:
        freq.update(pivot_graphemes(r["orig_gt"]))
    for r in synth_records:
        freq.update(segment_graphemes_indic(r["gt"]))
    graphemes = sorted(g for g in freq if g.strip())
    g2i = {"<blank>": 0, "<unk>": 1}
    for g in graphemes:
        g2i[g] = len(g2i)
    json.dump({"grapheme2idx": g2i,
               "metadata": {"tag": tag, "num_graphemes": len(graphemes),
                            "space": "shared-abugida-pivot"}},
              open(out_path, "w"), ensure_ascii=False, indent=2)
    return set(graphemes)


def main():
    rng = random.Random(SEED)
    os.makedirs(CROP_DIR, exist_ok=True)
    os.makedirs(SYNTH_DIR, exist_ok=True)

    # ---- Stage 1: regions, single-token filter, image-level split -------------
    rows = load_khmer_regions()
    single = [r for r in rows if " " not in r["text"]]
    imgs = sorted({r["fn"] for r in rows})
    rng.shuffle(imgs)
    n_test = int(round(len(imgs) * TEST_IMG_FRAC))
    test_imgs = set(imgs[:n_test])
    train_imgs = set(imgs[n_test:])
    print(f"[1] regions={len(rows)} single-token={len(single)} "
          f"images={len(imgs)} -> test_imgs={len(test_imgs)} train_imgs={len(train_imgs)}")

    # ---- Stage 2: test crops from single-token TEST-image regions -------------
    tgt_test, dropped_empty, degenerate = [], 0, 0
    for r in single:
        if r["fn"] not in test_imgs:
            continue
        piv, nk, nm = khmer_to_pivot(r["text"])
        if not piv.strip():
            dropped_empty += 1
            continue
        name = f"khm_{os.path.splitext(r['fn'])[0]}_{r['ridx']}"
        cp = os.path.join(CROP_DIR, name + ".jpg")
        if not crop_region(r["fn"], r["bbox"], cp):
            degenerate += 1
            continue
        tgt_test.append({"image": cp, "gt": piv, "name": name, "orig_gt": r["text"]})
    assert tgt_test, "no Khmer test crops produced — check paths"
    print(f"[2] test crops={len(tgt_test)} (dropped empty-pivot={dropped_empty}, "
          f"degenerate box={degenerate})")

    # ---- Stage 3: balanced source training data (all 11 Brahmic langs) --------
    src_train, src_val, miss = [], [], 0
    for s in base.BRAHMIC_LANGS:
        tr, m1 = base.pivotize(base.load_records(s, "train")); miss += m1
        va, m2 = base.pivotize(base.load_records(s, "val")); miss += m2
        rng.shuffle(tr); rng.shuffle(va)
        src_train += tr[:PER_SOURCE_TRAIN]
        src_val += va[:PER_SOURCE_VAL]
    rng.shuffle(src_train)
    print(f"[3] sources: train={len(src_train)} val={len(src_val)} (missing imgs skipped={miss})")

    # ---- Stage 4: Rung A (sources only) --------------------------------------
    json.dump({"train": src_train, "val": src_val, "test": tgt_test,
               "metadata": {"rung": "A", "script": "Khmer", "targets": ["khmer"],
                            "sources": base.BRAHMIC_LANGS}},
              open(os.path.join(BASE, f"splits_zeroshot_loso_rungA_{TAG}.json"), "w"),
              ensure_ascii=False)
    build_vocab_khmer(src_train, [], f"zs_loso_rungA_{TAG}",
                      os.path.join(BASE, f"grapheme_vocab_zeroshot_loso_rungA_{TAG}.json"))

    # ---- Stage 5: Khmer synth from TRAIN-image tokens; Rung B ------------------
    train_words = []
    for r in rows:
        if r["fn"] in train_imgs:
            for w in r["text"].split():
                if khmer_to_pivot(w)[0].strip():
                    train_words.append(w)
    uniq = sorted(set(train_words))
    assert uniq, "no renderable Khmer train words"
    fonts = [f for f in [os.path.join("/usr/share/fonts/truetype/ttf-khmeros-core/KhmerOS.ttf"),
                         os.path.join("/usr/share/fonts/truetype/ttf-khmeros-core/KhmerOSsys.ttf")]
             if os.path.exists(f) and base._renders_nonblank(f, uniq[0])]
    assert fonts, "no Khmer font renders on this machine"
    print(f"[5] synth lexicon: {len(uniq)} unique train words; fonts={[os.path.basename(f) for f in fonts]}")

    synth, k, fail = [], 0, 0
    pool = list(uniq)
    while len(synth) < SYNTH_TARGET and fail < SYNTH_TARGET * 3:
        rng.shuffle(pool)
        for w in pool:
            if len(synth) >= SYNTH_TARGET:
                break
            fp = os.path.join(SYNTH_DIR, f"synth_{TAG}_{k:06d}.jpg")
            if base.render_word(w, fonts, fp, rng):
                synth.append({"image": fp, "gt": khmer_to_pivot(w)[0],
                              "name": f"synth{k}", "orig_gt": w})
                k += 1
            else:
                fail += 1
    assert len(synth) >= SYNTH_TARGET, f"only rendered {len(synth)}/{SYNTH_TARGET} synth (fail={fail})"
    print(f"[5] synth rendered={len(synth)} (failed renders={fail})")

    rungB_train = src_train + synth
    rng.shuffle(rungB_train)
    json.dump({"train": rungB_train, "val": src_val, "test": tgt_test,
               "metadata": {"rung": "B", "script": "Khmer", "targets": ["khmer"],
                            "sources": base.BRAHMIC_LANGS, "n_synth": len(synth)}},
              open(os.path.join(BASE, f"splits_zeroshot_loso_rungB_{TAG}.json"), "w"),
              ensure_ascii=False)
    build_vocab_khmer(src_train, synth, f"zs_loso_rungB_{TAG}",
                      os.path.join(BASE, f"grapheme_vocab_zeroshot_loso_rungB_{TAG}.json"))

    # ---- Stage 6: tok_cov recheck over the FROZEN test set --------------------
    src_vocab = set()
    for r in src_train:
        src_vocab.update(pivot_graphemes(r["orig_gt"]))
    tok = Counter()
    for r in tgt_test:
        tok.update(segment_graphemes_indic(r["gt"]))
    tot = sum(tok.values())
    cov_tok = 100 * sum(c for g, c in tok.items() if g in src_vocab) / max(tot, 1)
    # codepoint map-rate over the test set (for the prereg's 94.48% comparison)
    nk = nm = 0
    for r in tgt_test:
        _, a, b = khmer_to_pivot(r["orig_gt"]); nk += a; nm += b
    maprate = 100 * nm / max(nk, 1)

    meta = {"script": "Khmer", "unit": "single-token region crop", "seed": SEED,
            "test_img_frac": TEST_IMG_FRAC, "n_test": len(tgt_test),
            "n_src_train": len(src_train), "n_synth": len(synth),
            "n_train_words_unique": len(uniq),
            "tokcov_test_by_source_vocab_pct": round(cov_tok, 2),
            "prereg_tokcov": PREREG_TOKCOV,
            "tokcov_delta_vs_prereg": round(cov_tok - PREREG_TOKCOV, 2),
            "codepoint_map_rate_test_pct": round(maprate, 2),
            "fonts": [os.path.basename(f) for f in fonts]}
    json.dump(meta, open(os.path.join(BASE, f"zeroshot_loso_meta_{TAG}.json"), "w"),
              ensure_ascii=False, indent=2)
    print(f"[6] TEST tok_cov={cov_tok:.2f}% (prereg 89.02, delta {cov_tok-PREREG_TOKCOV:+.2f}); "
          f"codepoint map-rate={maprate:.2f}% (prereg 94.48)")
    print(f"    N_test={len(tgt_test)}  N_synth={len(synth)}")
    print("DONE. splits_zeroshot_loso_rung{A,B}_khmer.json + vocabs + meta written.")


if __name__ == "__main__":
    main()
