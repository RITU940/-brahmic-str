"""
eval_parseq_anchor.py -- SUPERVISED SPECIALIST CEILING for the WACV paper.
============================================================================
The IndicPhotoOCR PARSeq recognizers (trained WITH real labeled target images, the
BSTD authors' own specialist models) on the SAME LOSO test crops our zero-real-image
model is evaluated on, scored under OUR exact-match metric in the shared pivot space.
This is the different-axis upper bound: it presupposes the labeled target data whose
absence defines our setting.

Predictions (native script) are mapped to pivot via to_pivot and scored against the
pivot GT with metrics.evaluate_corpus -- byte-identical to how our recognizer is scored.

Inference only, GPU. Nine Brahmic scripts (PARSeq has no Khmer model). Per-language
checkpoints auto-download on first use (GitHub releases). Run in ritu_scenetext:
  PYTHONPATH=/c/ujjwalb/ritu1/IndicPhotoOCR \
  /c/ujjwalb/.conda/envs/ritu_scenetext/bin/python eval_parseq_anchor.py --scripts all --device cuda:0
Writes conf_parseq_<script>.json and result_anchor_parseq_<script>.json.

We drive PARseqrecogniser directly rather than the IndicPhotoOCR.ocr.OCR facade: the
facade imports the detection stack (cv2, matplotlib), which this env does not carry and
which we do not need -- our inputs are already word crops, so only recognition applies.
Batched (batch_size=32) for throughput; identical model and weights either way.
"""
import os, sys, json, argparse
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
sys.path.insert(0, "/c/ujjwalb/ritu1/IndicPhotoOCR")
from metrics import evaluate_corpus
from build_shared_grapheme_space import to_pivot

# script -> IndicPhotoOCR PARSeq language name (Oriya->odia, Devanagari->hindi, Gurmukhi->punjabi)
LANG = {"tamil": "tamil", "telugu": "telugu", "kannada": "kannada", "malayalam": "malayalam",
        "oriya": "odia", "gujarati": "gujarati", "bengali": "bengali",
        "devanagari": "hindi", "gurmukhi": "punjabi"}
ALL = list(LANG)  # no khmer (unsupported by the specialist)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scripts", default="all", help="'all' (9) or comma list")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    scripts = ALL if a.scripts == "all" else a.scripts.split(",")

    from IndicPhotoOCR.recognition.parseq_recogniser import PARseqrecogniser
    rec_engine = PARseqrecogniser()

    summary = []
    for s in scripts:
        lang = LANG[s]
        test = json.load(open(os.path.join(BASE, f"splits_zeroshot_loso_rungB_{s}.json")))["test"]

        # Resolve paths first; missing crops stay in the record as empty predictions so
        # N matches our own runs exactly (never silently dropped from the denominator).
        paths, present = [], []
        for r in test:
            img = r["image"] if os.path.isabs(r["image"]) else os.path.join(BASE, r["image"])
            ok = os.path.exists(img)
            present.append(ok)
            if ok:
                paths.append(img)
        if not paths:
            print(f"  SKIP {s}: no crops found on disk", flush=True)
            continue

        preds = []
        B = 32
        for i in range(0, len(paths), B):
            chunk = paths[i:i + B]
            try:
                preds.extend(rec_engine.recognise_batch(
                    lang, chunk, lang, False, a.device, return_confidence=True, batch_size=B))
            except Exception as e:
                print(f"  ERR {s} batch@{i}: {e}", flush=True)
                preds.extend([("", 0.0)] * len(chunk))
            if (i + B) % 640 == 0:
                print(f"  parseq {s}: {min(i+B, len(paths))}/{len(paths)}", flush=True)

        conf, k = [], 0
        for r, ok in zip(test, present):
            native, cf = "", 0.0
            if ok:
                res = preds[k]; k += 1
                native, cf = res if isinstance(res, tuple) else (res, 0.0)
            native = (native or "").strip()
            conf.append({"name": r.get("name", ""), "gt": r["gt"], "orig_gt": r["orig_gt"],
                         "pred_native": native, "pred": to_pivot(native),
                         "conf": round(float(cf or 0.0), 4)})
        json.dump(conf, open(os.path.join(BASE, f"conf_parseq_{s}.json"), "w"), ensure_ascii=False)
        m = evaluate_corpus([c["gt"] for c in conf], [c["pred"] for c in conf])
        rec = {"tool": "parseq_indicphotoocr", "script": s, "N": len(conf),
               "WRR": round(m["WRR"], 2), "CharAcc": round(m["char_accuracy"], 2),
               "CER": round(m["CER"], 2)}
        json.dump(rec, open(os.path.join(BASE, f"result_anchor_parseq_{s}.json"), "w"))
        print(f"[PARSEQ {s}] N={rec['N']} WRR={rec['WRR']} CharAcc={rec['CharAcc']} CER={rec['CER']}", flush=True)
        summary.append(rec)

    print("\n===== PARSEQ SUPERVISED CEILING (pivot-space WRR) =====")
    for r in summary:
        print(f"  {r['script']:<11} WRR={r['WRR']:>5}  CharAcc={r['CharAcc']:>5}  N={r['N']}")


if __name__ == "__main__":
    main()
