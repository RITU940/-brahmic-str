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
  /c/ujjwalb/.conda/envs/ritu_scenetext/bin/python eval_parseq_anchor.py --scripts all --device cuda:0
Writes conf_parseq_<script>.json and result_anchor_parseq_<script>.json.
"""
import os, sys, json, argparse
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
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

    from IndicPhotoOCR.ocr import OCR
    ocr = OCR(device=a.device, verbose=False)

    summary = []
    for s in scripts:
        lang = LANG[s]
        test = json.load(open(os.path.join(BASE, f"splits_zeroshot_loso_rungB_{s}.json")))["test"]
        conf = []
        for i, r in enumerate(test):
            img = r["image"] if os.path.isabs(r["image"]) else os.path.join(BASE, r["image"])
            native, cf = "", 0.0
            if os.path.exists(img):
                try:
                    res = ocr.recognise(img, lang, return_confidence=True)
                    native, cf = res if isinstance(res, tuple) else (res, 0.0)
                except Exception as e:
                    print(f"  ERR {s} {i}: {e}", flush=True)
            native = (native or "").strip()
            conf.append({"name": r.get("name", ""), "gt": r["gt"], "orig_gt": r["orig_gt"],
                         "pred_native": native, "pred": to_pivot(native),
                         "conf": round(float(cf or 0.0), 4)})
            if (i + 1) % 200 == 0:
                print(f"  parseq {s}: {i+1}/{len(test)}", flush=True)
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
