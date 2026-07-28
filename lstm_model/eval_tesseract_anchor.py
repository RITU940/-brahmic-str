"""
eval_tesseract_anchor.py -- OFF-THE-SHELF OCR FLOOR for the WACV paper.
============================================================================
Stock Tesseract 5 (LSTM engine, --oem 1) on the SAME LOSO test crops our model
is evaluated on, scored under OUR exact-match metric in the shared pivot space.
Answers the reviewer question "why not just use off-the-shelf OCR?".

Predictions are produced in the native script, then mapped to pivot (to_pivot for
the nine Brahmic scripts; khmer_to_pivot for Khmer) and scored against the pivot GT
with metrics.evaluate_corpus -- byte-identical to how our own recognizer is scored.
(Exact-match WRR is invariant to the lossless mapping; we map so CharAcc/CER line up too.)

Inference only, CPU (no GPU). Run in ritu_scenetext (shells out to the loca_accnt
tesseract binary by absolute path):
  /c/ujjwalb/.conda/envs/ritu_scenetext/bin/python eval_tesseract_anchor.py --scripts all
Writes conf_tesseract_<script>.json (per-crop native+pivot preds) and
result_anchor_tesseract_<script>.json ({tool,script,N,WRR,CharAcc,CER}).
"""
import os, sys, json, argparse, subprocess
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from metrics import evaluate_corpus
from build_shared_grapheme_space import to_pivot
from khmer_pivot_map import khmer_to_pivot

TESS = "/c/ujjwalb/.conda/envs/loca_accnt/bin/tesseract"
TESSDATA = "/c/ujjwalb/.conda/pkgs/tesseract-5.2.0-h6a678d5_2/share/tessdata"
TLIB = "/c/ujjwalb/.conda/envs/loca_accnt/lib"
# script -> stock Tesseract traineddata code (Devanagari uses hin; Gurmukhi uses pan)
LANG = {"tamil": "tam", "telugu": "tel", "kannada": "kan", "malayalam": "mal",
        "oriya": "ori", "gujarati": "guj", "bengali": "ben", "devanagari": "hin",
        "gurmukhi": "pan", "khmer": "khm"}
ALL = list(LANG)
PSM = 8   # single word — matches the word-level crops


def run_tess(img, code):
    env = dict(os.environ, TESSDATA_PREFIX=TESSDATA, LD_LIBRARY_PATH=TLIB)
    try:
        r = subprocess.run([TESS, img, "stdout", "-l", code, "--psm", str(PSM), "--oem", "1"],
                           capture_output=True, text=True, env=env, timeout=30)
        return " ".join(r.stdout.split())
    except Exception:
        return ""


def to_pivot_native(text, script):
    return khmer_to_pivot(text)[0] if script == "khmer" else to_pivot(text)


def eval_script(s):
    code = LANG[s]
    test = json.load(open(os.path.join(BASE, f"splits_zeroshot_loso_rungB_{s}.json")))["test"]
    conf = []
    for i, r in enumerate(test):
        img = r["image"] if os.path.isabs(r["image"]) else os.path.join(BASE, r["image"])
        native = run_tess(img, code) if os.path.exists(img) else ""
        conf.append({"name": r.get("name", ""), "gt": r["gt"], "orig_gt": r["orig_gt"],
                     "pred_native": native, "pred": to_pivot_native(native, s)})
        if (i + 1) % 200 == 0:
            print(f"  tesseract {s}: {i+1}/{len(test)}", flush=True)
    json.dump(conf, open(os.path.join(BASE, f"conf_tesseract_{s}.json"), "w"), ensure_ascii=False)
    m = evaluate_corpus([c["gt"] for c in conf], [c["pred"] for c in conf])
    rec = {"tool": "tesseract5_lstm", "psm": PSM, "script": s, "N": len(conf),
           "WRR": round(m["WRR"], 2), "CharAcc": round(m["char_accuracy"], 2),
           "CER": round(m["CER"], 2)}
    json.dump(rec, open(os.path.join(BASE, f"result_anchor_tesseract_{s}.json"), "w"))
    print(f"[TESSERACT {s}] N={rec['N']} WRR={rec['WRR']} CharAcc={rec['CharAcc']} CER={rec['CER']}", flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scripts", default="all", help="'all', 'benchmark' (9, no khmer), or comma list")
    a = ap.parse_args()
    if a.scripts == "all":
        scripts = ALL
    elif a.scripts == "benchmark":
        scripts = [s for s in ALL if s != "khmer"]
    else:
        scripts = a.scripts.split(",")
    recs = [eval_script(s) for s in scripts]
    print("\n===== TESSERACT OFF-THE-SHELF FLOOR (pivot-space WRR) =====")
    for r in recs:
        print(f"  {r['script']:<11} WRR={r['WRR']:>5}  CharAcc={r['CharAcc']:>5}  N={r['N']}")


if __name__ == "__main__":
    main()
