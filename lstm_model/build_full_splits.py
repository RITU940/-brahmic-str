"""
Build UNCAPPED (full-data) Florence-2 splits per language from the BSTD
recognition labels — for the *best-system* leaderboard track (NOT the controlled
b1800 law run; that one stays equal-budget on purpose).

Output: florence2_splits_bstd_<lang>_full.json  with {train,val,test,metadata}.
  train = all official-train crops for the language minus a held-out val
  val   = VAL_N held-out from official train (seed 42)
  test  = the FULL official test set for the language (same as b1800 test)

Image paths are absolute on server3 (/c/ujjwalb/...), matching train_florence2.py.
"""
import os, json, random, argparse

BASE = os.path.dirname(os.path.abspath(__file__))
REC  = os.path.join(BASE, "benchmarks", "bstd", "Recognition")
IMG_ROOT = os.path.join(BASE, "benchmarks", "bstd")   # paths in json are "Recognition/..."
VAL_N = 200
SEED = 42

def load(split):
    return json.load(open(os.path.join(REC, f"{split}_recognition_data.json")))

def entries_for(lang, raw):
    out = []
    for fn, v in raw.items():
        if v.get("language") != lang:
            continue
        txt = (v.get("text") or "").strip()
        if not txt or txt == "###":
            continue
        img = os.path.join(IMG_ROOT, v["path"])
        if not os.path.exists(img):
            continue
        out.append({"image": img, "gt": txt, "name": os.path.splitext(fn)[0]})
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True)
    args = ap.parse_args()
    random.seed(SEED)

    tr = entries_for(args.lang, load("train"))
    te = entries_for(args.lang, load("test"))
    random.shuffle(tr)
    val = tr[:VAL_N]
    train = tr[VAL_N:]

    out = {
        "train": train, "val": val, "test": te,
        "metadata": {"lang": args.lang, "source": "bstd_full_uncapped",
                     "seed": SEED, "n_train": len(train),
                     "n_val": len(val), "n_test": len(te)},
    }
    fn = os.path.join(BASE, f"florence2_splits_bstd_{args.lang}_full.json")
    json.dump(out, open(fn, "w"), ensure_ascii=False)
    print(f"[{args.lang}] full split -> {os.path.basename(fn)} "
          f"train={len(train)} val={len(val)} test={len(te)}")

if __name__ == "__main__":
    main()
