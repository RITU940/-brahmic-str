"""
Evaluate the BSTD authors' own trained PARSeq recognizer (IndicPhotoOCR
toolkit) on (a) OUR Bengali test set and (b) the BSTD Bengali test set.
This is the specialist-SOTA comparison reviewers will demand.

Run inside the `indicphotoocr` conda env:
  python eval_indicphotoocr_parseq.py --which ours
  python eval_indicphotoocr_parseq.py --which bstd
Saves conf_parseq_<which>.json (same format as other conf files) so all the
existing analysis (fusion/significance/figures) can consume it.
NOTE: metrics are computed afterwards in the main env via evaluate_corpus.
"""
import os, sys, json, argparse

BASE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--which', choices=['ours', 'bstd'], required=True)
    ap.add_argument('--lang', default='bengali')
    ap.add_argument('--device', default='cuda:0')
    a = ap.parse_args()

    splits = {'ours': 'florence2_splits.json', 'bstd': 'florence2_splits_bstd.json'}[a.which]
    test = json.load(open(os.path.join(BASE, splits)))['test']

    from IndicPhotoOCR.ocr import OCR
    ocr = OCR(verbose=False, device=a.device)

    out = []
    for i, p in enumerate(test):
        img = p['image']
        if not os.path.isabs(img):
            img = os.path.join(BASE, img)
        try:
            r = ocr.recognise(img, a.lang, return_confidence=True)
            text, conf = (r if isinstance(r, tuple) else (r, 0.0))
        except Exception as e:
            print(f"  ERR {i}: {e}", flush=True)
            text, conf = '', 0.0
        out.append({'gt': p['gt'], 'pred': (text or '').strip(),
                    'conf': round(float(conf or 0.0), 4)})
        if (i + 1) % 100 == 0:
            print(f"  parseq_{a.which}: {i+1}/{len(test)}", flush=True)

    path = os.path.join(BASE, f'conf_parseq_{a.which}.json')
    json.dump(out, open(path, 'w'), ensure_ascii=False, indent=2)
    print(f"saved {path} ({len(out)} samples)")


if __name__ == '__main__':
    main()
