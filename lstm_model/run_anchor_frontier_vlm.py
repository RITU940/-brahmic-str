#!/usr/bin/env python3
"""
Frontier-VLM reference anchor, scored on the identical crops as every other
system in the paper.

WHY THIS EXISTS. The preregistered VLM baseline (Amendment 4c) is one open 7B
model under one fixed prompt. That is a fair reading of what a small open VLM
does, but it is a weak reading of what "a VLM" can do, and a reviewer is right
to say so. This script measures a frontier model on the same crops, under the
same normalization and the same metric.

PROSPECTIVITY. This is a POST-HOC addition, filed after the campaign's results
existed. It is a reference measurement, not a hypothesis test -- nothing is
predicted, nothing is scored against a band. It must be reported as filed after
the fact, exactly like the disclosed most-favourable Qwen re-score. Do not let
it into the prospective scorecard.

SCORING. Two columns, mirroring the Qwen treatment so the two are comparable:
  raw       -- exact match on the raw generation, shared normalization
  extracted -- the same disclosed most-favourable extraction (first quoted span)
Both go through metrics.evaluate_corpus, the same function every other number in
the paper uses.

RESUMABLE. Predictions stream to conf_frontier_<script>.jsonl as they arrive;
re-running skips crops already present. Safe to interrupt.

  export ANTHROPIC_API_KEY=...
  python run_anchor_frontier_vlm.py --scripts tamil            # one script first
  python run_anchor_frontier_vlm.py                            # all nine
  python run_anchor_frontier_vlm.py --limit 50 --dry-run       # cost check only
"""
import argparse, base64, json, os, re, sys, threading
from concurrent.futures import ThreadPoolExecutor

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from metrics import evaluate_corpus
from rescore_vlm_extracted import extract  # identical extraction to the Qwen re-score

SCRIPTS = ['tamil', 'telugu', 'kannada', 'malayalam', 'oriya',
           'gujarati', 'bengali', 'devanagari', 'gurmukhi']

MODEL = "claude-opus-5"

# Fixed before any crop was sent. Deliberately close to the Amendment-4c prompt:
# asking for the transcription and nothing else. Do not tune this per script --
# a per-script prompt would make the anchor incomparable across the nine.
PROMPT = ("Transcribe the text in this image exactly as written, in its own "
          "script. Reply with the transcription only -- no translation, no "
          "transliteration, no explanation, no quotation marks.")

_lock = threading.Lock()


def b64(path):
    with open(path, 'rb') as f:
        return base64.standard_b64encode(f.read()).decode()


def media_type(path):
    ext = os.path.splitext(path)[1].lower()
    return {'.png': 'image/png', '.gif': 'image/gif',
            '.webp': 'image/webp'}.get(ext, 'image/jpeg')


def transcribe(client, rec):
    """One crop -> raw generation. Returns None on a refusal."""
    resp = client.messages.create(
        model=MODEL,
        max_tokens=128,
        system=[{"type": "text", "text": PROMPT,
                 "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64",
                                         "media_type": media_type(rec['image']),
                                         "data": b64(rec['image'])}},
        ]}],
    )
    if resp.stop_reason == "refusal":
        return None
    return "".join(b.text for b in resp.content if b.type == "text").strip()


def run_script(client, script, limit, workers, out_path):
    split = json.load(open(os.path.join(BASE,
        f'splits_zeroshot_loso_rungB_{script}.json'), encoding='utf-8'))
    recs = split['test'][:limit] if limit else split['test']

    done = {}
    if os.path.exists(out_path):
        with open(out_path, encoding='utf-8') as f:
            for line in f:
                r = json.loads(line)
                done[r['name']] = r
    todo = [r for r in recs if r['name'] not in done]
    print(f"{script}: {len(recs)} crops, {len(done)} already done, {len(todo)} to run")

    if todo:
        fh = open(out_path, 'a', encoding='utf-8')

        def work(rec):
            try:
                pred = transcribe(client, rec)
            except Exception as e:                     # keep going; retry on rerun
                print(f"  ! {rec['name']}: {type(e).__name__}: {e}")
                return
            row = {'name': rec['name'], 'orig_gt': rec['orig_gt'],
                   'pred': pred if pred is not None else '',
                   'refused': pred is None}
            with _lock:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
                done[rec['name']] = row

        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(work, todo))
        fh.close()

    rows = [done[r['name']] for r in recs if r['name'] in done]
    if len(rows) < len(recs):
        print(f"  {len(recs) - len(rows)} crops still missing -- rerun to finish")
        return None

    gts = [r['orig_gt'] for r in rows]
    raw = evaluate_corpus(gts, [r['pred'] for r in rows])
    ext = evaluate_corpus(gts, [extract(r['pred']) for r in rows])
    refused = sum(r['refused'] for r in rows)

    out = {'tool': f'frontier-vlm ({MODEL})', 'script': script, 'N': len(rows),
           'WRR': ext['WRR'], 'CharAcc': ext['CharAcc'], 'CER': ext['CER'],
           'WRR_raw': raw['WRR'], 'CharAcc_raw': raw['CharAcc'],
           'refusals': refused, 'prompt': PROMPT, 'post_hoc': True}
    with open(os.path.join(BASE, f'result_anchor_frontier_{script}.json'), 'w') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"  {script}: WRR raw {raw['WRR']:.2f} | extracted {ext['WRR']:.2f} "
          f"| CharAcc {ext['CharAcc']:.2f} | refusals {refused}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scripts', nargs='*', default=SCRIPTS)
    ap.add_argument('--limit', type=int, default=0, help='crops per script (0 = all)')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--dry-run', action='store_true', help='count and cost only')
    args = ap.parse_args()

    if args.dry_run:
        total = 0
        for s in args.scripts:
            sp = json.load(open(os.path.join(BASE,
                f'splits_zeroshot_loso_rungB_{s}.json'), encoding='utf-8'))
            n = len(sp['test'][:args.limit] if args.limit else sp['test'])
            total += n
            print(f"{s:<11} {n:>6}")
        # crops average ~269x78 px => ~30 image tokens; prompt is cached after
        # the first call, output is a single short word
        cost = total * (90 * 5 + 15 * 25) / 1e6
        print(f"{'TOTAL':<11} {total:>6} crops, order ${cost:.2f} at {MODEL} rates")
        return

    import anthropic
    client = anthropic.Anthropic()

    results = []
    for s in args.scripts:
        r = run_script(client, s, args.limit, args.workers,
                       os.path.join(BASE, f'conf_frontier_{s}.jsonl'))
        if r:
            results.append(r)

    if len(results) == len(SCRIPTS):
        macro = sum(r['WRR'] for r in results) / len(results)
        print(f"\nmacro-average extracted WRR over nine: {macro:.2f}")


if __name__ == '__main__':
    main()
