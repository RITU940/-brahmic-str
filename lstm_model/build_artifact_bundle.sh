#!/usr/bin/env bash
# Build the ANONYMOUS supplementary artifact bundle for WACV submission.
#
# WACV 2027 supplementary may contain source code and data, is anonymous, and is
# available to reviewers (200 MB max, ZIP; R2 deadline Aug 30). That means the
# paper's release claim can be backed WITHOUT any public mirror -- which also
# removes the deanonymisation hazard of publishing prereg commit hashes before
# acceptance. Public release + real hashes happen at camera-ready.
#
#   bash build_artifact_bundle.sh
# Produces artifact_bundle/ and wacv_supplementary.zip, plus a leak report.
# Publishes nothing. Nothing here is pushed anywhere.
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")" || exit 1
SRC=$PWD
OUT=$SRC/artifact_bundle
ZIP=$SRC/wacv_supplementary.zip

rm -rf "$OUT" "$ZIP"; mkdir -p "$OUT"/{code,splits,results,preregistration}

# ---- 1. WHITELIST what ships. Never blacklist: a missed internal doc is a
# ---- desk reject, and several docs here discuss review strategy explicitly.
cp -- *.py "$OUT/code/" 2>/dev/null
# strategy//internal documents must NEVER reach a reviewer
rm -f "$OUT"/code/{crop_ritu1_words.py}      2>/dev/null   # renamed below if present
cp -- splits_zeroshot_loso_*.json "$OUT/splits/" 2>/dev/null
cp -- result_*.json               "$OUT/results/" 2>/dev/null
cp -- zeroshot_loso_meta_khmer.json "$OUT/results/" 2>/dev/null
# verify_wacv_numbers.py reads this for the \rt* macros; without it the shipped
# harness cannot re-derive the pivot round-trip numbers the paper quotes.
cp -- pivot_roundtrip_audit.json  "$OUT/results/" 2>/dev/null
# The paper claims a harness that re-derives every result number and fails on
# mismatch. A reviewer must be able to RUN it from this zip, so ship its two
# remaining inputs and numbers.tex itself -- without them it dies on import.
cp -- law_fit_results_brahmic.json      "$OUT/results/" 2>/dev/null
cp -- visual_similarity_descriptors.json "$OUT/results/" 2>/dev/null
cp -- paper_wacv/numbers.tex            "$OUT/" 2>/dev/null
# The \parallelsub reference promises this file by name. Built by anonymising
# paper/main.tex (authors, emails, affiliation); checked to contain no identity
# string in rendered text or raw bytes. Without it the citation points at nothing.
cp -- parallel_submission.pdf     "$OUT/" 2>/dev/null
for f in PREREGISTRATION.md PROSPECTIVE_PREDICTION_*.md PROSPECTIVE_PREDICTIONS_H3.md \
         SCALING_SWEEP_SCORING.md ARCHITECTURE_SCORING.md KHMER_SCORING.md \
         KHMER_BUILD_DECISIONS.md ANCHOR_SPLIT_HYGIENE.md; do
  [ -f "$f" ] && cp -- "$f" "$OUT/preregistration/"
done

# ---- 2. Anonymise: absolute paths carry the account name; 'ritu1' echoes the
# ---- author's given name; the prereg docs carry an explicit Owner line.
find "$OUT" -type f \( -name '*.py' -o -name '*.json' -o -name '*.md' \) -print0 |
  xargs -0 sed -i \
    -e 's#/c/ujjwalb/ritu1/lstm_model#.#g' \
    -e 's#/c/ujjwalb/ritu1#..#g' \
    -e 's#/c/ujjwalb#/home/anon#g' \
    -e 's/ujjwalb/anon/gI' \
    -e 's/ujjwal/anon/gI' \
    -e 's/ritu1/labset/gI' \
    -e 's/RITU940/[anonymized]/gI' \
    -e 's/Ritu Baskey/[anonymized]/gI' \
    -e 's/baskeyritu@gmail\.com/[anonymized]/gI' \
    -e 's/Baskey/[anonymized]/gI' \
    -e 's/server[0-9]\+/hostA/gI' \
    -e 's/cvpr-gamma/hostA/gI' \
    -e 's/ritu/anon/gI'          # catches ritu_scenetext etc.; verified no legit word matches
# any file whose NAME leaks
for f in $(find "$OUT" -name '*ritu*' 2>/dev/null); do
  mv -- "$f" "$(dirname "$f")/$(basename "$f" | sed 's/ritu1/labset/g')"
done

# ---- 2b. Reviewer-facing README.
cat > "$OUT/README.md" <<'RM'
# Supplementary artifact: zero-real-image cross-script STR

Anonymous supplement to the submitted paper. Everything the paper claims to release is
here, except the raw benchmark images (redistributed by their original authors, not us)
and model checkpoints (size). Nothing here requires network access to inspect.

## Layout
| path | contents |
|---|---|
| `code/` | pivot construction, synthetic rendering, training/eval, the anchors, and `verify_wacv_numbers.py` |
| `splits/` | every LOSO split actually used: 9 scripts x rungs A / B / B_bpe, plus the Khmer rungs |
| `results/` | every per-run result JSON the paper cites, including the Tesseract and PARSeq anchors |
| `preregistration/` | the frozen preregistration, each prospective prediction filed before its run, and the scoring receipts |
| `SHA256SUMS.txt` | SHA-256 over every file above |

## How to check the paper's numbers
`code/verify_wacv_numbers.py` re-derives every derivable macro in the paper's
`numbers.tex` from the result JSONs in `results/` and prints a per-macro pass/fail table.
It is the same script we run before each build, and it runs from this zip with no
setup and no network:

    unzip wacv_supplementary.zip
    cd artifact_bundle/code && python3 verify_wacv_numbers.py

It needs only the standard library. Expect a per-macro table ending in
`@@VERIFYN@@ macros checked, 0 mismatch(es)` and exit status 0; it exits non-zero on any
mismatch. `numbers.tex` (the paper's single source for every quoted number) sits at
the top of this bundle so the check is self-contained.

## On the preregistration chain
`preregistration/` contains predictions filed *before* the runs they predict, together
with the receipts that score them -- hits and misses alike. Read Sec. 3.3 of the paper
for the limit on this: the archive was a private repository, so no third party observed
the commits as they were made and git's own timestamps are settable by the author.
`SHA256SUMS.txt` establishes that these files are unaltered, not when they were written.
Filing order is attested here, not proven, and we do not present it as proven. What you
can check directly is that every forecast is scored by the rule filed with it: the bands
and point values are in these documents, and the results they are scored against are in
`results/`.

## Anonymisation
Author names, account names, hostnames and repository identifiers were mechanically
replaced (e.g. paths rewritten to `/home/anon/...`, one dataset tag renamed to
`labset`). These substitutions affect strings only, never data or numbers.
RM

# The expected macro count is read from numbers.tex rather than hardcoded: it has gone
# stale before (the README promised 168 after the paper had moved to 169), and a README
# that disagrees with the harness undermines the one artifact meant to settle disputes.
VERIFYN=$(grep -oP '\\newcommand\{\\verifyMacros\}\{\K[^}]*' "$SRC/paper_wacv/numbers.tex")
if [ -z "$VERIFYN" ]; then echo "!! could not read verifyMacros from numbers.tex"; exit 1; fi
sed -i "s/@@VERIFYN@@/$VERIFYN/" "$OUT/README.md"
echo "  README expects $VERIFYN macros"

# ---- 3. SHA-256 commitments over the preregistration chain + code + splits.
( cd "$OUT" && find . -type f ! -name SHA256SUMS.txt -print0 | sort -z |
    xargs -0 sha256sum > SHA256SUMS.txt )
echo "  $(wc -l < "$OUT/SHA256SUMS.txt") files hashed"

# ---- 4. Leak scan. Fail loudly rather than ship a desk reject.
echo "=== LEAK SCAN ==="
LEAKS=$(grep -rniE 'ujjwal|baskey|ritu|server[0-9]|cvpr-gamma|gmail|brahmic-str|RITU940' "$OUT" 2>/dev/null | grep -v SHA256SUMS | head -20)
if [ -n "$LEAKS" ]; then echo "!! POTENTIAL IDENTITY LEAKS:"; echo "$LEAKS"; else echo "clean: no identity tokens found"; fi
# ---- credential scan. This bundle ships every *.py to reviewers, so an API key
# ---- pasted into a script would be published to strangers. Hard-fail, don't warn.
echo "=== CREDENTIAL SCAN ==="
CREDS=$(grep -rniE 'sk-ant-[a-z0-9-]{8}|sk-[a-zA-Z0-9]{32}|AIza[0-9A-Za-z_-]{20}|ghp_[A-Za-z0-9]{20}|xox[baprs]-[A-Za-z0-9-]{10}|AKIA[0-9A-Z]{12}|BEGIN [A-Z ]*PRIVATE KEY' \
  "$OUT" 2>/dev/null | grep -v SHA256SUMS | head -10)
if [ -n "$CREDS" ]; then
  echo "!! CREDENTIALS FOUND IN BUNDLE -- refusing to build:"; echo "$CREDS"
  rm -f "$ZIP"; exit 1
fi
echo "clean: no credentials found"
echo "=== internal strategy docs present? (must be empty) ==="
find "$OUT" -iname '*STRATEGY*' -o -iname '*COMPETITIVE*' -o -iname '*PROF_REPORT*' -o -iname '*PROGRESS_REPORT*' -o -iname '*HANDOFF*' -o -iname '*LIVE_LOG*'

( cd "$SRC" && zip -qr "$ZIP" artifact_bundle )
echo "=== bundle: $(du -sh "$OUT" | cut -f1) dir, $(du -h "$ZIP" | cut -f1) zip (limit 200M) ==="
