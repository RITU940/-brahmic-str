# PROSPECTIVE PREDICTIONS — Amendment-4a synthetic-exposure scaling sweep
**Filed:** 2026-07-18 (IST). **Status at filing:** 0 of the 12 sweep runs trained; sweep data
generation NOT yet started (this file and PREREGISTRATION Amendment 5 are committed and pushed
first). The LOSO campaign stands at 23/27 (Bbpe gujarati training; bengali/devanagari/gurmukhi
Bbpe pending — none of those affect this file's numbers).

## Instrument (frozen)
The 9-point two-factor checkpoint refit (PREREGISTRATION Amendment 4d; scored in
`PROSPECTIVE_PREDICTION_GURMUKHI.md` lineage):

    WRR(script, 3240) = observed Rung-B result (base anchor)
    WRR(script, budget) = RungB + 11.48 · log2(budget / 3240)      [linear-in-log2 extension]
    clip: max(prediction, RungA baseline)                          [clip disclosed per cell]
    bands: ±1·RMSE = ±4.18, ±2·RMSE = ±8.36 (RMSE of the 9-pt fit, assumed to transport)

The per-doubling coefficient +11.48 comes from the natural 2×-synth experiment
(bengali/devanagari); the declared confirmatory test remains Amendment 4a's: whether the swept,
fitted c is compatible with the declared c ≈ +12.3 (8-pt fit at 4a filing; 11.48 at the 9-pt
checkpoint). The table below is the exploratory per-rung receipt chain.

## Point predictions (WRR %, real test images; clip → Rung-A baseline)

| Script | RungA | RungB (3240) | 810 | 1620 | 6480 | 12960 |
|---|---|---|---|---|---|---|
| malayalam | 0.18 | 9.69 | **0.18** (clipped; raw −13.3) | **0.18** (clipped; raw −1.8) | **21.2** | **32.7** |
| kannada | 2.78 | 15.42 | **2.78** (clipped; raw −7.5) | **3.9** | **26.9** | **38.4** |
| telugu | 1.28 | 19.82 | **1.28** (clipped; raw −3.1) | **8.3** | **31.3** | **42.8** |

## What would falsify what (scoring rules, fixed now)
1. **Low end (810/1620):** the linear-in-log2 form implies near-collapse toward the Rung-A
   baseline. If 810-budget WRR lands clearly ABOVE the clipped prediction + 2·RMSE (e.g.
   malayalam ≥ ~8.5, telugu ≥ ~9.6), the linear form is falsified at the low end — transfer is
   concave in log-budget (a little synthetic exposure goes a long way). Reported either way.
2. **The faithful step (3240→6480):** per-script gains ≈ +11.5 expected. Mean gain across the
   three scripts within [+7.3, +15.7] (±1·RMSE) scores a hit; outside ±2·RMSE scores a miss.
3. **Top end (6480→12960):** per Amendment 5.4 this step adds renders at (near-)fixed lexicon;
   a markedly smaller gain than the faithful step is evidence of saturation/lexicon-exhaustion
   (indistinguishable here, both reported); a gain comparable to +11.5 would indicate pure
   image-quantity driving transfer.
4. Misses are reported exactly like hits, per the standing protocol.

## Provenance
- Base anchors = result_zs_loso_rungA/B_{malayalam,kannada,telugu}.json (committed).
- Instrument coefficients = 9-pt refit logged 2026-07-09 (`ZEROSHOT_LOSO_LIVE_LOG.md` rung log)
  and PREREGISTRATION Amendment 4d.
- Implementation constraints and run order: PREREGISTRATION Amendment 5 (2026-07-18).
- The git commit+push of THIS file, before any sweep data or run exists, is the prospectivity
  proof; cite the hash in the manuscript.
