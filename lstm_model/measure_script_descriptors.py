"""
measure_script_descriptors.py  --  Part ① of the "tokenization law" paper.
==========================================================================
Computes, per language/script, the PRE-REGISTERED descriptors that we will use
to PREDICT (a) which tokenization (BPE vs grapheme) wins and (b) the fusion gain.

NO GPU NEEDED. Pure text statistics over the BSTD train labels.

Descriptors per language:
  bytes_per_cluster   : mean UTF-8 bytes per grapheme cluster  (tokenizer-FREE,
                        fully neutral fertility proxy -- reviewer-proof)
  bpe_fertility       : mean BPE sub-tokens per grapheme cluster, measured with
                        the FIXED standard Florence-2 tokenizer (the real BPE
                        branch used in all experiments; identical 71,254-token
                        vocab across every language => unbiased reference)
  bpe_tokens_per_word : mean BPE sub-tokens per word
  grapheme_entropy    : Shannon entropy (bits) of the grapheme-cluster distribution
  conjunct_density    : fraction of clusters containing a virama (true conjuncts)
  chars_per_cluster   : mean Unicode codepoints per cluster (script complexity)
  clusters_per_word   : mean grapheme clusters per word
  n_words / n_unique_clusters : corpus size signals
  bpe_cluster_fragmentation : fraction of grapheme-cluster OCCURRENCES that the
                        NEUTRAL GPT-2 BPE splits into >1 token (PRE-REG ★ #6 -- a
                        cleaner per-cluster cousin of fertility)
  strr                : Single-Token Retention Rate = fraction of whole words the
                        NEUTRAL GPT-2 BPE encodes as a SINGLE token (PRE-REG ★ #7 --
                        the metric arXiv 2510.09947 argues beats fertility; the
                        horse-race competitor, added BEFORE reading law results)

Output: script_descriptors.json  (+ a printed table)

Usage:
  PY=/home/ujjwal/miniconda3/envs/ritu_scenetext/bin/python
  $PY measure_script_descriptors.py
"""
import os, json, math
from collections import Counter

# reuse the EXACT segmenter used to build the training grapheme vocabs
from build_grapheme_vocab_lang import segment_graphemes_indic, VIRAMAS

BASE = os.path.dirname(os.path.abspath(__file__))
TRAIN_JSON = os.path.join(BASE, 'benchmarks/bstd/Recognition/train_recognition_data.json')
STD_TOKENIZER = os.path.join(BASE, 'checkpoints_florence2_standard/best_model')
OUT = os.path.join(BASE, 'script_descriptors.json')

# language -> writing system (for grouping by script in the paper)
LANG2SCRIPT = {
    'bengali': 'Bengali', 'assamese': 'Bengali',
    'hindi': 'Devanagari', 'marathi': 'Devanagari',
    'gujarati': 'Gujarati', 'punjabi': 'Gurmukhi',
    'kannada': 'Kannada', 'malayalam': 'Malayalam',
    'odia': 'Odia', 'tamil': 'Tamil', 'telugu': 'Telugu',
    'english': 'Latin',  # non-Brahmic control
}


def load_tokenizers():
    """Returns (experiment_tk, neutral_tk).
    experiment_tk = the 71,254-token tokenizer actually used by the BPE branch
                    (NOTE: extended with Bengali tokens -> biased across scripts;
                    kept only because it matches the real experiments).
    neutral_tk    = GPT-2 byte-level BPE = the canonical English-centric reference
                    Florence-2/BART inherit; the FAIR cross-script fertility measure.
    """
    exp = neutral = None
    try:
        from transformers import AutoTokenizer
        exp = AutoTokenizer.from_pretrained(STD_TOKENIZER, trust_remote_code=True)
        print(f"[tokenizer] experiment BPE tokenizer size {len(exp)} (Bengali-extended; biased)")
    except Exception as e:
        print(f"[tokenizer] WARN experiment tokenizer failed: {str(e)[:80]}")
    try:
        from transformers import GPT2TokenizerFast
        neutral = GPT2TokenizerFast.from_pretrained('gpt2')
        print(f"[tokenizer] neutral GPT-2 byte-level BPE size {len(neutral)} (fair reference)")
    except Exception as e:
        print(f"[tokenizer] WARN neutral tokenizer failed: {str(e)[:80]}")
    return exp, neutral


def entropy_bits(counter):
    total = sum(counter.values())
    if total == 0:
        return 0.0
    return -sum((c / total) * math.log2(c / total) for c in counter.values() if c > 0)


def main():
    data = json.load(open(TRAIN_JSON, encoding='utf-8'))
    # group raw texts by language
    by_lang = {}
    for v in data.values():
        lang = v.get('language', '?')
        txt = (v.get('text') or '').strip()
        if txt:
            by_lang.setdefault(lang, []).append(txt)

    exp_tk, neu_tk = load_tokenizers()
    rows = {}
    for lang in sorted(by_lang):
        texts = by_lang[lang]
        cl_counter = Counter()
        n_clusters = n_chars = n_bytes = n_conj = 0
        exp_tokens = neu_tokens = 0
        n_single_tok_words = 0                         # STRR numerator
        for t in texts:
            clusters = segment_graphemes_indic(t)
            cl_counter.update(clusters)
            n_clusters += len(clusters)
            n_chars += len(t)                         # Unicode codepoints
            n_bytes += len(t.encode('utf-8'))
            n_conj += sum(1 for c in clusters if any(v in c for v in VIRAMAS))
            if exp_tk is not None:
                exp_tokens += len(exp_tk.tokenize(t))
            if neu_tk is not None:
                nt = neu_tk.tokenize(t)
                neu_tokens += len(nt)
                if len(nt) == 1:                      # whole word -> 1 neutral BPE token
                    n_single_tok_words += 1

        # PRE-REG ★ #6: per-cluster fragmentation under the NEUTRAL tokenizer.
        # Computed once per UNIQUE cluster, weighted by occurrence count (matches
        # the occurrence-weighted denominator used for fertility).
        frag_cluster_occ = None
        if neu_tk is not None and n_clusters:
            frag_cluster_occ = 0
            for cl, cnt in cl_counter.items():
                if len(neu_tk.tokenize(cl)) > 1:
                    frag_cluster_occ += cnt

        nw = len(texts)
        rows[lang] = {
            'script': LANG2SCRIPT.get(lang, '?'),
            'n_words': nw,
            'n_unique_clusters': len(cl_counter),
            'clusters_per_word': round(n_clusters / nw, 4),
            'chars_per_cluster': round(n_chars / n_clusters, 4) if n_clusters else 0,
            'bytes_per_cluster': round(n_bytes / n_clusters, 4) if n_clusters else 0,
            'grapheme_entropy': round(entropy_bits(cl_counter), 4),
            'conjunct_density': round(n_conj / n_clusters, 4) if n_clusters else 0,
            # PRIMARY fair descriptor: neutral GPT-2 byte-level BPE fertility
            'bpe_fertility_neutral': round(neu_tokens / n_clusters, 4) if (neu_tk and n_clusters) else None,
            # secondary: the (biased) in-experiment tokenizer, for reference only
            'bpe_fertility_experiment': round(exp_tokens / n_clusters, 4) if (exp_tk and n_clusters) else None,
            # PRE-REG ★ #6 / #7: horse-race competitors to fertility (neutral tokenizer)
            'bpe_cluster_fragmentation': round(frag_cluster_occ / n_clusters, 4) if frag_cluster_occ is not None else None,
            'strr': round(n_single_tok_words / nw, 4) if (neu_tk and nw) else None,
        }

    json.dump(rows, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)

    # pretty table
    cols = ['script', 'n_words', 'bytes_per_cluster', 'bpe_fertility_neutral',
            'bpe_cluster_fragmentation', 'strr', 'grapheme_entropy', 'conjunct_density']
    print('\n' + '=' * 120)
    print(f"{'lang':<11}" + ''.join(f"{c:>18}" for c in cols))
    print('-' * 120)
    for lang in sorted(rows, key=lambda L: (rows[L]['bpe_fertility_neutral'] or 0), reverse=True):
        r = rows[lang]
        print(f"{lang:<11}" + ''.join(
            f"{(r[c] if r[c] is not None else 'NA'):>18}" for c in cols))
    print('=' * 110)
    print(f"\nsaved {OUT}")
    print("HYPOTHESIS: higher bpe_fertility / bytes_per_cluster => grapheme branch")
    print("wins by more, and fusion gain is larger. (Verified once branches are trained.)")


if __name__ == '__main__':
    main()
