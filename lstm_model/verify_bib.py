"""
Bibliography existence audit for the WACV submission.

WACV 2027's author policy makes a paper "which cites non-existent material"
liable to rejection, potentially without review, and all text is run through a
plagiarism checker. Every entry in ``paper_wacv/main.bib`` is therefore checked
against the arXiv API: the arXiv id must resolve, and the fetched title must
match the title in the .bib (normalised, token-overlap based).

Entries without an arXiv id are listed as MANUAL — they must be checked by hand
against the venue (they cannot be machine-resolved here).

USAGE:
    python verify_bib.py            # audit
    python verify_bib.py --quiet    # only failures + summary
Exit code 0 = every resolvable entry verified, non-zero otherwise.
"""
import argparse
import json
import re
import sys
import time
import urllib.parse
import urllib.request

BIB = "paper_wacv/main.bib"
API = "http://export.arxiv.org/api/query?id_list="
STOP = {"a", "an", "the", "of", "for", "and", "in", "on", "with", "to", "is", "are"}


def parse_bib(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    entries = []
    for m in re.finditer(r"@\w+\{([^,]+),(.*?)\n\}", raw, re.S):
        key, body = m.group(1).strip(), m.group(2)

        def field(name):
            fm = re.search(name + r"\s*=\s*\{(.*?)\}\s*,?\s*\n", body, re.S)
            return " ".join(fm.group(1).split()) if fm else ""

        blob = body
        aid = re.search(r"arXiv[:\s]*(\d{4}\.\d{4,5})", blob)
        entries.append({
            "key": key,
            "title": field("title"),
            "author": field("author"),
            "year": field("year"),
            "arxiv": aid.group(1) if aid else None,
        })
    return entries


def norm(s):
    toks = re.findall(r"[a-z0-9]+", s.lower())
    return [t for t in toks if t not in STOP]


def overlap(a, b):
    sa, sb = set(norm(a)), set(norm(b))
    return len(sa & sb) / max(len(sa | sb), 1)


def fetch(arxiv_id, tries=5):
    """Resolve an arXiv id. Retries on 429/5xx so rate-limiting is never
    reported as a non-existent paper."""
    url = API + urllib.parse.quote(arxiv_id) + "&max_results=1"
    delay = 5
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                xml = r.read().decode("utf-8", "replace")
            break
        except urllib.error.HTTPError as exc:
            if exc.code in (429, 500, 502, 503) and attempt < tries - 1:
                time.sleep(delay)
                delay *= 2
                continue
            raise
    if "<entry>" not in xml:
        return None
    t = re.search(r"<entry>.*?<title>(.*?)</title>", xml, re.S)
    p = re.search(r"<entry>.*?<published>(\d{4})", xml, re.S)
    if not t:
        return None
    return {"title": " ".join(t.group(1).split()), "year": p.group(1) if p else ""}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    entries = parse_bib(BIB)
    ok = fail = manual = 0
    problems = []
    for e in entries:
        if not e["arxiv"]:
            manual += 1
            if not a.quiet:
                print(f"MANUAL {e['key']:<18} {e['title'][:60]}")
            continue
        try:
            got = fetch(e["arxiv"])
        except Exception as exc:                                  # network, 5xx
            got = None
            problems.append((e["key"], f"fetch failed: {exc}"))
        if got is None:
            fail += 1
            print(f"FAIL   {e['key']:<18} arXiv {e['arxiv']} does not resolve")
        else:
            sim = overlap(e["title"], got["title"])
            if sim < 0.6:
                fail += 1
                print(f"FAIL   {e['key']:<18} title mismatch (overlap {sim:.2f})")
                print(f"         bib: {e['title']}")
                print(f'         arXiv {e["arxiv"]}: {got["title"]}')
            else:
                ok += 1
                if not a.quiet:
                    print(f"ok     {e['key']:<18} arXiv {e['arxiv']} "
                          f"(title overlap {sim:.2f})")
        time.sleep(3.0)                                            # arXiv API courtesy

    print(f"\n{len(entries)} entries: {ok} verified, {fail} failed, "
          f"{manual} need manual venue check.")
    for k, msg in problems:
        print(f"  note: {k}: {msg}")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
