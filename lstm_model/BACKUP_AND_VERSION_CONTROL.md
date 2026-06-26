# BACKUP & VERSION CONTROL — Brahmic-STR project
**Set up:** 2026-06-26. **Problem it solves:** the entire 113 GB project lived only on
server3 (`cvpr-gamma`) with **no commits and no off-server copy** — one disk failure = total loss.

## What lives where (the rule)
| Artifact | Size | Home | Status |
|---|---|---|---|
| Code, scripts, docs, manuscript, result/config JSONs | ~110 MB | **git → GitHub (private)** | ✅ committed locally (`bf4079f`); push = TODO |
| Trained model adapters (`*/best_model/`) | 454 MB each | **Hugging Face Hub** or rsync | TODO |
| Datasets, annotations, synth images, fonts | ~GBs | **rsync** to 2nd machine / drive | TODO (`backup_research.sh`) |
| Intermediate `epoch_*` checkpoints | ~2 GB each | regenerable — don't back up | skip |

## 1. Code → GitHub (the irreplaceable 110 MB)
Already committed locally with a clean `.gitignore`. To get it off-server:
```bash
cd /c/ujjwalb/ritu1
gh auth login                                   # one-time (or use a token)
gh repo create brahmic-str --private --source=. --remote=origin
git push -u origin main
```
Keep it **private until arXiv/paper is out** (avoid scooping); flip to **public at publication**
— the code+benchmark release is itself a citable contribution.
> Double-blind submissions (WACV/CVPR/ICDAR): at submission time, share an **anonymized** repo
> (strip name/email, use an anon GitHub or anonymous.4open.science).

## 2. Trained models → Hugging Face Hub (best_model adapters)
GitHub caps files at 100 MB; the 454 MB adapters belong on HF (built for weights, free):
```bash
pip install -U huggingface_hub
huggingface-cli login
# example for one model:
huggingface-cli upload <user>/brahmic-str-zeroshot-tamil \
    lstm_model/checkpoints_zeroshot_loso_rungB_tamil/best_model .
```
The 9-day LOSO run's `best_model` adapters are the expensive-to-regenerate output — back these up.

## 3. Everything big → rsync (`backup_research.sh`)
One-shot off-server backup of models + datasets (incremental; safe to re-run):
```bash
cd /c/ujjwalb/ritu1/lstm_model
bash backup_research.sh user@otherhost:/backup/ritu1     # to another machine
bash backup_research.sh /mnt/external/ritu1              # to a mounted drive
FULL=1 bash backup_research.sh <dest>                    # include epoch_* too
```
Default skips regenerable `epoch_*` checkpoints but keeps every `best_model/`.

## Recommended cadence
- **Code:** `git add -A && git commit && git push` after every meaningful change (daily-ish).
- **Models/data:** re-run `backup_research.sh` after each rung batch / dataset change.
- Consider a weekly `cron` calling `backup_research.sh`.
