"""
Recover per-epoch training logs from printed output, so the authors' scripts can be
compared with ours without modifying them.

Capture the output when training:
    python Heston_train_adv.py --N 10000 --delta 1.0 --alpha 10 --attack_method SV \
        2>&1 | tee ../Result/stdout_SVatt_N1e4_delta1.0.txt

Then:
    python parse_logs.py ../Result/stdout_*.txt
writes one ../Result/<run>_part<K>_log.csv per model (same format as Heston_train_pen.py,
see runlog.py). Handles the output of Heston_train_adv.py, Heston_train_clean.py and
Heston_train_pen.py (the last already writes its own CSVs; parsing it is harmless).

Conventions for the authors' scripts (from their code):
  * adversarial script: warm-up = first 300 epochs; the adversary's radius is not
    printed, so it is set to the NOMINAL delta in the robust phase (the budget attack
    projects the per-path budgets to radius delta; clipping of the sign profile can make
    the realised radius slightly smaller);
  * learning rate reconstructed from each script's schedule (adv/pen: 0.005 with x0.1 at
    epochs 200/500/600; clean: 0.05 with x0.1 at 200/400/600).
"""
import argparse
import os
import re

from naming import parse_name
from runlog import write_log

START = re.compile(r"Start Running (?P<name>\S+)_part(?P<part>\d+)")
ADV = re.compile(r"^epoch(?P<epoch>\d+),clean_loss: (?P<clean>[-0-9.eE+naif]+), "
                 r"att_loss: (?P<att>[-0-9.eE+naif]+),\s*(?:radius: (?P<radius>[-0-9.eE+naif]+), )?"
                 r"time: (?P<sec>[-0-9.eE+]+)s, p0_clean: (?P<p0c>[-0-9.eE+naif]+), "
                 r"p0_att: (?P<p0a>[-0-9.eE+naif]+)")
CLN = re.compile(r"^epoch (?P<epoch>\d+), train loss: (?P<clean>[-0-9.eE+naif]+), "
                 r"time: (?P<sec>[-0-9.eE+]+)")


def _lr(method, epoch):
    if method == "clean_authors":
        base, steps = 0.05, (200, 400, 600)
    else:
        base, steps = 0.005, (200, 500, 600)
    return base * 0.1 ** sum(epoch >= s for s in steps)


def parse_stdout(text, warmup=300):
    """Returns {(run_name, part_1based): [row, ...]}."""
    runs, key, meta = {}, None, None
    for line in text.splitlines():
        line = line.strip()
        m = START.search(line)
        if m:
            key = (m["name"], int(m["part"]) + 1)        # printed 0-based, saved 1-based
            meta = parse_name(m["name"])
            runs[key] = []
            continue
        if key is None:
            continue
        m = ADV.match(line)
        if m:
            e = int(m["epoch"])
            robust = e >= warmup
            if m["radius"] is not None:
                radius = m["radius"] if robust else ""
            elif robust and meta and meta["delta"] is not None:
                radius = meta["delta"]                    # nominal radius of constrained attack
            else:
                radius = ""
            runs[key].append(dict(epoch=e, phase="robust" if robust else "clean",
                                  clean_loss=m["clean"], att_loss=m["att"], radius=radius,
                                  seconds=m["sec"], p0_clean=m["p0c"], p0_att=m["p0a"],
                                  lr=_lr(meta["method"] if meta else "", e)))
            continue
        m = CLN.match(line)
        if m:
            e = int(m["epoch"])
            runs[key].append(dict(epoch=e, phase="clean", clean_loss=m["clean"], att_loss=0.0,
                                  radius="", seconds=m["sec"], p0_clean="", p0_att="",
                                  lr=_lr("clean_authors", e)))
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="captured stdout files")
    ap.add_argument("--outdir", default="../Result")
    ap.add_argument("--warmup", type=int, default=300, help="authors' warm-up length")
    ap.add_argument("--overwrite", action="store_true",
                    help="overwrite existing *_log.csv (default: keep CSVs written live)")
    args = ap.parse_args()
    for f in args.files:
        with open(f) as fh:
            runs = parse_stdout(fh.read(), warmup=args.warmup)
        for (name, part), rows in runs.items():
            out = os.path.join(args.outdir, f"{name}_part{part}_log.csv")
            if os.path.exists(out) and not args.overwrite:
                print(f"skip (exists): {out}")
                continue
            write_log(out, rows)
            print(f"{f}: {name} part{part}: {len(rows)} epochs -> {out}")


if __name__ == "__main__":
    main()
