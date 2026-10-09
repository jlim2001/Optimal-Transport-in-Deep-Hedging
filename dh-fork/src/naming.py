"""
Parse run names produced by the training scripts into structured fields.

Formats (from the scripts' `name = ...` lines; `part` is 1-based in saved files):
  Heston_train_adv.py   : Heston_SVatt_N1e4_delta1.0_alpha10_tran0e0[_partK]
  Heston_train_pen.py   : Heston_SVpen_N1e4_lam12.5_alpha10_tran0e0[_partK]
  Heston_train_clean.py : HestonClean_N1e4_tran0e0[_partK]

Method labels used everywhere downstream:
  constrained    : authors' adversarial training, delta > 0
  clean          : authors' adversarial script with delta = 0 (matched clean control:
                   same learning rate and schedule as the robust arms)
  clean_authors  : authors' separate clean script (different LR, 0.05; reported for reference)
  penalized      : our penalized training
No torch import, so this module is usable for log/CSV processing anywhere.
"""
import os
import re

_ATT = re.compile(r"^Heston_(?P<attack>S|SV)att_N(?P<N>[0-9.e+-]+)_delta(?P<delta>[0-9.e+-]+)"
                  r"_alpha(?P<alpha>\d+)_tran(?P<tran>[^_]+?)(?:_part(?P<part>\d+))?$")
_PEN = re.compile(r"^Heston_(?P<attack>S|SV)pen_N(?P<N>[0-9.e+-]+)_lam(?P<lam>[0-9.e+-]+)"
                  r"_alpha(?P<alpha>\d+)_tran(?P<tran>[^_]+?)(?:_part(?P<part>\d+))?$")
_CLN = re.compile(r"^HestonClean_N(?P<N>[0-9.e+-]+)_tran(?P<tran>[^_]+?)(?:_part(?P<part>\d+))?$")

FIELDS = ["run", "method", "attack", "N", "delta", "lam", "alpha", "part"]


def _strip(path_or_name):
    base = os.path.basename(str(path_or_name))
    for suffix in (".pth", "_log.csv", ".csv", ".txt"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
    return base


def parse_name(path_or_name):
    """
    Returns a dict with keys FIELDS ('run' = name without the _partK suffix) or None
    if the name matches no known format. Numeric fields are int/float; missing = None.
    """
    base = _strip(path_or_name)
    m = _ATT.match(base)
    if m:
        delta = float(m["delta"])
        out = dict(method="constrained" if delta > 0 else "clean", attack=m["attack"],
                   N=int(float(m["N"])), delta=delta, lam=None, alpha=int(m["alpha"]))
    elif (m := _PEN.match(base)):
        out = dict(method="penalized", attack=m["attack"], N=int(float(m["N"])),
                   delta=None, lam=float(m["lam"]), alpha=int(m["alpha"]))
    elif (m := _CLN.match(base)):
        out = dict(method="clean_authors", attack=None, N=int(float(m["N"])),
                   delta=0.0, lam=None, alpha=None)
    else:
        return None
    out["part"] = int(m["part"]) if m["part"] else None
    out["run"] = re.sub(r"_part\d+$", "", base)
    return {k: out[k] for k in FIELDS}


if __name__ == "__main__":
    for s in ["../Result/Heston_SVatt_N1e4_delta1.0_alpha10_tran0e0_part3.pth",
              "Heston_SVatt_N1e5_delta0.0_alpha1_tran0e0_part1.pth",
              "Heston_SVpen_N5e3_lam12.5_alpha1_tran0e0_part2_log.csv",
              "HestonClean_N2e4_tran0e0_part1.pth",
              "Heston_SVatt_N1e4_delta1.0_alpha10_tran5e-03",
              "Heston.pth"]:
        print(s, "->", parse_name(s))
