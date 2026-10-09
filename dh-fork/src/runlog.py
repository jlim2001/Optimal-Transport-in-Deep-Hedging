"""
Per-epoch training log format, shared by Heston_train_pen.py (written live) and
parse_logs.py (recovered from the authors' printed output).

One CSV per trained model:  ../Result/<run>_part<K>_log.csv
Columns:
  epoch       0-based epoch index
  phase       "clean" (warm-up) or "robust"
  clean_loss  mean clean CVaR loss over the epoch's batches
  att_loss    mean attacked loss (0 during warm-up)
  radius      mean realised transport radius of the adversary (penalized: measured;
              constrained: nominal delta; empty if unknown)
  seconds     wall-clock time of the epoch
  p0_clean, p0_att   trained CVaR thresholds
  lr          learning rate used in the epoch (empty if unknown)
"""
import csv

LOG_COLUMNS = ["epoch", "phase", "clean_loss", "att_loss", "radius", "seconds",
               "p0_clean", "p0_att", "lr"]


class EpochLogger:
    """Append-and-flush CSV writer, so a crashed run still leaves its log."""

    def __init__(self, path):
        self.path = path
        self._fh = open(path, "w", newline="")
        self._w = csv.DictWriter(self._fh, fieldnames=LOG_COLUMNS)
        self._w.writeheader()
        self._fh.flush()

    def log(self, **row):
        self._w.writerow({k: row.get(k, "") for k in LOG_COLUMNS})
        self._fh.flush()

    def close(self):
        self._fh.close()


def write_log(path, rows):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=LOG_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in LOG_COLUMNS})
