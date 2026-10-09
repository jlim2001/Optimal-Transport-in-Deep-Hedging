"""
Turn evaluation CSVs and training logs into the report's tables and figures.

Inputs
  --eval   CSVs written by evaluate.py (rows per model file, tagged with method/N/delta/lam)
  --logs   per-epoch training logs (*_log.csv) from Heston_train_pen.py or parse_logs.py
Outputs (in --outdir, default ../Result/summary)
  runs.csv              one row per run: mean/std over partitions of every metric, the
                        adversary's training radius, seconds per robust epoch, n_parts
  main_table.tex        RQ1  in-dist / OOD mean / OOD worst CVaR per N and run
  cross_table.csv/.tex  RQ3  every run under every attack column present in --eval
  sensitivity.csv       RQ2  spread and regret of OOD CVaR over each method's grid at --N
  fig_sensitivity.pdf/.png   RQ2  OOD CVaR vs training radius, both methods, at --N
  fig_radius.pdf/.png        adversary radius during the robust phase, at --N
  cost.csv              seconds per robust-phase epoch per method (context only)

Training radius (the common x-axis for RQ2):
  constrained : nominal delta (the budget attack is projected to radius delta)
  penalized   : mean logged radius over the robust phase, averaged over partitions
Clean runs (authors' adversarial script with delta = 0) are the reference line.

Usage:
    python summarise.py --eval ../Result/eval_*.csv --logs "../Result/*_log.csv" --N 10000
Needs pandas and matplotlib.
"""
import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from naming import parse_name

# ---- colours: validated categorical slots 1-2 (light surface), neutral for reference ----
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
METHOD_STYLE = {
    "constrained": dict(color="#2a78d6", marker="o", label="Constrained (radius $\\delta$)"),
    "penalized":   dict(color="#eb6834", marker="s", label="Penalized (penalty $\\lambda$)"),
}
REF_COLOR = INK2
METRICS = ["cvar_in", "cvar_ood_mean", "cvar_ood_worst"]
METRIC_NAMES = {"cvar_in": "In-dist.", "cvar_ood_mean": "OOD mean", "cvar_ood_worst": "OOD worst"}


def style_axes(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)


def save(fig, outdir, stem):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(outdir, f"{stem}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ loading
def load_eval(patterns):
    files = sorted({f for p in patterns for f in glob.glob(p)})
    if not files:
        raise SystemExit("no evaluation CSVs matched --eval")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    # older CSVs without metadata columns: parse from the model path
    if "method" not in df.columns:
        meta = pd.DataFrame([parse_name(m) or {} for m in df["model"]])
        df = pd.concat([df, meta], axis=1)
    df = df.dropna(subset=["method"])
    # the same model may appear in several eval files (different attack grids): merge them
    df = df.groupby("model", as_index=False).first()
    return df


def load_logs(pattern):
    rows = []
    for f in sorted(glob.glob(pattern)):
        meta = parse_name(f)
        if not meta:
            continue
        log = pd.read_csv(f)
        robust = log[log["phase"] == "robust"]
        rows.append(dict(run=meta["run"], part=meta["part"],
                         train_radius_logged=pd.to_numeric(robust["radius"], errors="coerce").mean(),
                         sec_robust_epoch=pd.to_numeric(robust["seconds"], errors="coerce").mean(),
                         sec_clean_epoch=pd.to_numeric(log.loc[log["phase"] == "clean", "seconds"],
                                                       errors="coerce").mean(),
                         n_epochs=len(log)))
    return pd.DataFrame(rows), {parse_name(f)["run"] + f"|{parse_name(f)['part']}": f
                                for f in glob.glob(pattern) if parse_name(f)}


# ------------------------------------------------------------------ aggregation
def per_run(ev, logs):
    keys = ["run", "method", "attack", "N", "delta", "lam", "alpha"]
    num = [c for c in ev.columns if c.startswith("cvar_") or c.startswith("radius_")]
    g = ev.groupby(keys, dropna=False)
    out = g[num].mean().add_suffix("_mean").join(g[num].std().add_suffix("_std"))
    out["n_parts"] = g.size()
    out = out.reset_index()
    if len(logs):
        lg = logs.groupby("run")[["train_radius_logged", "sec_robust_epoch", "sec_clean_epoch"]].mean()
        out = out.merge(lg, left_on="run", right_index=True, how="left")
    else:
        out["train_radius_logged"] = np.nan
        out["sec_robust_epoch"] = np.nan
        out["sec_clean_epoch"] = np.nan
    out["train_radius"] = np.where(out["method"] == "constrained", out["delta"],
                                   np.where(out["method"] == "penalized",
                                            out["train_radius_logged"], 0.0))
    return out.sort_values(["N", "method", "delta", "lam"]).reset_index(drop=True)


def hyper_label(r):
    if r["method"] == "penalized":
        return f"$\\lambda={r['lam']:g}$"
    if r["method"] == "constrained":
        return f"$\\delta={r['delta']:g}$"
    return ""


def fmt(m, s):
    if pd.isna(m):
        return "--"
    return f"{m:.3f}" if pd.isna(s) else f"{m:.3f} $\\pm$ {s:.3f}"


METHOD_TEX = {"clean": "Clean", "clean_authors": "Clean (authors' script)",
              "constrained": "Constrained", "penalized": "Penalized"}


# ------------------------------------------------------------------ tables
def main_table(runs, path):
    lines = [r"\begin{tabular}{llccccc}\toprule",
             r"$N$ & Method & Hyperparam. & In-dist. & OOD mean & OOD worst & parts\\\midrule"]
    for i, (N, grp) in enumerate(runs.groupby("N")):
        if i:
            lines.append(r"\midrule")
        for j, (_, r) in enumerate(grp.iterrows()):
            cells = [f"{N // 1000}k" if j == 0 else "", METHOD_TEX.get(r["method"], r["method"]),
                     hyper_label(r)]
            cells += [fmt(r[f"{m}_mean"], r[f"{m}_std"]) for m in METRICS]
            cells.append(str(int(r["n_parts"])))
            lines.append(" & ".join(cells) + r"\\")
    lines.append(r"\bottomrule\end{tabular}")
    open(path, "w").write("\n".join(lines) + "\n")


def cross_table(runs, outdir):
    att_cols = sorted({c[:-5] for c in runs.columns
                       if c.endswith("_mean") and (c.startswith("cvar_constrained_d")
                                                   or c.startswith("cvar_penalized_l"))},
                      key=lambda c: (c.split("_")[1], float(re.sub(r"^.*_[dl]", "", c))))
    keep = ["run", "method", "N", "delta", "lam", "train_radius"] + [f"{c}_mean" for c in att_cols]
    runs[keep].to_csv(os.path.join(outdir, "cross_table.csv"), index=False)
    if not att_cols:
        return
    head = []
    for c in att_cols:
        kind, val = c.split("_")[1], re.sub(r"^.*_[dl]", "", c)
        head.append(f"Constr.\\ $\\delta={val}$" if kind == "constrained" else f"Pen.\\ $\\lambda={val}$")
    lines = [r"\begin{tabular}{ll" + "c" * len(att_cols) + r"}\toprule",
             r"$N$ & Trained with & " + " & ".join(head) + r"\\\midrule"]
    for _, r in runs.iterrows():
        name = METHOD_TEX.get(r["method"], r["method"]) + (" " + hyper_label(r) if hyper_label(r) else "")
        vals = [fmt(r[f"{c}_mean"], r.get(f"{c}_std", np.nan)) for c in att_cols]
        lines.append(f"{int(r['N']) // 1000}k & {name} & " + " & ".join(vals) + r"\\")
    lines.append(r"\bottomrule\end{tabular}")
    open(os.path.join(outdir, "cross_table.tex"), "w").write("\n".join(lines) + "\n")


def sensitivity_stats(runs, N, path):
    sub = runs[(runs["N"] == N) & runs["method"].isin(["constrained", "penalized"])]
    rows = []
    for meth, g in sub.groupby("method"):
        y = g["cvar_ood_mean_mean"].dropna()
        if y.empty:
            continue
        best = y.min()
        rows.append(dict(method=meth, N=N, grid_points=len(y),
                         radius_min=g["train_radius"].min(), radius_max=g["train_radius"].max(),
                         ood_best=best, ood_worst_choice=y.max(), spread=y.max() - y.min(),
                         mean_regret=(y - best).mean(),
                         note="spread/regret over the hyperparameter grid; lower = less sensitive"))
    pd.DataFrame(rows).to_csv(path, index=False)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ figures
def fig_sensitivity(runs, N, outdir):
    sub = runs[runs["N"] == N]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    style_axes(ax)
    for meth in ("constrained", "penalized"):
        g = sub[(sub["method"] == meth)].dropna(subset=["train_radius", "cvar_ood_mean_mean"])
        g = g[g["train_radius"] > 0].sort_values("train_radius")
        if g.empty:
            continue
        st = METHOD_STYLE[meth]
        ax.errorbar(g["train_radius"], g["cvar_ood_mean_mean"], yerr=g["cvar_ood_mean_std"].fillna(0),
                    color=st["color"], marker=st["marker"], markersize=6, linewidth=2,
                    elinewidth=1, capsize=3, label=st["label"],
                    markeredgecolor="white", markeredgewidth=1.2)
        last = g.iloc[-1]                                     # direct label at line end
        ax.annotate(meth.capitalize(), (last["train_radius"], last["cvar_ood_mean_mean"]),
                    xytext=(10, 0), textcoords="offset points", va="center", fontsize=9, color=INK2)
    clean = sub[sub["method"] == "clean"]["cvar_ood_mean_mean"].dropna()
    if not clean.empty:
        ax.axhline(clean.mean(), color=REF_COLOR, linestyle="--", linewidth=1.2, label="Clean (no attack)")
    ax.set_xscale("log")
    ax.set_xlabel("Adversary radius during training (log scale)")
    ax.set_ylabel("Out-of-distribution CVaR$_{0.5}$ (lower is better)")
    ax.set_title(f"Sensitivity to the robustness hyperparameter, N = {N:,}", fontsize=11,
                 color=INK, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="upper center",
              bbox_to_anchor=(0.5, -0.16), ncol=3)
    save(fig, outdir, "fig_sensitivity")


def fig_radius(log_files, runs, N, outdir):
    pen = runs[(runs["N"] == N) & (runs["method"] == "penalized")]
    con = runs[(runs["N"] == N) & (runs["method"] == "constrained")]
    if pen.empty and con.empty:
        return
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    style_axes(ax)
    first = True
    for _, r in pen.iterrows():
        logs = [pd.read_csv(f) for k, f in log_files.items() if k.split("|")[0] == r["run"]]
        if not logs:
            continue
        rob = [l[l["phase"] == "robust"][["epoch", "radius"]].set_index("epoch")["radius"] for l in logs]
        curve = pd.concat(rob, axis=1).apply(pd.to_numeric, errors="coerce").mean(axis=1)
        ax.plot(curve.index, curve.values, color=METHOD_STYLE["penalized"]["color"], linewidth=2,
                label=METHOD_STYLE["penalized"]["label"] if first else None)
        first = False
        ax.annotate(f"$\\lambda={r['lam']:g}$", (curve.index[-1], curve.values[-1]), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=9, color=INK2)
    for i, (_, r) in enumerate(con.iterrows()):
        ax.axhline(r["delta"], color=METHOD_STYLE["constrained"]["color"], linestyle="--", linewidth=1.2,
                   label="Constrained (fixed $\\delta$)" if i == 0 else None)
        ax.annotate(f"$\\delta={r['delta']:g}$", (0.0, r["delta"]), xycoords=("axes fraction", "data"),
                    xytext=(4, 3), textcoords="offset points", va="bottom", fontsize=9, color=INK2)
    ax.set_yscale("log")
    ax.set_xlabel("Epoch (robust phase)")
    ax.set_ylabel("Adversary radius (log scale)")
    ax.set_title(f"Radius of the adversary during training, N = {N:,}", fontsize=11, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="upper center",
              bbox_to_anchor=(0.5, -0.16), ncol=2)
    save(fig, outdir, "fig_radius")


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", nargs="+", required=True)
    ap.add_argument("--logs", default="../Result/*_log.csv")
    ap.add_argument("--N", type=int, default=None, help="N for the sensitivity/radius figures "
                                                       "(default: the N with the most runs)")
    ap.add_argument("--outdir", default="../Result/summary")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    ev = load_eval(args.eval)
    logs, log_files = load_logs(args.logs)
    runs = per_run(ev, logs)
    runs.to_csv(os.path.join(args.outdir, "runs.csv"), index=False)

    main_table(runs, os.path.join(args.outdir, "main_table.tex"))
    cross_table(runs, args.outdir)
    N = args.N or int(runs[runs["method"].isin(["constrained", "penalized"])]["N"].mode().iloc[0])
    sens = sensitivity_stats(runs, N, os.path.join(args.outdir, "sensitivity.csv"))
    fig_sensitivity(runs, N, args.outdir)
    fig_radius(log_files, runs, N, args.outdir)
    runs.groupby(["N", "method"])[["sec_clean_epoch", "sec_robust_epoch"]].mean().to_csv(
        os.path.join(args.outdir, "cost.csv"))

    missing = runs[(runs["method"] == "penalized") & runs["train_radius"].isna()]["run"].tolist()
    if missing:
        print("WARNING: no training log for these penalized runs (left out of fig_sensitivity):",
              *missing, sep="\n  ")
    print(f"{len(ev)} model rows, {len(runs)} runs -> {args.outdir}")
    if len(sens):
        print(sens[["method", "grid_points", "spread", "mean_regret"]].to_string(index=False))


if __name__ == "__main__":
    main()
