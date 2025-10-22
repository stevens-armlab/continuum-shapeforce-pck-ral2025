#!/usr/bin/env python3
# 2x2 correlation plots with log-|e_F| axis, straight trend lines in log space,
# and larger, configurable font sizes. X-axis labels use the paper's metric symbols.

import argparse, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def _norm_sub(s: str) -> str:
    s = str(s).strip()
    m = re.search(r'(\d+)\s*[-]\s*(\d+)', s)
    return f"C{int(m.group(1))}-{int(m.group(2))}" if m else s

def pearson_r(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if x.size < 2 or np.std(x) == 0.0 or np.std(y) == 0.0: return np.nan
    return float(np.corrcoef(x, y)[0, 1])

def add_logy_fit(ax, x, y, eps, lw):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size < 2 or np.std(x) == 0.0: return
    yl = np.log10(y + eps)               # fit in log space
    a, b = np.polyfit(x, yl, 1)
    xx = np.linspace(x.min(), x.max(), 200)
    yy = (10.0**(a*xx + b)) - eps
    ax.plot(xx, yy, linewidth=lw, zorder=1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", type=str, default="shape_table.csv")
    ap.add_argument("--force", type=str, default="force_table.csv")
    ap.add_argument("--eps", type=float, default=1e-3, help="epsilon added to |e_F| for log axis")
    # Font & style controls (bigger sensible defaults)
    ap.add_argument("--font", type=float, default=13, help="base font size")
    ap.add_argument("--axis", type=float, default=13, help="axis label size")
    ap.add_argument("--tick", type=float, default=11, help="tick label size")
    ap.add_argument("--legend", type=float, default=11, help="legend font size")
    ap.add_argument("--title", type=float, default=14, help="figure title size")
    ap.add_argument("--ms", type=float, default=40, help="marker size (points^2 in scatter)")
    ap.add_argument("--lw", type=float, default=1.6, help="trend line width")
    args = ap.parse_args()

    # Global font settings
    plt.rcParams.update({
        "font.size": args.font,
        "axes.labelsize": args.axis,
        "axes.titlesize": args.title,
        "legend.fontsize": args.legend,
        "xtick.labelsize": args.tick,
        "ytick.labelsize": args.tick,
    })

    S = pd.read_csv(args.shape)
    F = pd.read_csv(args.force)
    S["Sub"] = S["subplot"].map(_norm_sub)
    F["Sub"] = F["subplot"].map(_norm_sub)

    # |e_F| per method
    for tag in ["pck0","pck1","pck2"]:
        F[f"{tag}_abs_err"] = np.sqrt(F[f"Fx_err_{tag}"].astype(float)**2 +
                                      F[f"Fz_err_{tag}"].astype(float)**2)

    DF = pd.merge(
        S,
        F[["case_idx","Sub","pck0_abs_err","pck1_abs_err","pck2_abs_err"]],
        on=["case_idx","Sub"], how="inner"
    )

    # Columns for each method
    shape_cols = {
        "PCK0": ("P0_tip_pos_mm","P0_tip_rot_deg","P0_shape_pos_mm","P0_shape_rot_deg"),
        "PCK1": ("P1_tip_pos_mm","P1_tip_rot_deg","P1_shape_pos_mm","P1_shape_rot_deg"),
        "PCK2": ("P2_tip_pos_mm","P2_tip_rot_deg","P2_shape_pos_mm","P2_shape_rot_deg"),
    }
    force_col = {"PCK0":"pck0_abs_err","PCK1":"pck1_abs_err","PCK2":"pck2_abs_err"}

    # X-axis labels that match your metric definitions (raw strings for mathtext)
    metric_names = [
        (r'$e_{\mathrm{tip,pos}}$ [mm]',   0),
        (r'$e_{\mathrm{tip,rot}}$ [deg]',  1),
        (r'$e_{\mathrm{shape,pos}}$ [mm]', 2),
        (r'$e_{\mathrm{shape,rot}}$ [deg]',3),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.2), constrained_layout=True)
    axes = axes.ravel()

    for ax, (xlabel, idx) in zip(axes, metric_names):
        handles, labels = [], []
        for method in ["PCK0","PCK1","PCK2"]:
            scol = shape_cols[method][idx]; fcol = force_col[method]
            x = DF[scol].astype(float).to_numpy()
            y = DF[fcol].astype(float).to_numpy()
            y_plot = y + args.eps  # >0 for log axis
            h = ax.scatter(x, y_plot, alpha=0.85, label=method, s=args.ms)
            add_logy_fit(ax, x, y, eps=args.eps, lw=args.lw)
            rho = pearson_r(x, np.log10(y_plot))  # correlation in the plotted space
            labels.append(f"{method} (\u03C1={rho:.2f})" if np.isfinite(rho) else f"{method} (\u03C1=NA)")
            handles.append(h)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r'$e_F$ [N]')   # scalar magnitude e_F = ||e_F||_2
        ax.set_yscale("log", base=10)
        ax.legend(handles, labels, loc= "lower right",frameon=True)

    # Optional title
    # fig.suptitle("Shape error vs. force error magnitude (\u03C1 on log e_F)", fontsize=args.title)
    plt.show()

if __name__ == "__main__":
    main()
