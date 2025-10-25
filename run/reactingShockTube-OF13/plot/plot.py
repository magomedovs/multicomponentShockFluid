#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot reactingShockTube results using OpenFOAM v13 sampling output.

Supports:
- v13 format: ../postProcessing/sample/<time>/x.xy with header line:
  "# x T U_x U_y U_z H"
- Legacy fallback: x_H_T_U.xy assumed column order [x, H, T, Ux, Uy?, Uz?]

Also picks the closest numeric time directory to requested microsecond times.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# ---------------- user config ----------------
TIME_POINTS_US = [170, 190, 230]           # microseconds
POST_BASE = Path("../postProcessing/sample")
SET_NAME = "x"                              # sampling set name
REF_DIR = Path("benchmarkData")             # T_170.csv, u_170.csv, H_170.csv
USE_TEX = True                              # requires dvipng/texlive installed
# --------------------------------------------

try:
    plt.rcParams['text.usetex'] = bool(USE_TEX)
except Exception:
    plt.rcParams['text.usetex'] = False

def closest_time_dir(pp_root: Path, target_s: float) -> Path:
    cands = []
    for p in pp_root.iterdir():
        if p.is_dir():
            try:
                val = float(p.name)
                cands.append((abs(val - target_s), val, p))
            except ValueError:
                pass
    if not cands:
        raise FileNotFoundError(f"No numeric time directories found under {pp_root}")
    cands.sort(key=lambda x: x[0])
    return cands[0][2]

def find_xy_with_header(t_dir: Path, set_name: str) -> Path:
    cand = t_dir / f"{set_name}.xy"
    if cand.is_file():
        return cand
    # legacy filename pattern
    legacy = t_dir / f"{set_name}_H_T_U.xy"
    if legacy.is_file():
        return legacy
    # last resort: any matching .xy for the set
    alts = sorted(t_dir.glob(f"{set_name}_*.xy"))
    if alts:
        return alts[0]
    raise FileNotFoundError(f"Neither {set_name}.xy nor {set_name}_*.xy found in {t_dir}")

def parse_header_first_line(path: Path):
    """Return list of header tokens if first non-empty line starts with #, else None."""
    with path.open("r", errors="ignore") as f:
        for line in f:
            if not line.strip():
                continue
            if line.lstrip().startswith("#"):
                # tokens after '#'
                toks = line.lstrip("#").strip().split()
                return toks
            break
    return None

def load_xy(path: Path):
    """Load numeric data (transpose to shape (n_cols, n_pts))."""
    return np.loadtxt(path, comments="#").T

def map_columns(header_tokens, path: Path):
    """
    Determine indices for x, T, u, H.
    - v13 header like: ['x','T','U_x','U_y','U_z','H']
    - legacy no header: assume x_H_T_U.xy => [x,H,T,Ux,Uy,Uz]
    """
    if header_tokens:
        lower = [t.lower() for t in header_tokens]
        def find_any(names, default=None):
            for n in names:
                if n in lower:
                    return lower.index(n)
            return default
        ix = find_any(["x"])
        iT = find_any(["t","temperature"])
        iu = find_any(["u_x","ux","u"])  # axial component
        iH = find_any(["h","y_h","yh","massfraction_h"])
        return {"x": ix, "T": iT, "u": iu, "H": iH}
    # legacy ordering for x_H_T_U.xy
    name = path.name.lower()
    if name.endswith("_h_t_u.xy") or name == "x_h_t_u.xy":
        return {"x": 0, "H": 1, "T": 2, "u": 3}
    # no header, generic fallback: assume [x, T, Ux, Uy, Uz, H] if 6 cols, else [x,H,T,u]
    return {"x": 0, "T": 1, "u": 2, "H": -1}

def load_ref(var: str, t_us: int):
    p = REF_DIR / f"{var}_{t_us}.csv"
    return np.loadtxt(p).T

def main():
    # load references
    ref_T = [load_ref("T", t) for t in TIME_POINTS_US]
    ref_u = [load_ref("u", t) for t in TIME_POINTS_US]
    ref_H = [load_ref("H", t) for t in TIME_POINTS_US]

    sim = []
    for t_us in TIME_POINTS_US:
        t_s = round(t_us * 1e-6, 10)
        t_dir = closest_time_dir(POST_BASE, t_s)
        xy_path = find_xy_with_header(t_dir, SET_NAME)
        header = parse_header_first_line(xy_path)
        arr = load_xy(xy_path)  # (n_cols, n_pts)
        col = map_columns(header, xy_path)

        # Handle negative index for H fallback (last column) if needed
        def safe_get(i):
            if i is None:
                return None
            if i == -1:
                return arr[-1] if arr.shape[0] >= 2 else None
            if 0 <= i < arr.shape[0]:
                return arr[i]
            return None

        x = safe_get(col["x"])
        T = safe_get(col["T"])
        u = safe_get(col["u"])
        H = safe_get(col["H"])

        if x is None:
            raise RuntimeError(f"Could not locate 'x' column in {xy_path}")
        sim.append({"x": x, "T": T, "u": u, "H": H, "path": str(xy_path)})

    # --------- T plot ---------
    fig1, ax1 = plt.subplots(figsize=(7, 6))
    for i, t_us in enumerate(TIME_POINTS_US):
        ax1.plot(ref_T[i][0] * 1e2, ref_T[i][1], color="black", marker="o",
                 markersize=6, linestyle="none", fillstyle="none",
                 label="Ref." if i == 0 else None)
        if sim[i]["T"] is not None:
            ax1.plot(sim[i]["x"] * 1e2, sim[i]["T"], color="black", linestyle="-",
                     label="reactingRhoCentralFoam" if i == 0 else None)
    ax1.set_xlim(0, 12); ax1.set_ylim(500, 3000)
    ax1.set_xlabel(r'$x$ [cm]', fontsize=25)
    ax1.set_ylabel(r'$T$ [K]', fontsize=25)
    ax1.grid(); ax1.tick_params(axis='x', labelsize=20); ax1.tick_params(axis='y', labelsize=20)
    ax1.legend(bbox_to_anchor=(0.9, 1.11), fontsize=15, ncol=2)
    fig1.savefig('reactingShockTube_T.pdf', bbox_inches='tight')

    # --------- u plot ---------
    fig2, ax2 = plt.subplots(figsize=(7, 6))
    for i, t_us in enumerate(TIME_POINTS_US):
        ax2.plot(ref_u[i][0] * 1e2, ref_u[i][1], color="black", marker="o",
                 markersize=6, linestyle="none", fillstyle="none",
                 label="Ref." if i == 0 else None)
        if sim[i]["u"] is not None:
            ax2.plot(sim[i]["x"] * 1e2, sim[i]["u"], color="black", linestyle="-",
                     label="reactingRhoCentralFoam" if i == 0 else None)
    ax2.set_xlim(0, 12); ax2.set_ylim(-600, 600)
    ax2.set_ylabel(r'$u$ [m/s]', fontsize=25)
    ax2.grid(); ax2.tick_params(axis='x', labelsize=20, labelbottom=False); ax2.tick_params(axis='y', labelsize=20)
    ax2.legend(bbox_to_anchor=(0.9, 1.11), fontsize=15, ncol=2)
    fig2.savefig('reactingShockTube_u.pdf', bbox_inches='tight')

    # --------- H plot ---------
    fig3, ax3 = plt.subplots(figsize=(7, 6))
    for i, t_us in enumerate(TIME_POINTS_US):
        ax3.plot(ref_H[i][0] * 1e2, ref_H[i][1] * 1e2, color="black", marker="o",
                 markersize=6, linestyle="none", fillstyle="none",
                 label="Ref." if i == 0 else None)
        if sim[i]["H"] is not None:
            ax3.plot(sim[i]["x"] * 1e2, sim[i]["H"] * 1e2, color="black", linestyle="-",
                     label="reactingRhoCentralFoam" if i == 0 else None)
    ax3.set_xlim(0, 12); ax3.set_ylim(0, 0.25)
    ax3.set_xlabel(r'$x$ [cm]', fontsize=25)
    ax3.set_ylabel(r'$Y_{H}$ [-]', fontsize=25)
    ax3.grid(); ax3.tick_params(axis='x', labelsize=20); ax3.tick_params(axis='y', labelsize=20)
    fig3.savefig('reactingShockTube_H.pdf', bbox_inches='tight')

    plt.show()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
