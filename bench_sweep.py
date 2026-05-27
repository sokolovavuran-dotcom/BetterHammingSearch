"""Sweep linear_scan.exe and qadp_normal.exe across (N x r) and produce
QPS plots + a speedup heatmap.

Datasets : siftsmall (10k), sift_quarter (250k), sift_half (500k), sift (1M)
Radii    : 4, 8, 12, 16, 20
Algos    : linear_scan, qadp_normal  (C++ binaries built in cpp/)

Outputs:
  sweep_results.csv       — raw numbers, one row per (algo, dataset, r)
  sweep_results.json      — same data, machine-readable
  sweep_qps.png           — grid of QPS-vs-N subplots, one per radius
  sweep_speedup.png       — heatmap of qadp_qps / linear_qps (>1 = qadp wins)
"""
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CPP_DIR    = HERE / 'cpp'
LINEAR     = CPP_DIR / 'linear_scan.exe'
QADP_N     = CPP_DIR / 'qadp_normal.exe'
QADP_S     = CPP_DIR / 'qadp_small.exe'
MIH        = CPP_DIR / 'mih.exe'

DATASETS = [
    ('siftsmall',    10_000),
    ('sift_quarter', 250_000),
    ('sift_half',    500_000),
    ('sift',       1_000_000),
]
RADII   = [2, 4, 6, 8, 12, 16, 20]
ALGOS   = [
    ('linear_scan', LINEAR),
    ('qadp_normal', QADP_N),
    ('qadp_small',  QADP_S),     # designed for r < 0.05*d = 6.4 at d=128
    ('mih',         MIH),         # MIH with m=8 (16-bit subs), exact range query
]
Q       = 10                  # queries per run

# MSYS2 MinGW runtime needs to be on PATH so the .exe can find its DLLs.
_MINGW = r'C:\msys64\mingw64\bin'
ENV = {**os.environ}
if os.path.isdir(_MINGW):
    ENV['PATH'] = _MINGW + os.pathsep + ENV.get('PATH', '')

QPS_RE   = re.compile(r'QPS\s*:\s*([0-9.]+)')
AVG_RE   = re.compile(r'avg query\s*:\s*([0-9.]+)\s*ms')
HITS_RE  = re.compile(r'avg hits\s*:\s*([0-9.]+)')
TOTAL_RE = re.compile(r'total\s*:\s*([0-9.]+)\s*s')


def run_one(exe: Path, dataset: str, radius: int) -> dict:
    if not exe.exists():
        raise FileNotFoundError(f'Binary not found: {exe}')
    cmd = [str(exe),
           '--dataset',     dataset,
           '--radius',      str(radius),
           '--query_count', str(Q)]
    t0 = time.perf_counter()
    res = subprocess.run(cmd, capture_output=True, text=True, env=ENV,
                         cwd=str(CPP_DIR))
    wall = time.perf_counter() - t0
    stdout = res.stdout + res.stderr
    out = {
        'qps'      : None,
        'avg_ms'   : None,
        'avg_hits' : None,
        'total_s'  : None,
        'wall_s'   : wall,
        'rc'       : res.returncode,
    }
    for key, regex in (('qps', QPS_RE), ('avg_ms', AVG_RE),
                       ('avg_hits', HITS_RE), ('total_s', TOTAL_RE)):
        m = regex.search(stdout)
        if m: out[key] = float(m.group(1))
    if out['qps'] is None:
        out['stdout'] = stdout[-1500:]      # only on failure, keep last bit
    return out


def main():
    needed = [LINEAR, QADP_N, QADP_S, MIH]
    missing = [str(p) for p in needed if not p.exists()]
    if missing:
        sys.exit(f'Build first: cd cpp && g++ ... (see cpp/README.md)\n'
                 f'  missing: {missing}')

    results = []
    n_total = len(DATASETS) * len(RADII) * len(ALGOS)
    n_done  = 0
    t_start = time.perf_counter()

    for ds_name, ds_n in DATASETS:
        for r in RADII:
            for algo_name, algo_exe in ALGOS:
                n_done += 1
                print(f'[{n_done:>2}/{n_total}]  {algo_name:<12s}  {ds_name:<14s}  r={r:<2d}',
                      end='  ', flush=True)
                try:
                    out = run_one(algo_exe, ds_name, r)
                except Exception as e:
                    print(f'FAILED: {e}')
                    out = {'qps': None, 'avg_ms': None, 'avg_hits': None,
                           'total_s': None, 'wall_s': None, 'rc': -1,
                           'error': str(e)}
                row = {'algo': algo_name, 'dataset': ds_name, 'N': ds_n,
                       'r': r, **out}
                results.append(row)
                if out.get('qps') is None:
                    print(f'(no QPS parsed, rc={out["rc"]})')
                else:
                    print(f'QPS={out["qps"]:>10.1f}   avg={out["avg_ms"]:>7.2f} ms   '
                          f'hits={out["avg_hits"]:>8.1f}   wall={out["wall_s"]:>5.1f} s')

    elapsed = time.perf_counter() - t_start
    print(f'\nSweep finished in {elapsed:.1f} s')

    # --- write CSV + JSON --------------------------------------------------
    csv_path  = HERE / 'sweep_results.csv'
    json_path = HERE / 'sweep_results.json'
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['algo', 'dataset', 'N', 'r', 'qps', 'avg_ms', 'avg_hits',
                    'total_s', 'wall_s', 'rc'])
        for row in results:
            w.writerow([row.get(k) for k in ('algo','dataset','N','r','qps',
                                              'avg_ms','avg_hits','total_s',
                                              'wall_s','rc')])
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f'  -> {csv_path.name}')
    print(f'  -> {json_path.name}')

    # --- plot --------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as e:
        print(f'matplotlib unavailable ({e}); skipping plots.')
        return

    def lookup(algo, dataset, r):
        for row in results:
            if row['algo'] == algo and row['dataset'] == dataset and row['r'] == r:
                return row
        return None

    # ---- Figure 1: QPS vs N grid, one subplot per radius -----------------
    fig, axes = plt.subplots(1, len(RADII), figsize=(4.5 * len(RADII), 4.5),
                             sharey=True)
    Ns = [n for _, n in DATASETS]

    for ax, r in zip(axes, RADII):
        for algo_name, _ in ALGOS:
            ys = [(lookup(algo_name, ds, r) or {}).get('qps') for ds, _ in DATASETS]
            ax.loglog(Ns, ys, marker='o', label=algo_name)
        ax.set_title(f'r = {r}')
        ax.set_xlabel('N (base size)')
        ax.grid(True, which='both', alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel('QPS  (queries / sec)')
    fig.suptitle('Linear scan vs QADP-NC — QPS vs N  (C++ implementations)')
    plt.tight_layout()
    qps_png = HERE / 'sweep_qps.png'
    plt.savefig(qps_png, dpi=110)
    plt.close(fig)
    print(f'  -> {qps_png.name}')

    # ---- Figure 2: per-algorithm speedup heatmaps (vs linear_scan) -------
    other_algos = [a for a, _ in ALGOS if a != 'linear_scan']
    fig, axes = plt.subplots(1, len(other_algos), figsize=(7 * len(other_algos), 5))
    if len(other_algos) == 1:
        axes = [axes]
    for ax, algo in zip(axes, other_algos):
        mat = np.full((len(RADII), len(DATASETS)), np.nan)
        for i, r in enumerate(RADII):
            for j, (ds, _) in enumerate(DATASETS):
                ls = lookup('linear_scan', ds, r)
                qd = lookup(algo, ds, r)
                if ls and qd and ls.get('qps') and qd.get('qps'):
                    mat[i, j] = qd['qps'] / ls['qps']

        finite = mat[np.isfinite(mat) & (mat > 0)]
        vmax = float(max(finite.max(), 1.0 / finite.min())) if finite.size else 2.0
        norm = matplotlib.colors.LogNorm(vmin=1.0 / vmax, vmax=vmax)
        im = ax.imshow(mat, aspect='auto', cmap='RdYlGn', norm=norm)
        ax.set_xticks(range(len(DATASETS)))
        ax.set_xticklabels([f'{ds}\nN={n:,}' for ds, n in DATASETS], fontsize=8)
        ax.set_yticks(range(len(RADII)))
        ax.set_yticklabels([f'r={r}' for r in RADII])
        ax.set_xlabel('Dataset')
        ax.set_ylabel('Hamming radius')
        ax.set_title(f'{algo} QPS / linear_scan QPS\n'
                     '(green = wins, red = loses)')
        plt.colorbar(im, ax=ax, label='speedup (log)')
        for i in range(len(RADII)):
            for j in range(len(DATASETS)):
                v = mat[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                            color='black' if 0.5 < v < 2 else 'white',
                            fontsize=9)
    fig.suptitle('Per-algorithm speedup ratios versus linear_scan')
    plt.tight_layout()
    sp_png = HERE / 'sweep_speedup.png'
    plt.savefig(sp_png, dpi=110)
    plt.close(fig)
    print(f'  -> {sp_png.name}')

    # ---- Figure 3: winner-take-all map -----------------------------------
    # For each (N, r) cell, which algorithm has the highest QPS?
    winner_map = np.zeros((len(RADII), len(DATASETS)), dtype=int)
    algo_names = [a for a, _ in ALGOS]
    speedup_over_2nd = np.zeros((len(RADII), len(DATASETS)))
    for i, r in enumerate(RADII):
        for j, (ds, _) in enumerate(DATASETS):
            qps = []
            for a in algo_names:
                row = lookup(a, ds, r)
                qps.append(row['qps'] if row and row.get('qps') else 0.0)
            best = int(np.argmax(qps))
            winner_map[i, j] = best
            sorted_qps = sorted(qps, reverse=True)
            if sorted_qps[1] > 0:
                speedup_over_2nd[i, j] = sorted_qps[0] / sorted_qps[1]
            else:
                speedup_over_2nd[i, j] = float('inf')

    fig, ax = plt.subplots(figsize=(8, 5))
    palette = plt.get_cmap('Set2')(np.linspace(0, 1, len(algo_names)))
    cmap = matplotlib.colors.ListedColormap(palette)
    ax.imshow(winner_map, aspect='auto', cmap=cmap,
              vmin=-0.5, vmax=len(algo_names) - 0.5)
    ax.set_xticks(range(len(DATASETS)))
    ax.set_xticklabels([f'{ds}\nN={n:,}' for ds, n in DATASETS], fontsize=8)
    ax.set_yticks(range(len(RADII)))
    ax.set_yticklabels([f'r={r}' for r in RADII])
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Hamming radius')
    ax.set_title('Fastest algorithm at each (N, r) cell\n'
                 '(annotation: QPS speedup over runner-up)')
    # Legend
    handles = [plt.Rectangle((0, 0), 1, 1, color=palette[i])
               for i in range(len(algo_names))]
    ax.legend(handles, algo_names, loc='upper right', bbox_to_anchor=(1.4, 1))
    for i in range(len(RADII)):
        for j in range(len(DATASETS)):
            s = speedup_over_2nd[i, j]
            txt = f'{algo_names[winner_map[i,j]]}\n{s:.1f}x' if np.isfinite(s) else algo_names[winner_map[i,j]]
            ax.text(j, i, txt, ha='center', va='center', fontsize=8)
    plt.tight_layout()
    win_png = HERE / 'sweep_winners.png'
    plt.savefig(win_png, dpi=110, bbox_inches='tight')
    plt.close(fig)
    print(f'  -> {win_png.name}')


if __name__ == '__main__':
    main()
