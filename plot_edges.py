# Save as: C:\eda_v2\eda_output_tasks\plot_edges.py
# Run: python plot_edges.py

import numpy as np
import matplotlib.pyplot as plt
import os

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12,
    'figure.facecolor': 'white', 'savefig.facecolor': 'white',
    'savefig.bbox': 'tight', 'savefig.dpi': 200
})

DATA = r"C:\eda_v2\eda_output_tasks\tensor_data.npz"
OUT = r"C:\eda_v2\eda_output_tasks\plots_filtered"
os.makedirs(OUT, exist_ok=True)

B_W, A05_W, A2_W, A3_W = 0, 1, 2, 3
B_L, A05_L, A2_L, A3_L = 4, 5, 6, 7
FZ, ZMAX, THICK, XFRAC = 8, 9, 10, 11

d = np.load(DATA)['data']
print(f"Loaded: {d.shape}")

interior = (
    (d[:, XFRAC] > 0) & (d[:, XFRAC] < 9) &
    (d[:, FZ] > 3.0) &
    (d[:, FZ] < d[:, ZMAX] - 3.0)
)
d_int = d[interior]
print(f"All: {len(d):,}, Interior: {len(d_int):,} ({len(d_int)/len(d)*100:.1f}%)")

COLORS = ['#2196F3', '#4CAF50']

for tag, col_w, col_l in [('0.5m', A05_W, A05_L), ('2.0m', A2_W, A2_L), ('3.0m', A3_W, A3_L)]:
    ab_all = np.concatenate([d[:, col_w] - d[:, B_W], d[:, col_l] - d[:, B_L]])
    ab_int = np.concatenate([d_int[:, col_w] - d_int[:, B_W], d_int[:, col_l] - d_int[:, B_L]])
    xlim = np.percentile(np.abs(ab_all), 99.5)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for col, (data_arr, label, color) in enumerate([
        (ab_all, 'All points (incl. edges)', COLORS[0]),
        (ab_int, 'Interior only (edges removed)', COLORS[1])
    ]):
        for row, use_log in enumerate([True, False]):
            ax = axes[row, col]
            ax.hist(data_arr, bins=80, color=color, alpha=0.85, edgecolor='white', linewidth=0.3, log=use_log)
            ax.axvline(0, color='black', linestyle='--', linewidth=0.8)
            ax.axvline(data_arr.mean(), color='red', linestyle='--', linewidth=1.5)
            ax.text(0.97, 0.95, f'mean={data_arr.mean():.3f}\nstd={data_arr.std():.3f}\nn={len(data_arr):,}',
                    transform=ax.transAxes, ha='right', va='top', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            ax.set_xlim(-xlim, xlim)
            scale = 'Log scale' if use_log else 'Linear scale'
            ax.set_title(f'{label} — {scale}', fontweight='bold')
            ax.set_xlabel('Δp (shell − surface)')
            ax.set_ylabel(f'Count ({"log" if use_log else "linear"})')

    fig.suptitle(f'(a-b) at {tag}: edge effect comparison', fontsize=16, fontweight='bold')
    fig.tight_layout()
    path = os.path.join(OUT, f'edge_compare_ab_{tag}.png')
    fig.savefig(path)
    plt.close()
    print(f"  Saved {path}")

print("Done!")