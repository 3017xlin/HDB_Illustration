"""
plot.py — Generate all EDA plots from extracted tensor.
Input: tensor_data.npz with shape (N, 12)
Columns: p_b_w, p_a05_w, p_a2_w, p_a3_w, p_b_l, p_a05_l, p_a2_l, p_a3_l, floor_z, z_max, thickness, x_frac
"""
import argparse, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12,
    'figure.facecolor': 'white', 'savefig.facecolor': 'white',
    'savefig.bbox': 'tight', 'savefig.dpi': 200
})
COLORS = ['#2196F3', '#FF9800', '#4CAF50', '#673AB7']
EPS = 1e-10

# Column indices
B_W, A05_W, A2_W, A3_W = 0, 1, 2, 3
B_L, A05_L, A2_L, A3_L = 4, 5, 6, 7
FZ, ZMAX, THICK, XFRAC = 8, 9, 10, 11

H_EDGES = [0, 40, 70, 9999]
H_LABELS = ['< 40m', '40–70m', '> 70m']


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def h_mask(d, i):
    return (d[:, ZMAX] >= H_EDGES[i]) & (d[:, ZMAX] < H_EDGES[i + 1])


def t_edges(d):
    t = d[:, THICK]
    t25, t50, t75 = np.percentile(t, [25, 50, 75])
    return [t.min(), t25, t50, t75, t.max() + 0.1]


def t_mask(d, edges, i):
    return (d[:, THICK] >= edges[i]) & (d[:, THICK] < edges[i + 1])


def save(fig, path):
    fig.savefig(path)
    plt.close(fig)
    print(f"  {path}")


# ============================================================
# HISTOGRAM: 2x2 subplot (overall + 3 bins)
# ============================================================
def plot_hist_grouped(meta, values, title, xlabel, path, bin_type='height', thickness_edges=None):
    if bin_type == 'height':
        labels = ['Overall'] + H_LABELS
        masks = [np.ones(len(meta), bool)] + [h_mask(meta, i) for i in range(3)]
        colors = ['#2196F3', '#2196F3', '#FF9800', '#4CAF50']
    else:
        edges = thickness_edges
        labels = ['Overall'] + [f'{edges[i]:.0f}–{edges[i+1]:.0f}m' for i in range(len(edges) - 1)]
        masks = [np.ones(len(meta), bool)]
        for i in range(len(edges) - 1):
            masks.append(t_mask(meta, edges, i))
        colors = ['#673AB7'] * len(labels)

    n_panels = len(labels)
    cols = min(n_panels, 4)
    rows = (n_panels + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    axes = np.atleast_1d(axes).flatten()

    all_valid = values[np.isfinite(values)]
    if len(all_valid) == 0:
        plt.close(fig)
        return
    xlim = np.percentile(np.abs(all_valid), 99.5)

    for i, (label, mask, color) in enumerate(zip(labels, masks, colors)):
        ax = axes[i]
        v = values[mask]
        v = v[np.isfinite(v)]
        if len(v) == 0:
            ax.set_title(label, fontweight='bold')
            continue
        ax.hist(v, bins=80, color=color, alpha=0.85, edgecolor='white', linewidth=0.3, log=True)
        ax.axvline(0, color='black', linestyle='--', linewidth=0.8)
        ax.axvline(v.mean(), color='red', linestyle='--', linewidth=1.5)
        ax.text(0.97, 0.95,
                f'mean={v.mean():.3f}\nstd={v.std():.3f}\nn={len(v):,}',
                transform=ax.transAxes, ha='right', va='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax.set_xlim(-xlim, xlim)
        ax.set_title(label, fontweight='bold')
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Count (log)')

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(title, fontsize=15, fontweight='bold')
    fig.tight_layout()
    save(fig, path)


# ============================================================
# PROFILE: errorbar line plot
# ============================================================
def plot_profile(meta, values, title, ylabel, path, x_col=FZ, x_label='Height z (m)', bin_step=3.0):
    x = meta[:, x_col]
    x_bins = np.arange(x.min(), x.max() + bin_step, bin_step)
    x_mid = 0.5 * (x_bins[:-1] + x_bins[1:])
    means, stds, counts = [], [], []
    for i in range(len(x_bins) - 1):
        m = (x >= x_bins[i]) & (x < x_bins[i + 1])
        v = values[m]
        v = v[np.isfinite(v)]
        if len(v) > 5:
            means.append(v.mean())
            stds.append(v.std())
            counts.append(len(v))
        else:
            means.append(np.nan)
            stds.append(np.nan)
            counts.append(0)

    means, stds = np.array(means), np.array(stds)
    v = ~np.isnan(means)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.errorbar(x_mid[v], means[v], yerr=stds[v], marker='o', capsize=3,
                color='#673AB7', linewidth=1.5, markersize=4)
    ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
    ax.set_xlabel(x_label)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=14, fontweight='bold')
    fig.tight_layout()
    save(fig, path)


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Path to tensor_data.npz')
    parser.add_argument('--out_dir', default='./plots', help='Root output directory')
    args = parser.parse_args()

    d = np.load(args.input)['data']
    print(f"Loaded tensor: {d.shape}")

    # Create subdirectories
    dir_hh = ensure_dir(os.path.join(args.out_dir, 'hist_by_height'))
    dir_ht = ensure_dir(os.path.join(args.out_dir, 'hist_by_thickness'))
    dir_ov = ensure_dir(os.path.join(args.out_dir, 'overlay'))
    dir_sc = ensure_dir(os.path.join(args.out_dir, 'scatter'))
    dir_pz = ensure_dir(os.path.join(args.out_dir, 'profile_by_z'))
    dir_pt = ensure_dir(os.path.join(args.out_dir, 'profile_by_thick'))

    # Thickness bin edges (computed once, used everywhere)
    te = t_edges(d)
    print(f"Thickness bin edges: {[f'{e:.1f}' for e in te]}")

    # ============================================================
    # DERIVED QUANTITIES
    # ============================================================

    # (a-b) shell minus surface, per point, windward side
    ab_05_w = d[:, A05_W] - d[:, B_W]
    ab_2_w  = d[:, A2_W]  - d[:, B_W]
    ab_3_w  = d[:, A3_W]  - d[:, B_W]

    # (a-b) leeward side
    ab_05_l = d[:, A05_L] - d[:, B_L]
    ab_2_l  = d[:, A2_L]  - d[:, B_L]
    ab_3_l  = d[:, A3_L]  - d[:, B_L]

    # (a-b) combined (windward + leeward stacked)
    ab_05 = np.concatenate([ab_05_w, ab_05_l])
    ab_2  = np.concatenate([ab_2_w,  ab_2_l])
    ab_3  = np.concatenate([ab_3_w,  ab_3_l])
    d2 = np.vstack([d, d])  # duplicated metadata for combined

    # (b,b) surface windward minus leeward
    bb = d[:, B_W] - d[:, B_L]

    # (a,a) shell windward minus leeward
    aa_05 = d[:, A05_W] - d[:, A05_L]
    aa_2  = d[:, A2_W]  - d[:, A2_L]
    aa_3  = d[:, A3_W]  - d[:, A3_L]

    # (a,b) skewed = (a-b)/b per point, combined
    b_combined = np.concatenate([d[:, B_W], d[:, B_L]])
    ab_skew_05 = ab_05 / (np.abs(b_combined) + EPS)
    ab_skew_2  = ab_2  / (np.abs(b_combined) + EPS)
    ab_skew_3  = ab_3  / (np.abs(b_combined) + EPS)

    # ============================================================
    # 1. HISTOGRAMS BY HEIGHT (10 plots)
    # ============================================================
    print("\n=== Histograms by height ===")

    for tag, vals, dd in [('0.5m', ab_05, d2), ('2.0m', ab_2, d2), ('3.0m', ab_3, d2)]:
        plot_hist_grouped(dd, vals,
                          f'(a-b) shell − surface at {tag}', 'Δp (shell − surface)',
                          os.path.join(dir_hh, f'ab_{tag}.png'))

    for tag, vals in [('0.5m', aa_05), ('2.0m', aa_2), ('3.0m', aa_3)]:
        plot_hist_grouped(d, vals,
                          f'(a,a) shell W−L at {tag}', 'Δp (windward − leeward)',
                          os.path.join(dir_hh, f'aa_{tag}.png'))

    plot_hist_grouped(d, bb,
                      '(b,b) surface W−L', 'Δp (windward − leeward)',
                      os.path.join(dir_hh, 'bb.png'))

    for tag, vals, dd in [('0.5m', ab_skew_05, d2), ('2.0m', ab_skew_2, d2), ('3.0m', ab_skew_3, d2)]:
        plot_hist_grouped(dd, vals,
                          f'(a,b) skewed at {tag}', '(shell − surface) / surface',
                          os.path.join(dir_hh, f'ab_skew_{tag}.png'))

    # ============================================================
    # 2. HISTOGRAMS BY THICKNESS (4 plots)
    # ============================================================
    print("\n=== Histograms by thickness ===")

    for tag, vals in [('0.5m', aa_05), ('2.0m', aa_2), ('3.0m', aa_3)]:
        plot_hist_grouped(d, vals,
                          f'(a,a) shell W−L at {tag}', 'Δp (windward − leeward)',
                          os.path.join(dir_ht, f'aa_{tag}.png'), bin_type='thickness', thickness_edges=te)

    plot_hist_grouped(d, bb,
                      '(b,b) surface W−L', 'Δp (windward − leeward)',
                      os.path.join(dir_ht, 'bb.png'), bin_type='thickness', thickness_edges=te)

    # ============================================================
    # 3. OVERLAY PLOTS (3 plots)
    # ============================================================
    print("\n=== Overlay plots ===")

    # (a-b) overlay
    fig, ax = plt.subplots(figsize=(10, 6))
    for tag, vals, c in [('0.5m', ab_05, COLORS[0]), ('2.0m', ab_2, COLORS[1]), ('3.0m', ab_3, COLORS[2])]:
        ax.hist(vals, bins=100, alpha=0.5, color=c,
                label=f'{tag} (mean={vals.mean():.3f})', log=True, edgecolor='white', linewidth=0.3)
    ax.axvline(0, color='black', linestyle='--')
    ax.set_xlabel('Δp (shell − surface)')
    ax.set_ylabel('Count (log)')
    ax.set_title('(a-b) overlay: 0.5m vs 2.0m vs 3.0m', fontweight='bold')
    ax.legend()
    fig.tight_layout()
    save(fig, os.path.join(dir_ov, 'ab_overlay.png'))

    # (a,a) + (b,b) overlay
    fig, ax = plt.subplots(figsize=(10, 6))
    for tag, vals, c in [('0.5m', aa_05, COLORS[0]), ('2.0m', aa_2, COLORS[1]), ('3.0m', aa_3, COLORS[2])]:
        ax.hist(vals, bins=100, alpha=0.5, color=c,
                label=f'shell {tag} (mean={vals.mean():.3f})', log=True, edgecolor='white', linewidth=0.3)
    ax.hist(bb, bins=100, alpha=0.4, color=COLORS[3],
            label=f'surface (mean={bb.mean():.3f})', log=True, edgecolor='white', linewidth=0.3)
    ax.axvline(0, color='black', linestyle='--')
    ax.set_xlabel('Δp (windward − leeward)')
    ax.set_ylabel('Count (log)')
    ax.set_title('(a,a) vs (b,b): shell W−L vs surface W−L', fontweight='bold')
    ax.legend()
    fig.tight_layout()
    save(fig, os.path.join(dir_ov, 'aa_bb_overlay.png'))

    # (a,b) skewed overlay
    fig, ax = plt.subplots(figsize=(10, 6))
    for tag, vals, c in [('0.5m', ab_skew_05, COLORS[0]), ('2.0m', ab_skew_2, COLORS[1]), ('3.0m', ab_skew_3, COLORS[2])]:
        clip = np.percentile(np.abs(vals[np.isfinite(vals)]), 99)
        v = vals[(np.abs(vals) < clip) & np.isfinite(vals)]
        ax.hist(v, bins=100, alpha=0.5, color=c,
                label=f'{tag} (mean={v.mean():.3f})', log=True, edgecolor='white', linewidth=0.3)
    ax.axvline(0, color='black', linestyle='--')
    ax.set_xlabel('(shell − surface) / surface')
    ax.set_ylabel('Count (log)')
    ax.set_title('(a,b) skewed overlay: normalized difference', fontweight='bold')
    ax.legend()
    fig.tight_layout()
    save(fig, os.path.join(dir_ov, 'ab_skew_overlay.png'))

    # ============================================================
    # 4. SCATTER PLOTS (3 plots)
    # ============================================================
    print("\n=== Scatter plots ===")

    for tag, aa_vals, c in [('0.5m', aa_05, COLORS[0]), ('2.0m', aa_2, COLORS[1]), ('3.0m', aa_3, COLORS[2])]:
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.scatter(bb, aa_vals, s=2, alpha=0.15, color=c)
        lim = max(np.percentile(np.abs(bb), 99.5), np.percentile(np.abs(aa_vals), 99.5))
        ax.plot([-lim, lim], [-lim, lim], 'k--', linewidth=1, label='y = x')
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel('Surface ΔP: (b,b)')
        ax.set_ylabel(f'Shell ΔP: (a,a) at {tag}')
        ax.set_title(f'Surface vs shell W−L ({tag})', fontweight='bold')
        ax.set_aspect('equal')
        ax.legend()
        fig.tight_layout()
        save(fig, os.path.join(dir_sc, f'bb_vs_aa_{tag}.png'))

    # ============================================================
    # 5. PROFILE PLOTS BY Z (7 plots)
    # ============================================================
    print("\n=== Profile plots by z ===")

    for tag, vals in [('0.5m', ab_05_w), ('2.0m', ab_2_w), ('3.0m', ab_3_w)]:
        plot_profile(d, vals,
                     f'(a-b) windward vs height ({tag})', 'Mean Δp (shell − surface)',
                     os.path.join(dir_pz, f'ab_{tag}.png'))

    for tag, vals in [('0.5m', aa_05), ('2.0m', aa_2), ('3.0m', aa_3)]:
        plot_profile(d, vals,
                     f'(a,a) shell W−L vs height ({tag})', 'Mean Δp (windward − leeward)',
                     os.path.join(dir_pz, f'aa_{tag}.png'))

    plot_profile(d, bb,
                 '(b,b) surface W−L vs height', 'Mean Δp (windward − leeward)',
                 os.path.join(dir_pz, 'bb.png'))

    # ============================================================
    # 6. PROFILE PLOTS BY THICKNESS (4 plots)
    # ============================================================
    print("\n=== Profile plots by thickness ===")

    for tag, vals in [('0.5m', aa_05), ('2.0m', aa_2), ('3.0m', aa_3)]:
        plot_profile(d, vals,
                     f'(a,a) shell W−L vs thickness ({tag})', 'Mean Δp (windward − leeward)',
                     os.path.join(dir_pt, f'aa_{tag}.png'),
                     x_col=THICK, x_label='Building thickness (m)', bin_step=2.0)

    plot_profile(d, bb,
                 '(b,b) surface W−L vs thickness', 'Mean Δp (windward − leeward)',
                 os.path.join(dir_pt, 'bb.png'),
                 x_col=THICK, x_label='Building thickness (m)', bin_step=2.0)

    # ============================================================
    # SUMMARY
    # ============================================================
    print(f"\n{'='*50}")
    print(f"SUMMARY")
    print(f"{'='*50}")
    print(f"Total samples:    {len(d):,}")
    print(f"z_max range:      {d[:,ZMAX].min():.1f} – {d[:,ZMAX].max():.1f} m")
    print(f"Thickness range:  {d[:,THICK].min():.1f} – {d[:,THICK].max():.1f} m")
    print(f"Height bins:      {H_LABELS}")
    print(f"Thickness edges:  {[f'{e:.1f}' for e in te]}")

    print(f"\n{'Metric':<30} {'mean':>8} {'std':>8} {'n':>10}")
    print('-' * 58)
    for label, vals in [
        ('(b,b) surface W-L', bb),
        ('(a,a) 0.5m W-L', aa_05),
        ('(a,a) 2.0m W-L', aa_2),
        ('(a,a) 3.0m W-L', aa_3),
        ('(a-b) 0.5m', ab_05),
        ('(a-b) 2.0m', ab_2),
        ('(a-b) 3.0m', ab_3),
        ('(a,b) skew 0.5m', ab_skew_05),
        ('(a,b) skew 2.0m', ab_skew_2),
        ('(a,b) skew 3.0m', ab_skew_3),
    ]:
        v = vals[np.isfinite(vals)]
        print(f"{label:<30} {v.mean():>8.4f} {v.std():>8.4f} {len(v):>10,}")

    total_plots = 10 + 4 + 3 + 3 + 7 + 4
    print(f"\nTotal plots generated: {total_plots}")
    print(f"All saved under: {args.out_dir}/")


if __name__ == '__main__':
    main()