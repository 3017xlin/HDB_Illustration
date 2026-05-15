"""
Extract 8-point pressure tensor from PT files.
Output: (N_total, 12) array saved as .npz
"""
import argparse, os, time, sys
import numpy as np
import torch
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN
from multiprocessing import Pool

G = 9.81
U_REF = 2.0
FLOOR_GAP = 3.0
X_BINS = 10
OFFSETS = [0.5, 2.0, 3.0]
IDW_K = 4
EPS = 1e-10

def idw_batch(tree, values, targets, k=IDW_K):
    d, idx = tree.query(targets, k=k, workers=1)
    if k == 1:
        d, idx = d[:, None], idx[:, None]
    d = np.maximum(d, EPS)
    w = 1.0 / d
    w /= w.sum(axis=1, keepdims=True)
    return (values[idx] * w).sum(axis=1)

def detect_floors(z_vals):
    zs = np.sort(np.unique(np.round(z_vals, 1)))
    if len(zs) < 2:
        return [(zs[0], zs[-1])]
    d = np.diff(zs)
    breaks = np.where(d > FLOOR_GAP)[0]
    floors = []
    start = zs[0]
    for b in breaks:
        floors.append((start, zs[b]))
        start = zs[b + 1]
    floors.append((start, zs[-1]))
    return floors

def process_case(pt_path):
    try:
        data = torch.load(pt_path, weights_only=False, map_location='cpu')
        s_pos = data['surface_pos'].numpy().astype(np.float64)
        s_norm = data['surface_normals'].numpy().astype(np.float64)
        s_fields = data['surface_fields'].numpy().astype(np.float64)
        v_pos = data['volume_pos'].numpy().astype(np.float64)
        v_fields = data['volume_fields'].numpy().astype(np.float64)

        p_surf = s_fields[:, 0]
        p_vol = v_fields[:, 3]

        # Detrend
        p_surf_aero = p_surf + G * s_pos[:, 2] / (U_REF ** 2)
        p_vol_aero = p_vol + G * v_pos[:, 2] / (U_REF ** 2)

        # Wall mask
        wall = np.abs(s_norm[:, 2]) < 0.5
        if wall.sum() < 50:
            return []

        w_pos = s_pos[wall]
        w_norm = s_norm[wall]
        w_p = p_surf_aero[wall]
        w_ny = w_norm[:, 1]

        # DBSCAN clustering
        clusters = DBSCAN(eps=5.0, min_samples=10).fit_predict(w_pos[:, :2])

        # Volume KDTree (once per case)
        vol_tree = cKDTree(v_pos)

        rows = []

        for bldg_id in np.unique(clusters):
            if bldg_id == -1:
                continue

            bm = clusters == bldg_id
            b_pos = w_pos[bm]
            b_norm = w_norm[bm]
            b_p = w_p[bm]
            b_ny = b_norm[:, 1]

            wind = b_ny < -0.5
            lee = b_ny > 0.5

            if wind.sum() < 5 or lee.sum() < 5:
                continue

            wi_pos = b_pos[wind]
            wi_norm = b_norm[wind]
            wi_p = b_p[wind]

            le_pos = b_pos[lee]
            le_norm = b_norm[lee]
            le_p = b_p[lee]

            z_max = float(b_pos[:, 2].max())
            thickness = float(abs(wi_pos[:, 1].mean() - le_pos[:, 1].mean()))

            # Overlapping x range
            x_lo = max(wi_pos[:, 0].min(), le_pos[:, 0].min())
            x_hi = min(wi_pos[:, 0].max(), le_pos[:, 0].max())
            if x_hi - x_lo < 2.0:
                continue

            x_samples = np.linspace(x_lo, x_hi, X_BINS + 2)[1:-1]

            # Floor detection
            floors = detect_floors(b_pos[:, 2])
            if not floors:
                continue

            # KDTrees on xz for nearest surface point lookup
            wi_xz = wi_pos[:, [0, 2]]
            le_xz = le_pos[:, [0, 2]]
            wi_xz_tree = cKDTree(wi_xz)
            le_xz_tree = cKDTree(le_xz)

            # Collect all sampling queries
            queries_xz = []
            floor_z_list = []
            for fmin, fmax in floors:
                fz = (fmin + fmax) / 2.0
                for xi, xs in enumerate(x_samples):
                    queries_xz.append([xs, fz])
                    floor_z_list.append(fz)

            if not queries_xz:
                continue

            queries_xz = np.array(queries_xz)
            n_q = len(queries_xz)

            # Batch find nearest windward and leeward surface points
            d_wi, idx_wi = wi_xz_tree.query(queries_xz, k=1)
            d_le, idx_le = le_xz_tree.query(queries_xz, k=1)

            # Filter: skip if nearest is too far (> 5m in xz)
            valid = (d_wi < 5.0) & (d_le < 5.0)

            for q in range(n_q):
                if not valid[q]:
                    continue

                # Surface values (nearest point)
                p_b_wind = float(wi_p[idx_wi[q]])
                p_b_lee = float(le_p[idx_le[q]])

                w_pt = wi_pos[idx_wi[q]]
                w_nrm = wi_norm[idx_wi[q]]
                l_pt = le_pos[idx_le[q]]
                l_nrm = le_norm[idx_le[q]]

                # Shell points
                shells_wind = np.array([w_pt + w_nrm * o for o in OFFSETS])
                shells_lee = np.array([l_pt + l_nrm * o for o in OFFSETS])

                # IDW from volume for shell pressures
                all_shell = np.vstack([shells_wind, shells_lee])  # (6, 3)
                d_sh, idx_sh = vol_tree.query(all_shell, k=IDW_K)
                d_sh = np.maximum(d_sh, EPS)
                wt = 1.0 / d_sh
                wt /= wt.sum(axis=1, keepdims=True)
                p_shell_raw = (p_vol[idx_sh] * wt).sum(axis=1)
                # Detrend shell pressures
                p_shell_aero = p_shell_raw + G * all_shell[:, 2] / (U_REF ** 2)

                # p_shell_aero[0,1,2] = wind shells 0.5, 2, 3
                # p_shell_aero[3,4,5] = lee shells 0.5, 2, 3

                fz = floor_z_list[q]
                x_frac = q % X_BINS

                rows.append([
                    p_b_wind,
                    p_shell_aero[0], p_shell_aero[1], p_shell_aero[2],
                    p_b_lee,
                    p_shell_aero[3], p_shell_aero[4], p_shell_aero[5],
                    fz, z_max, thickness, x_frac
                ])

        return rows

    except Exception as e:
        print(f"ERROR {pt_path}: {e}", file=sys.stderr)
        return []

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pt_dir', required=True)
    parser.add_argument('--out', default='tensor_data.npz')
    parser.add_argument('--workers', type=int, default=60)
    args = parser.parse_args()

    files = sorted([
        os.path.join(args.pt_dir, f)
        for f in os.listdir(args.pt_dir) if f.endswith('.pt')
    ])
    print(f"Found {len(files)} PT files, using {args.workers} workers")

    t0 = time.time()
    all_rows = []

    with Pool(args.workers) as pool:
        for i, rows in enumerate(pool.imap_unordered(process_case, files)):
            all_rows.extend(rows)
            if (i + 1) % 50 == 0 or (i + 1) == len(files):
                elapsed = time.time() - t0
                print(f"  [{i+1}/{len(files)}] {elapsed:.0f}s, {len(all_rows)} samples so far")
                sys.stdout.flush()

    data = np.array(all_rows, dtype=np.float64)
    print(f"\nFinal tensor shape: {data.shape}")
    print(f"Columns: p_b_w, p_a05_w, p_a2_w, p_a3_w, p_b_l, p_a05_l, p_a2_l, p_a3_l, floor_z, z_max, thickness, x_frac")

    # Quick stats
    print(f"\nz_max range: {data[:,9].min():.1f} - {data[:,9].max():.1f} m")
    print(f"thickness range: {data[:,10].min():.1f} - {data[:,10].max():.1f} m")
    print(f"floor_z range: {data[:,8].min():.1f} - {data[:,8].max():.1f} m")

    # Thickness distribution for binning
    t = data[:, 10]
    print(f"\nThickness percentiles: 25%={np.percentile(t,25):.1f}, 50%={np.percentile(t,50):.1f}, 75%={np.percentile(t,75):.1f}")

    np.savez_compressed(args.out, data=data)
    print(f"\nSaved to {args.out} ({os.path.getsize(args.out)/1024/1024:.1f} MB)")
    print(f"Total time: {(time.time()-t0)/60:.1f} min")

if __name__ == '__main__':
    main()