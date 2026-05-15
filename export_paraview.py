# Save as: C:\eda_v2\eda_output_tasks\export_paraview.py
# Run: python export_paraview.py

import numpy as np
import torch
import pyvista as pv
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN
from pathlib import Path

# Pick one PT file to visualize
PT_FILE = r"C:\eda_v2\eda_output_tasks\case_HDB_0238b037_E.pt"
OUT_DIR = Path(r"C:\eda_v2\eda_output_tasks\paraview")
OUT_DIR.mkdir(exist_ok=True)

G = 9.81
FLOOR_STEP = 3.0
X_BINS = 10
OFFSETS = [0.5, 2.0, 3.0]
IDW_K = 4
EPS = 1e-10
SDF_MIN = 0.1

data = torch.load(PT_FILE, weights_only=False, map_location='cpu')
s_pos = data['surface_pos'].numpy().astype(np.float64)
s_norm = data['surface_normals'].numpy().astype(np.float64)
s_fields = data['surface_fields'].numpy().astype(np.float64)
v_pos = data['volume_pos'].numpy().astype(np.float64)
v_fields = data['volume_fields'].numpy().astype(np.float64)
v_sdf = data['volume_sdf'].numpy().astype(np.float64)
u_ref = float(data['global_params'][0])
case_name = data.get('case_name', 'case')

p_surf = s_fields[:, 0]
p_vol = v_fields[:, 3]
p_surf_aero = p_surf + G * s_pos[:, 2] / (u_ref ** 2)

wall = np.abs(s_norm[:, 2]) < 0.5
w_pos = s_pos[wall]
w_norm = s_norm[wall]
w_p = p_surf_aero[wall]

clusters = DBSCAN(eps=5.0, min_samples=10).fit_predict(w_pos[:, :2])
vol_tree = cKDTree(v_pos)

# Collect all points with their delta_p
all_surf_pts = []
all_shell_pts = {o: [] for o in OFFSETS}
all_delta_p = {o: [] for o in OFFSETS}
all_labels = []  # 'normal' or 'outlier'

for bldg_id in np.unique(clusters):
    if bldg_id == -1:
        continue
    bm = clusters == bldg_id
    b_pos = w_pos[bm]; b_norm = w_norm[bm]; b_p = w_p[bm]
    b_ny = b_norm[:, 1]
    wind = b_ny < -0.5; lee = b_ny > 0.5
    if wind.sum() < 5 or lee.sum() < 5:
        continue

    for side, mask in [('wind', wind), ('lee', lee)]:
        s_p = b_pos[mask]; s_n = b_norm[mask]; s_pr = b_p[mask]
        z_min, z_max = s_p[:, 2].min(), s_p[:, 2].max()
        floor_zs = np.arange(z_min + FLOOR_STEP/2, z_max, FLOOR_STEP)
        if len(floor_zs) == 0:
            floor_zs = np.array([(z_min + z_max) / 2])

        x_lo, x_hi = s_p[:, 0].min(), s_p[:, 0].max()
        if x_hi - x_lo < 2:
            continue
        x_samples = np.linspace(x_lo, x_hi, X_BINS + 2)[1:-1]

        xz_tree = cKDTree(s_p[:, [0, 2]])

        for fz in floor_zs:
            for xs in x_samples:
                d_q, idx_q = xz_tree.query([xs, fz], k=1)
                if d_q > 5.0:
                    continue

                pt = s_p[idx_q]
                nm = s_n[idx_q]
                p_surface = float(s_pr[idx_q])
                all_surf_pts.append(pt)

                for o in OFFSETS:
                    shell_pt = pt + nm * o
                    d_sh, idx_sh = vol_tree.query(shell_pt.reshape(1, 3), k=IDW_K)
                    d_sh = np.maximum(d_sh, EPS)

                    # SDF check
                    neighbor_sdf = v_sdf[idx_sh.flatten()]
                    if np.any(neighbor_sdf < SDF_MIN):
                        all_shell_pts[o].append(shell_pt)
                        all_delta_p[o].append(np.nan)
                        continue

                    w = 1.0 / d_sh
                    w /= w.sum()
                    p_sh_raw = float((p_vol[idx_sh.flatten()] * w.flatten()).sum())
                    p_sh_aero = p_sh_raw + G * shell_pt[2] / (u_ref ** 2)
                    dp = p_sh_aero - p_surface

                    all_shell_pts[o].append(shell_pt)
                    all_delta_p[o].append(dp)

# Export VTP files
surf_pts = np.array(all_surf_pts)
surf_pv = pv.PolyData(surf_pts)
surf_pv.save(str(OUT_DIR / f'{case_name}_surface_samples.vtp'))
print(f"Saved surface samples: {len(surf_pts)} points")

for o in OFFSETS:
    tag = f"{o}m"
    pts = np.array(all_shell_pts[o])
    dp = np.array(all_delta_p[o])

    shell_pv = pv.PolyData(pts)
    shell_pv['delta_p'] = dp
    shell_pv['abs_delta_p'] = np.abs(dp)
    shell_pv['is_outlier'] = (np.abs(dp) > 2.0).astype(float)

    shell_pv.save(str(OUT_DIR / f'{case_name}_shell_{tag}.vtp'))
    
    valid = ~np.isnan(dp)
    print(f"Shell {tag}: {len(pts)} points, {valid.sum()} valid, "
          f"mean dp={dp[valid].mean():.3f}, outliers (|dp|>2): {(np.abs(dp[valid])>2).sum()}")

# Also export building STL for reference
stl_pts = data['stl_vertices'].numpy()
stl_faces_raw = data['stl_faces'].numpy().astype(int)
n_faces = len(stl_faces_raw) // 3
faces = np.column_stack([np.full(n_faces, 3), stl_faces_raw.reshape(-1, 3)])
building = pv.PolyData(stl_pts, faces.flatten())
building.save(str(OUT_DIR / f'{case_name}_building.vtp'))
print(f"Saved building mesh")

print(f"\nOpen in ParaView: {OUT_DIR}")
print("Color shell by 'delta_p' or 'is_outlier'")