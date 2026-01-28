import os

import numpy as np
import open3d as o3d
import torch

directory = '/Users/cedric/TRASE/sadg_example_models/sear_steak/point_cloud/iteration_30000/'
pcd_f = f'{directory}/point_cloud.ply'
pcd = o3d.io.read_point_cloud(pcd_f)
print('loaded pcd')
cluster_data = torch.load(f'{directory}/clusters.pt', weights_only=False, map_location='cpu')

rgb = cluster_data["rgb"]
if torch.is_tensor(rgb):
    rgb = rgb.detach().cpu().numpy()
rgb = np.asarray(rgb)

pts_n = np.asarray(pcd.points).shape[0]
if rgb.ndim != 2 or rgb.shape[1] != 3:
    raise ValueError(f"RGB shape {rgb.shape} is not (N,3)")
if rgb.shape[0] != pts_n:
    raise ValueError(f"RGB rows {rgb.shape[0]} != point count {pts_n}")

# Open3D expects float colors in 0..1.
rgb = rgb.astype(np.float64, copy=False)
if rgb.size and float(rgb.max()) > 1.0:
    rgb = rgb / 255.0
rgb = np.clip(rgb, 0.0, 1.0)

pcd.colors = o3d.utility.Vector3dVector(rgb)
print(f"pcd.colors stats: min={rgb.min(axis=0)} max={rgb.max(axis=0)}")

o3d.visualization.draw_geometries([pcd])
