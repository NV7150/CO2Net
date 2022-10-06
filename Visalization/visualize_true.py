import json

import torch
from tqdm import tqdm
import numpy as np
from datetime import datetime
import open3d as o3d

from datasets.Co2dDS import CO2DataLoader, normalization, farthest_point_sample_np
from models.CO2Net import CO2Net, CO2NetLoss

BATCH_SIZE = 16

if __name__ == "__main__":
    with open("../FakeData/FK2-1/fk2-1-110.txt", encoding="utf-8") as f:
        d = json.load(f)

    accs = normalization(np.array(d["true"]))
    acc_idx = farthest_point_sample_np(accs[:, :3], 1024)
    accs = accs[acc_idx]
    coords_d = np.array(accs)

    results = coords_d[:, 3:4] * 800 + 500

    coords = coords_d[:, :3]
    coords /= np.max(coords)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords)
    min_v = 500
    max_v = 1000
    colors = (((results - min_v) / (max_v - min_v)) - 0.5) / 0.5 * np.array([1.0, 0.0, -1.0])

    pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([pcd])






