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

    model = CO2Net().cpu().eval()

    state = torch.load("../log2/model-20221004014452-81.pth")

    with torch.no_grad():
        model.load_state_dict(state["model_state"])

        t = normalization(np.array(d["train"][0]))
        t_idx = farthest_point_sample_np(t[:, :3], 1024)
        inp = t[t_idx]
        inp = np.transpose(inp, (1, 0))
        inp = torch.Tensor([inp])

        r = model(inp)[0].cpu().numpy() * 800 + 500

    coords_d = np.array(t[t_idx]).reshape([-1, 5])
    results = r.reshape(-1, 1)

    coords = coords_d[:, :3]
    coords /= np.max(coords)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords)
    min_v = 500
    max_v = 1000
    colors = (((results - min_v) / (max_v - min_v)) - 0.5) / 0.5 * np.array([1.0, 0.0, -1.0])

    pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([pcd])

    print(r)






