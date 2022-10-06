import open3d as o3d
import json
import numpy as np



with open("FakeData/FK7/fk7-0.txt", encoding="utf-8") as f:
    d = json.load(f)

np_d = np.array(d["train"][1])

points = np_d[:, :3]
co2 = np_d[:, 3:4].reshape(len(points), 1)
min_v = 500
max_v = 1000
colors = (((co2 - min_v) / (max_v - min_v)) - 0.5) / 0.5 * np.array([1.0, 0.0, -1.0])

pcd = o3d.geometry.PointCloud()
pcd.colors = o3d.utility.Vector3dVector(colors)
pcd.points = o3d.utility.Vector3dVector(points)

c = np.var(np_d)

o3d.visualization.draw_geometries([pcd])
