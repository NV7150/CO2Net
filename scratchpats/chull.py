import open3d as o3d

import numpy as np
from scipy.spatial import ConvexHull

vs = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1]), np.array([0, 0,  0])]
h = ConvexHull(vs)

ps = np.random.random((1000, 3))

colors = []
for p in ps:
    if np.array_equal(h.vertices, ConvexHull(np.concatenate([h.points, [p]], axis=0)).vertices):
        colors.append([1, 0, 0])
    else:
        colors.append([0, 0, 1])

lines = []
for i, v1 in enumerate(vs):
    for j in range(i + 1, len(vs)):
        lines.append((i, j))

lineset = o3d.geometry.LineSet(
    points=o3d.utility.Vector3dVector(vs),
    lines=o3d.utility.Vector2iVector(lines)
)


pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(ps)
pcd.colors = o3d.utility.Vector3dVector(colors)

o3d.visualization.draw_geometries([pcd, lineset])

