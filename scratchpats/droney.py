from scipy.spatial import Delaunay
import numpy as np
import open3d as o3d
from collections import defaultdict

from ThreeDLibs.VisLibrary import  add_point

points = (np.random.rand(10, 3) - 0.5) * 10

d = Delaunay(points)

pcd = o3d.geometry.PointCloud()

ls = []
# for l in d.simplices:
#     for i, p1 in enumerate(l):
#             for j in range(i + 1, len(l)):
#                 if p1 == -1 or l[j] == -1:
#                     continue
#                 ls.append([p1, l[j]])

convexes = np.array(d.convex_hull)
idxs = list(set(convexes.flatten()))
hull_v = np.array(d.points[idxs])
g = np.mean(hull_v, axis=0)

norms = {}
# for c in convexes:
#     vs = d.points[c]
#     v1 = vs[1] - vs[0]
#     v2 = vs[2] - vs[0]
#     norm = np.cross(v1, v2)
#     dir_vec = g - np.mean(vs, axis=0)
#     norm *= 1 if np.dot(norm, np.transpose(dir_vec)) < 0 else -1
#
#     norm /= np.linalg.norm(norm)
#
#     for v in c:
#         if v not in norms.keys():
#             norms.setdefault(v, np.zeros(3))
#         norms[v] += norm

# for i in idxs:
#     p = d.points[i]
#     norm = p - g
#     norm /= np.linalg.norm(norm)
#     norms.setdefault(i, norm)

for l in d.simplices[0:1]:
    for i, p1 in enumerate(l):
        for j in range(i + 1, len(l)):
            if p1 == -1 or l[j] == -1:
                continue
            ls.append([p1, l[j]])

points = points.tolist()
amount = 20.0
for n, v in norms.items():
    points.append((v / np.linalg.norm(v)) * amount)
    ls.append([n, len(points) - 1])

points = np.array(points)

pcd.points = o3d.utility.Vector3dVector(points)
add_point(pcd, g)
lineset = o3d.geometry.LineSet(
    points=o3d.utility.Vector3dVector(points),
    lines=o3d.utility.Vector2iVector(ls)
)

o3d.visualization.draw_geometries([pcd, lineset])
