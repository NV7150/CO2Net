import open3d as o3d
import numpy as np
import torch
from models.pointnet2_utils import query_ball_point
from ThreeDLibs.PcdLoader import load_pcd
from ThreeDLibs.VisLibrary import add_point
from datasets.Co2dDS import farthest_point_sample

pcd, t = load_pcd("../datas/bus.ply")

points = np.array(pcd.points)

fps_idx = farthest_point_sample(points, 1024)

points = points[fps_idx]

fps_idx_2 = farthest_point_sample(points, 128)
fps_points_2 = points[fps_idx_2]

# q = query_ball_point(1, 32, points, fps_points_2)

# fps_points_3 = points[0, q[0]].cpu().numpy()[0]

# print(fps_points_3)

fps_pcd = o3d.geometry.PointCloud()
fps_pcd.points = o3d.utility.Vector3dVector(points)
# fps_pcd.colors = o3d.utility.Vector3dVector(np.zeros(points[0].shape))
# for p in fps_points_3:
#     print(p)
#     add_point(fps_pcd, p)
o3d.visualization.draw_geometries([fps_pcd])
