import open3d as o3d
import numpy as np
import random

from ThreeDLibs.PcdLoader import load_pcd
from randomDataGen.RandomDataGen import load_sensor_pos
from ThreeDLibs.VisLibrary import add_point

pcd, trans = load_pcd("datas/bus.ply")
sns_pos = load_sensor_pos("datas/bus2-pos.json")
pcd = pcd.voxel_down_sample(0.1)

for n, pos in sns_pos.items():
    r_color = np.random.rand(3)
    add_point(pcd, np.array(pos), color=r_color)
    print(n, ":", r_color)

o3d.visualization.draw_geometries([pcd])




