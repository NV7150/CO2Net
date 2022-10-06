import numpy as np
import open3d
import json
import random

from randomDataGen.RandomDataGen import load_sensor_pos
from ThreeDLibs.VoronoiPointCloud import SensorVPcd
from ThreeDLibs.PcdLoader import load_pcd
from ThreeDLibs.VisLibrary import add_point

from randomDataGen.RandomDataGen import generate_random_data

poses = load_sensor_pos("../datas/bus2-pos.json")
pcd, trans = load_pcd("../datas/bus.ply")

vpcd = SensorVPcd(poses, outer_amount=20)

with open("../FakeData/config.2.x.json") as f:
    config = json.load(f)

pcd = pcd.voxel_down_sample(0.2)


v = {}

for (key, range_tuple) in config.items():
    v.setdefault(key, random.uniform(range_tuple[0], range_tuple[1]))


points = np.array(pcd.points)


def get_color(p_v):
    return (((p_v - 500) / 500) - 0.5) * 2 * np.array([1.0, 0, -1.0])


colors = []

for p in points:
    v_res = vpcd.get_weighted(v, p)
    color = get_color(v_res.sum()).tolist()
    colors.append(color)

colors = np.array(colors)
colors[colors < 0] = 0

pcd_d = open3d.geometry.PointCloud()
pcd_d.points = open3d.utility.Vector3dVector(points)
pcd_d.colors = open3d.utility.Vector3dVector(colors)
ls = vpcd.export_sm()

# for pos in poses.values():
#     add_point(pcd_d, pos)

open3d.visualization.draw_geometries([pcd_d])
