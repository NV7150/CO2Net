import random

import open3d as o3d
import json
import numpy as np
import random
import multiprocessing as mp
from tqdm import tqdm

from ThreeDLibs.PcdLoader import load_pcd

from ThreeDLibs.Util import softmax
from ThreeDLibs.VoronoiPointCloud import SensorVPcd

A = 30


def load_sensor_pos(json_path):
    with open(json_path, encoding="utf-8") as f:
        raw = json.load(f)
    pos_dict = {}
    for key, p in raw.items():
        pos_dict.setdefault(key, np.array(p))
    return pos_dict


# input size: sensor_pos_arr and sensor_value_arr is ndarray
def map_data(pcd, sensor_values, sensor_poses, vpcd):
    ps = np.array(pcd.points)

    out_ps = []
    distances = []
    pos_dicts = np.array([sensor_poses[k] for k in sensor_poses.keys()])

    for p in ps:
        res = vpcd.get_weighted(sensor_values, p)
        out_ps.append(res.sum())
        distances.append(np.min(np.linalg.norm(p - pos_dicts, axis=0)))
    return np.stack([np.array(out_ps), np.array(distances)], axis=1)

def map_data_dist2(pcd, sensor_values, sensor_poses):
    ps = np.array(pcd.points)

    out_ps = []
    distances = []
    sensor_poses_np = np.array([sensor_poses[k] for k in sensor_poses.keys()])
    sensor_vals_np = np.array([sensor_values[k] for k in sensor_poses.keys()])

    vals = []

    for p in ps:
        dists = np.linalg.norm(p - sensor_poses_np, axis=1)
        weights = A / np.power(dists, 2)
        weights = softmax(weights)
        vals.append((weights * sensor_vals_np).sum())
        distances.append(np.min(dists))

    return np.stack([np.array(vals), np.array(distances)], axis=1)


def generate_random_data(pcd, sensor_pos_dict, config_dict, vpcd, train_num=10, sns_num=5, dist_method=False):
    virtual_sensor_values = {}
    for (key, range_tuple) in config_dict.items():
        virtual_sensor_values.setdefault(key, random.uniform(range_tuple[0], range_tuple[1]))

    pos_np = []
    val_np = []
    for key in config_dict.keys():
        pos_np.append(list(sensor_pos_dict[key]))
        val_np.append(virtual_sensor_values[key])

    # pos_np = np.array(pos_np)
    # val_np = np.array(val_np)

    sorted_l = sorted([(n, p) for n, p in list(sensor_pos_dict.items())], key=lambda v: v[1][0])
    top = sorted_l[0][0]
    last = sorted_l[-1][0]
    sample_base = list(virtual_sensor_values.keys())
    sample_base.remove(top)
    sample_base.remove(last)

    masked_datas = []
    for cnt in range(train_num):
        # idx = np.random.choice(np.arange(len(pos_np)), sns_num, replace=False)

        r_keys = random.sample(sample_base, sns_num - 2)
        r_keys.extend([top, last])
        v_d = {}
        p_d = {}
        for key in r_keys:
            v_d.setdefault(key, virtual_sensor_values[key])
            p_d.setdefault(key, sensor_pos_dict[key])

        if not dist_method:
            vpcd_2 = SensorVPcd(p_d)
            masked_data = map_data(pcd, v_d, p_d, vpcd_2).tolist()
        else:
            masked_data = map_data_dist2(pcd, v_d, p_d).tolist()
        masked_datas.append(masked_data)

    true_data \
        = map_data(pcd, virtual_sensor_values, sensor_pos_dict, vpcd) \
        if not dist_method else \
        map_data_dist2(pcd, virtual_sensor_values, sensor_pos_dict)

    return true_data, np.array(masked_datas)


if __name__ == "__main__":
    np.seterr(all='raise')
    pcd, trns = load_pcd("../datas/bus.ply")
    pcd = pcd.voxel_down_sample(0.1)
    sns_pos = load_sensor_pos("../datas/bus2-pos.json")

    with open("../FakeData/config.3.x.json", encoding="utf-8") as f:
        config = json.load(f)

    # q = mp.Queue()

    vpcd = SensorVPcd(sns_pos)

    def task():
        generated_data, maskeds = generate_random_data(pcd, sns_pos, config, vpcd, dist_method=True, sns_num=5)
        ps = np.array(pcd.points)
        output = np.hstack([ps, generated_data])

        ps_ext = np.tile(ps, (maskeds.shape[0], 1, 1))
        train_output = np.concatenate([ps_ext, maskeds], axis=2)
        return output, train_output

    results = []

    for i in tqdm(range(200)):
        true_val, train = task()
        with open(f"../FakeData/FK2-1/fk2-1-{i}.txt", encoding="utf-8", mode="a+") as f:
            f.write(json.dumps({"true": true_val.tolist(), "train": train.tolist()}))

