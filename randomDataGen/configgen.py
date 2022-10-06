import json
import numpy as np
from RandomDataGen import load_sensor_pos, generate_random_data

s_pos = load_sensor_pos("../datas/bus2-pos.json")

z_pairs = [(n, l[0]) for n, l in s_pos.items()]
zs = np.array([z for n, z in z_pairs])

z_min = np.min(zs)
z_max = np.max(zs)

up_min = 800
down_min = 500
broad = 300

k = (up_min - down_min) / (z_max - z_min)

config_pairs = {}

for n, z in z_pairs:
    min_v = z * k + down_min
    config_pairs.setdefault(n, [min_v, min_v + broad])

with open("../FakeData/config.2.x.json", mode="w", encoding="utf-8") as f:
    json.dump(config_pairs, f)

