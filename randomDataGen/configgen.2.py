import json
import numpy as np
from RandomDataGen import load_sensor_pos, generate_random_data

s_pos = load_sensor_pos("../datas/bus2-pos.json")

uppper = 3

z_pairs = [(n, l[0]) for n, l in s_pos.items()]

up = list(sorted(z_pairs, key=lambda x: x[1]))

up_r = [900, 600]
down_r = [700, 500]

config_pairs = {}

for n, v in up[:4]:
    config_pairs.setdefault(n, up_r)

for n, v in up[4:]:
    config_pairs.setdefault(n, down_r)

with open("../FakeData/config.3.x.json", mode="w", encoding="utf-8") as f:
    json.dump(config_pairs, f)

