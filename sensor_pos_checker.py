import math
import numpy as np

from ThreeDLibs.PcdLoader import load_pcd
from randomDataGen.RandomDataGen import load_sensor_pos
from ThreeDLibs.VisLibrary import add_point

sns_pos = load_sensor_pos("datas/bus2-pos.json")

int_pos = {}

for (n, pos) in sns_pos.items():
    i_pos = pos.astype(int) * 10
    int_pos.setdefault(n, i_pos)
    print(pos[1])

int_pos = dict(sorted(int_pos.items(), key=lambda x: (x[0][1], x[0][0])))
print(int_pos)





