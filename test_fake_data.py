import json

import torch
from tqdm import tqdm
import numpy as np
from datetime import datetime
import open3d as o3d

from datasets.Co2dDS import CO2DataLoader, normalization, farthest_point_sample_np
from models.CO2Net import CO2Net, CO2NetLoss

BATCH_SIZE = 16

if __name__ == "__main__":
    data = CO2DataLoader("FakeData/FK2-1", prefix="fk2-1", load_limit=[100, 200])
    loader = torch.utils.data.DataLoader(data, batch_size=BATCH_SIZE, num_workers=10)

    model = CO2Net().cpu().eval()

    state = torch.load("log2/model-20221004014452-81.pth")
    model.load_state_dict(state["model_state"])

    difs_model = []
    difs_train = []

    pathes = [f'FakeData/FK2-1/fk2-1-{i}.txt' for i in range(100, 200)]

    with torch.no_grad():

        for p in pathes:

            with open(p) as f:
                d = json.load(f)
            t = normalization(np.array(d["train"][0]))
            t_idx = farthest_point_sample_np(t[:, :3], 1024)
            inp = t[t_idx]
            inp = np.transpose(inp, (1, 0))
            inp = torch.Tensor([inp])
            r = model(inp)[0].cpu().numpy() * 800 + 500

            tr_d = normalization(np.array(d["true"]))[:, 3].flatten()[t_idx] * 800 + 500

            dif = np.mean(np.power(tr_d - r.flatten(), 2))
            t_dif = np.mean(np.power(tr_d - (inp.numpy()[:, 3].flatten() * 800 + 500) , 2))
            difs_model.append(dif)
            difs_train.append(t_dif)

        m = np.mean(np.array(difs_model))
        t = np.mean(np.array(difs_train))

        print('prediction error ', m)
        print('baseline error', t)





