import torch
from tqdm import tqdm
import numpy as np
from datetime import datetime

from datasets.Co2dDS import CO2DataLoader
from models.CO2Net import CO2Net, CO2NetLoss

BATCH_SIZE = 16
EPOCH = 100
LR = 0.01

if __name__ == "__main__":
    data = CO2DataLoader("../FakeData/FK5", prefix="fk5")
    loader = torch.utils.data.DataLoader(data, batch_size=BATCH_SIZE, num_workers=10)

    for (pnts, targets) in tqdm(loader, total=len(loader), smoothing=0.9):
        print(pnts)

        # pnts = pnts.numpy()
        # targets = targets.numpy()
        # pnts = np.transpose(pnts, (0, 2, 1))
        #
        # pnts = torch.Tensor(pnts)
        # pnts = pnts.float().cuda()
        # targets = torch.Tensor(targets)
        # targets = targets.float().cuda()
        #
        # # if np.any(pnts.cpu().numpy() < 1e-5):
        # print(pnts)



