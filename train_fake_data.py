import glob

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
    data = CO2DataLoader("FakeData/FK2-1", prefix="fk2-1", num_point=1024)
    loader = torch.utils.data.DataLoader(data, batch_size=BATCH_SIZE, num_workers=10)

    model = CO2Net().cuda()
    model.train()
    model_loss = CO2NetLoss().cuda()

    def bn_momentum_adjust(m, momentum):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)

    # checkpoint = torch.load("log/model-20220922005936-11.pth")
    # start_epoch = checkpoint['epoch']
    # model.load_state_dict(checkpoint['model_state'])
    start_epoch=0

    model.apply(weights_init)

    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = 10

    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)

    torch.autograd.set_detect_anomaly(True)

    for epoch in range(start_epoch, EPOCH):
        print(f"epoch:{epoch}")

        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
        if momentum < 0.01:
            momentum = 0.01
        print('BN momentum updated to: %f' % momentum)
        model = model.apply(lambda x: bn_momentum_adjust(x, momentum))

        for (pnts, targets) in tqdm(loader, total=len(loader), smoothing=0.9):
            # print("data", pnts)
            optimizer.zero_grad()

            pnts = pnts.numpy()
            targets = targets.numpy()
            pnts = np.transpose(pnts, (0, 2, 1))

            pnts = torch.Tensor(pnts)
            pnts = pnts.float().cuda()
            targets = torch.Tensor(targets)
            targets = targets.float().cuda()

            pnts_np = pnts.cpu().numpy()

            # print(torch.isnan(pnts).cpu().numpy().any())

            # print(pnts)
            # assert not (1e-5 > pnts).cpu().numpy().any()

            pred, feat = model(pnts)
            # print(torch.isnan(pred).cpu().numpy().any())

            targets = targets[:, :, 3:4]

            loss = model_loss(pred, targets)
            # print(torch.isnan(pred).cpu().numpy().any())
            # print(torch.isnan(loss))
            loss.backward()
            optimizer.step()

            # for p in model.parameters():
            #     print(p)
            torch.cuda.empty_cache()

        state = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'opt': optimizer.state_dict()
        }
        torch.save(state, f"log2/model-{datetime.now().strftime('%Y%m%d%H%M%S')}-{epoch}.pth")




