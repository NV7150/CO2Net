import torch.nn as nn
import torch.nn.functional as F
import torch
from models.pointnet2_utils import PointNetSetAbstraction, PointNetFeaturePropagation

# channels = {co2, distance}
CHANNELS = 2


class CO2Net(nn.Module):
    def __init__(self):
        super(CO2Net, self).__init__()
        # self.sa1 = PointNetSetAbstraction(1024, 0.3, 32, 3 + CHANNELS, [32, 32, 64], False)
        # self.sa2 = PointNetSetAbstraction(256, 0.6, 32, 64, [64, 64, 128], False)
        # self.sa3 = PointNetSetAbstraction(128, 8, 32, 128, [128, 128, 256], False)
        # self.sa4 = PointNetSetAbstraction(64, 10, 32, 256, [256, 256, 512], False)
        self.sa1 = PointNetSetAbstraction(1024, 0.1, 32, 3 + CHANNELS, [32, 32, 64], False)
        self.sa2 = PointNetSetAbstraction(256, 0.2, 32, 64, [64, 64, 128], False)
        self.sa3 = PointNetSetAbstraction(64, 0.4, 32, 128, [128, 128, 256], False)
        self.sa4 = PointNetSetAbstraction(16, 0.8, 32, 256, [256, 256, 512], False)
        self.fp4 = PointNetFeaturePropagation(768, [256, 256])
        self.fp3 = PointNetFeaturePropagation(384, [256, 256])
        self.fp2 = PointNetFeaturePropagation(320, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])

        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, 1, 1)

    def forward(self, xyz):
        l0_points = xyz
        l0_xyz = xyz[:, :3, :]

        # print("data", l0_points)
        # print(l0_xyz)

        # l0_points = normalization(l0_points)

        # print("layer1")
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        assert not torch.isnan(l1_points).any()

        # print("layer2")
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        assert not torch.isnan(l2_points).any()

        # print("layer3")
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        assert not torch.isnan(l2_points).any()

        # print("layer4")
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)
        assert not torch.isnan(l2_points).any()

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        assert not torch.isnan(l3_points).any()

        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        assert not torch.isnan(l2_points).any()

        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        assert not torch.isnan(l1_points).any()

        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)
        assert not torch.isnan(l0_points).any()

        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        assert not torch.isnan(x).any()
        x = self.conv2(x)
        assert not torch.isnan(x).any()
        x = x.permute(0, 2, 1)
        return x, l4_points


class CO2NetLoss(nn.Module):
    def __init__(self):
        super(CO2NetLoss, self).__init__()
        self.loss = nn.MSELoss()

    def forward(self, pred, target):
        return self.loss(pred, target)



