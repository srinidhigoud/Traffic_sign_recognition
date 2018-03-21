import torch
import torch.nn as nn
import torch.nn.functional as F

nclasses = 43 # GTSRB as 43 classes

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 70, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(70, 110, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(110, 180, kernel_size=3, padding=1)
        self.conv1_1 = nn.BatchNorm2d(70)
        self.conv2_1 = nn.BatchNorm2d(110)
        self.conv3_1 = nn.BatchNorm2d(180)
        self.conv1_drop = nn.Dropout2d()
        self.conv2_drop = nn.Dropout2d()
        self.conv3_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(4*4*180, 200)
        self.fbn = nn.BatchNorm1d(200)
        self.fc2 = nn.Linear(200, nclasses)


        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 4 * 4, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.fill_(0)
        self.fc_loc[2].bias.data = torch.FloatTensor([1, 0, 0, 0, 1, 0])

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 4 * 4)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def forward(self, x):
        x = self.stn(x)
        x = self.conv1_1(F.relu(F.max_pool2d(self.conv1_drop(self.conv1(x)), 2)))
        x = self.conv2_1(F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2)))
        x = self.conv3_1(F.relu(F.max_pool2d(self.conv3_drop(self.conv3(x)), 2)))
        # y = x
        # y = F.max_pool2d(y,2)
        # x = F.relu(F.max_pool2d(self.conv1_drop(self.conv2_1(self.conv2(x))), 2))
        # y = y.view(-1, 12*8*8)
        x = x.view(-1, 4*4*180)
        # x = torch.cat([x, y],1)
        x = F.relu(self.fbn(self.fc1(x)))
        x = F.dropout(x, training=self.training)
        # x = F.relu(self.fc2(x))
        x = self.fc2(x)
        return F.log_softmax(x)



# class BasicBlock(nn.Module):
#     expansion = 1

#     def __init__(self, inplanes, planes, stride=1, downsample=None):
#         super(BasicBlock, self).__init__()
#         self.conv1 = conv3x3(inplanes, planes, stride)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = conv3x3(planes, planes)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.downsample = downsample
#         self.stride = stride

#     def forward(self, x):
#         residual = x

#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         out = self.conv2(out)
#         out = self.bn2(out)

#         if self.downsample is not None:
#             residual = self.downsample(x)

#         out += residual
#         out = self.relu(out)

#         return out