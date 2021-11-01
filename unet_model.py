import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class DoubleConvReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(DoubleConvReLU, self).__init__()
        self.__conv_1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
        self.__conv_2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size)

    def forward(self, x):
        x = self.__conv_1(x)
        x = F.relu(x)
        x = self.__conv_2(x)
        x = F.relu(x)

        return x


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.__contracting_features = list()

        # Contracting path.
        self.__double_conv_relu_1 = DoubleConvReLU(in_channels=1, out_channels=64, kernel_size=3)
        self.__double_conv_relu_2 = DoubleConvReLU(in_channels=64, out_channels=128, kernel_size=3)
        self.__double_conv_relu_3 = DoubleConvReLU(in_channels=128, out_channels=256, kernel_size=3)
        self.__double_conv_relu_4 = DoubleConvReLU(in_channels=256, out_channels=512, kernel_size=3)
        self.__double_conv_relu_5 = DoubleConvReLU(in_channels=512, out_channels=1024, kernel_size=3)

        # Expansive path.
        self.__up_conv_1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)
        self.__double_conv_relu_6 = DoubleConvReLU(in_channels=1024, out_channels=512, kernel_size=3)
        self.__up_conv_2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        self.__double_conv_relu_7 = DoubleConvReLU(in_channels=512, out_channels=256, kernel_size=3)
        self.__up_conv_3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.__double_conv_relu_8 = DoubleConvReLU(in_channels=256, out_channels=128, kernel_size=3)
        self.__up_conv_4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.__double_conv_relu_9 = DoubleConvReLU(in_channels=128, out_channels=64, kernel_size=3)
        self.__final_conv = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1)

    @staticmethod
    def __concat(x_prev, x):
        prev_height, prev_width = np.shape(x_prev)[2:4]
        curr_height, curr_width = np.shape(x)[2:4]

        height_offset = (prev_height - curr_height) // 2
        width_offset = (prev_width - curr_width) // 2

        x_prev = x_prev[..., height_offset: height_offset+curr_height, width_offset: width_offset+curr_width]

        return torch.cat((x_prev, x), dim=1)

    def __contracting_path(self, x):
        x = self.__double_conv_relu_1(x)
        self.__contracting_features.append(x)
        x = F.max_pool2d(x, kernel_size=(2, 2), stride=2)

        x = self.__double_conv_relu_2(x)
        self.__contracting_features.append(x)
        x = F.max_pool2d(x, kernel_size=(2, 2), stride=2)

        x = self.__double_conv_relu_3(x)
        self.__contracting_features.append(x)
        x = F.max_pool2d(x, kernel_size=(2, 2), stride=2)

        x = self.__double_conv_relu_4(x)
        self.__contracting_features.append(x)
        x = F.max_pool2d(x, kernel_size=(2, 2), stride=2)

        x = self.__double_conv_relu_5(x)

        return x

    def __expansive_path(self, x):
        x = self.__up_conv_1(x)
        x_prev = self.__contracting_features[-1]
        x = self.__concat(x_prev, x)
        x = self.__double_conv_relu_6(x)

        x = self.__up_conv_2(x)
        x_prev = self.__contracting_features[-2]
        x = self.__concat(x_prev, x)
        x = self.__double_conv_relu_7(x)

        x = self.__up_conv_3(x)
        x_prev = self.__contracting_features[-3]
        x = self.__concat(x_prev, x)
        x = self.__double_conv_relu_8(x)

        x = self.__up_conv_4(x)
        x_prev = self.__contracting_features[-4]
        x = self.__concat(x_prev, x)
        x = self.__double_conv_relu_9(x)

        x = self.__final_conv(x)

        return x

    def forward(self, x):
        x = self.__contracting_path(x)
        x = self.__expansive_path(x)

        return x

#
# if __name__ == "__main__":
#     net = UNet()
#     net.cuda()
#     summary(net, (1, 250, 250))
