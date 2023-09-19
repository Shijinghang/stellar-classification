import torch
import torch.nn as nn


class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        # print(avgout.shape)
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out


class CBAM(nn.Module):
    def __init__(self, in_channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(in_channel)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x) * x
        # print('outchannels:{}'.format(out.shape))
        out = self.spatial_attention(out) * out
        return out


class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, p=1, stride=1):
        super(BasicConv, self).__init__()
        # kernel_size//2为padding 大小 图像大小保持不变
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=p, bias=False)
        # 每次卷积后都要经过一次标准化与激活函数
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = torch.nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class SCNet(nn.Module):
    def __init__(self, nc=7):
        super(SCNet, self).__init__()

        # gri
        self.stage1_gri = nn.Sequential(
            BasicConv(3, 64, 3, stride=2),
            CBAM(64),
            BasicConv(64, 64, 3),
            BasicConv(64, 64, 3),
            BasicConv(64, 128, 3, stride=2),
            BasicConv(128, 128, 3),
            BasicConv(128, 128, 3),
        )

        # urz
        self.stage1_urz = nn.Sequential(
            BasicConv(3, 64, 3, stride=2),
            CBAM(64),
            BasicConv(64, 64, 3),
            BasicConv(64, 64, 3),
            BasicConv(64, 128, 3, stride=2),
            BasicConv(128, 128, 3),
            BasicConv(128, 128, 3),
        )

        self.stage2 = nn.Sequential(
            CBAM(256),
            BasicConv(256, 256, 3),
            BasicConv(256, 256, 3),
            BasicConv(256, 256, 3),
            BasicConv(256, 512, 3, stride=2),
            BasicConv(512, 512, 3),
            BasicConv(512, 512, 3),
            BasicConv(512, 1024, 3, stride=2),
            BasicConv(1024, 1024, 3),
            CBAM(1024),
            BasicConv(1024, 1024, 3),
        )

        self.fc = torch.nn.Sequential(
            nn.Linear(16384, 1024),
            torch.nn.ReLU(),
            nn.Linear(1024, 512),
            torch.nn.ReLU(),
            nn.Linear(512, nc),
        )

    def forward(self, x):
        gri_data = x[:, 0:3, :, :]
        urz_data = x[:, 3:6, :, :]

        gri_data = self.stage1_gri(gri_data)
        urz_data = self.stage1_urz(urz_data)

        x = torch.cat([gri_data, urz_data], axis=1)
        x = self.stage2(x)
        x = x.view(x.size(0), -1)

        return self.fc(x)


class SCNetGRI(nn.Module):

    def __init__(self, nc=7):
        super(SCNetGRI, self).__init__()

        self.part1 = nn.Sequential(
            BasicConv(3, 64, 3, stride=2),
            CBAM(64),
            BasicConv(64, 64, 3),
            BasicConv(64, 64, 3),
        )

        self.part2 = nn.Sequential(
            BasicConv(64, 128, 3, stride=2),
            BasicConv(128, 128, 3),
            BasicConv(128, 128, 3),
        )

        self.part3 = nn.Sequential(
            BasicConv(128, 256, 3, stride=2),
            BasicConv(256, 256, 3),
            BasicConv(256, 256, 3),
        )

        self.part4 = nn.Sequential(
            BasicConv(256, 512, 3, stride=2),
            CBAM(512),
            BasicConv(512, 512, 3),
            BasicConv(512, 512, 3),
        )

        self.fc = torch.nn.Sequential(
            nn.Linear(8192, 1024),
            torch.nn.ReLU(),
            nn.Linear(1024, 128),
            torch.nn.ReLU(),
            nn.Linear(128, nc),
        )

    def forward(self, x):
        x = x[:, 0:3, :, :]

        x = self.part1(x)

        x = self.part2(x)

        x = self.part3(x)

        x = self.part4(x)

        x = x.view(x.size(0), -1)

        return self.fc(x)
