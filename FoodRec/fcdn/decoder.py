import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import modules


import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv3x3GNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, upsample=False):
        super().__init__()
        self.upsample = upsample
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, (3, 3), stride=1, padding=1, bias=False
            ),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
        )
        
        if self.upsample:
            self.upsample_layer = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        x = self.block(x)
        if self.upsample:
            x = self.upsample_layer(x)
        return x


class SegmentationBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_upsamples=0):
        super().__init__()

        blocks = [Conv3x3GNReLU(in_channels, out_channels, upsample=bool(n_upsamples))]

        if n_upsamples > 1:
            for _ in range(1, n_upsamples):
                blocks.append(Conv3x3GNReLU(out_channels, out_channels, upsample=True))

        self.block = nn.Sequential(*blocks)

    def forward(self, x):
        return self.block(x)


class MergeBlock(nn.Module):
    def __init__(self, policy):
        super().__init__()
        if policy not in ["add", "cat"]:
            raise ValueError(
                "`merge_policy` must be one of: ['add', 'cat'], got {}".format(
                    policy
                )
            )
        self.policy = policy

    def forward(self, x):
        if self.policy == 'add':
            return sum(x)
        elif self.policy == 'cat':
            return torch.cat(x, dim=1)
        else:
            raise ValueError(
                "`merge_policy` must be one of: ['add', 'cat'], got {}".format(self.policy)
            )


class FCDNDecoder(nn.Module):
    def __init__(
            self,
            encoder_channels,
            encoder_depth=5,
            pyramid_channels=512,
            segmentation_channels=256,
            dropout=0.2,
            merge_policy="add",
    ):
        super().__init__()

        self.out_channels = segmentation_channels if merge_policy == "add" else segmentation_channels * 4
        if encoder_depth < 3:
            raise ValueError("Encoder depth for FPN decoder cannot be less than 3, got {}.".format(encoder_depth))

        encoder_channels = encoder_channels[::-1]
        encoder_channels = encoder_channels[:encoder_depth + 1]


  
        self.p4 = nn.Conv2d(encoder_channels[0], pyramid_channels, kernel_size=1)
        self.p4d=nn.ConvTranspose2d(encoder_channels[0], pyramid_channels, 2, stride=2)

        self.seg_blocks = nn.ModuleList([
            SegmentationBlock(pyramid_channels, segmentation_channels, n_upsamples=n_upsamples)
            for n_upsamples in [3, 2]
            #for n_upsamples in [2, 1, 1, 0]
            #for n_upsamples in [3, 2, 1, 0]
        ])

        self.merge = MergeBlock(merge_policy)
        self.dropout = nn.Dropout2d(p=dropout, inplace=True)

    def forward(self, *features):
        c2, c3, c4, c5 = features[-4:]

        p4 = self.p4(c5)
        p4d = self.p4d(c5)

        feature_pyramid = [seg_block(p) for seg_block, p in zip(self.seg_blocks, [p4, p4d])]
        x = self.merge(feature_pyramid)
        x = self.dropout(x)

        return x
