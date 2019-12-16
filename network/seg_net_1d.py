import torch
from torch import nn
from torch.nn.functional import interpolate


def conv_ax_downsampler(in_c, out_c):
    """Conv with downsampling only in h-dimension (axial)

    Args:
        in_c: input channels
        out_c: output channels
    """
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, (5, 3), stride=(2, 1), padding=(2, 1), bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU6(inplace=True)
    )


class ConvAxUpsampler(nn.Module):
    """Upsampling block only in H-dimension (axial)

    - 2x bilinear upsampling axially
    - concatete skin connection
    - regular convolution

    Args:
        cat_c: channels of skip connection
        in_c: input channels
        out_c: output channels
    """
    def __init__(self, cat_c, in_c, out_c):
        super(ConvAxUpsampler, self).__init__()
        tot_in_c = cat_c + in_c
        self.block = nn.Sequential(
            nn.Conv2d(tot_in_c, out_c, (3, 3), stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU6(inplace=True)
        )

    def forward(self, x0, x):
        x = interpolate(x, scale_factor=(2, 1), mode='bilinear', align_corners=True)
        x = torch.cat((x0, x), dim=1)
        return self.block(x)


class Bottleneck(nn.Module):
    """MobileNetV2 Bottleneck Block

    - 1x1 conv expansion to `in_c * expand` channels
        - sub-block is omitted if `expand` is 1
    - 3x3 depthwise conv, with optional stride
    - 1x1 conv projection to `out_c` channels
    - Residual connection if able (i/o same shape)
    - BatchNorm and ReLU6 after every conv except last

    Args:
        in_c: input channels
        out_c: output channels
        expand: expansion factor. Default: 6
        stride: stride of first conv. Default: 1
    """
    def __init__(self, in_c, out_c, expand=6, stride=1):
        super(Bottleneck, self).__init__()
        mid_c = round(in_c * expand)
        if expand == 1:
            layers = []
        else:
            # Expand
            layers = [
                nn.Conv2d(in_c, mid_c, 1, bias=False),
                nn.BatchNorm2d(mid_c),
                nn.ReLU6(inplace=True),
            ]
        layers.extend([
            # Depthwise Conv
            nn.Conv2d(mid_c, mid_c, 3, stride, padding=1, groups=mid_c, bias=False),
            nn.BatchNorm2d(mid_c),
            nn.ReLU6(inplace=True),
            # Project
            nn.Conv2d(mid_c, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c),
        ])
        self.block = nn.Sequential(*layers)
        self.res_connect = (stride == 1 and in_c == out_c)

    def forward(self, x):
        if self.res_connect:
            return self.block(x) + x
        else:
            return self.block(x)


def down_block(in_c, out_c, expand, stride, number):
    """Downsampling block

    - a series of `Bottleneck` blocks
    - first block converts `in_c` channels to `out_c` channels
    - only the first block uses stride
    - all blocks use the same expand ratio

    Args:
        in_c: input channels
        out_c: output channels
        expand: expansion factor
        stride: stride of first bottleneck block
        number: number of blocks
    """
    # layers = []
    # for n in range(number):
    #     if n == 0:
    #         layers.append(Bottleneck(in_c, out_c, expand, stride))
    #     else:
    #         layers.append(Bottleneck(out_c, out_c, expand))
    layers = [Bottleneck((n == 0) * in_c or out_c, out_c, expand,
                         (n == 0) * stride or 1) for n in range(number)]
    return nn.Sequential(*layers)


class UpBlock(nn.Module):
    """Upsampling block

    - upsample 2x by bilinear interpolation
    - concatenate with skip connection
    - a series of `Bottleneck` blocks
    - first block converts to `out_c` channels
    - all blocks use the same expand ratio. The first block's expand ratio
        is adjusted to ignore the additional `cat_c` channels.

    Args:
        cat_c: channels of skin connection
        in_c: input channels
        out_c: output channels
        expand: expansion factor
        number: number of blocks
    """
    def __init__(self, cat_c, in_c, out_c, expand, number):
        super(UpBlock, self).__init__()
        tot_in_c = cat_c + in_c
        ratio = in_c / tot_in_c
        # layers = []
        # for n in range(number):
        #     if n == 0:
        #         layers.append(Bottleneck(tot_in_c, out_c, ratio * expand))
        #     else:
        #         layers.append(Bottleneck(out_c, out_c, expand))
        layers = [Bottleneck((n == 0) * tot_in_c or out_c, out_c,
                             ((n == 0) * ratio or 1) * expand) for n in range(number)]
        self.block = nn.Sequential(*layers)

    def forward(self, x0, x):
        x = interpolate(x, scale_factor=(2, 2), mode='bilinear', align_corners=True)
        x = torch.cat((x0, x), dim=1)
        return self.block(x)


class CoordRegress(nn.Module):
    """Coordinate Regression Module.

    - returns a singleton coordinate per channel
    - coordinate is normalized by length to [0, 1)
    - finds expected value of coordinate using the heatmap
        as the probability distribution

    Input  shape: (batches, channels, length)
    Output shape: (batches, channels)

    Args:
        length: length of spatial dimension
    """
    def __init__(self, length):
        super(CoordRegress, self).__init__()
        self.register_buffer(
            'xrange',
            torch.arange(length, dtype=torch.float32) / length
        )

    def forward(self, p):
        # pytorch automatically broadcasts size (L) to (b,c,L)
        coords = torch.mul(p, self.xrange)
        return torch.sum(coords, dim=-1)


class SegNet1D(nn.Module):
    def __init__(self,
                 input_size=(1024, 32),
                 c_mult=6,
                 e_fact=5,
                 ):
        """SegNet1D

        Args:
            input_size: Defaults to (1024, 32)
            c_mult: channel multiplier. Defaults to 6.
            e_fact: expansion factor. Defaults to 5.
        """
        super(SegNet1D, self).__init__()
        self.full_image_mode = False
        h, w = input_size
        assert h % 64 == 0
        assert w % 4 == 0
        self.center = w // 2
        # Adding coordinate prior improves performance
        arange = torch.arange(h, dtype=torch.float32) / h
        # aspect: 52 pixels per mm
        # ratio: 19.69 mm to full range
        arange = arange[None, None, :, None]
        self.register_buffer('arange', arange)

        # Initial params
        conv0_c = 4*c_mult
        down_params = [
            # TODO: Swap param ordering
            # expand, out_c, stride, number
            [1,      2*c_mult, 1, 1],
            [e_fact, 3*c_mult, 2, 2],
            [e_fact, 4*c_mult, 2, 3],
        ]
        up_params = [
            # expand, cat_c, out_c, number
            [e_fact, 3*c_mult, 3*c_mult, 1],
            [e_fact, 2*c_mult, 2*c_mult, 1],
        ]

        # Build network components
        self.innorm = nn.InstanceNorm2d(1, affine=False, track_running_stats=False)

        self.conv0 = conv_ax_downsampler(2, c_mult)
        self.conv1 = conv_ax_downsampler(c_mult, 2*c_mult)
        self.conv2 = conv_ax_downsampler(2*c_mult, 2*c_mult)
        self.conv3 = conv_ax_downsampler(2*c_mult, conv0_c)

        self.down = nn.ModuleList()
        in_c = conv0_c
        for expand, out_c, stride, number in down_params:
            self.down.append(down_block(in_c, out_c, expand, stride, number))
            in_c = out_c

        self.up = nn.ModuleList()
        for expand, cat_c, out_c, number in up_params:
            self.up.append(UpBlock(cat_c, in_c, out_c, expand, number))
            in_c = out_c

        self.conv_up0 = ConvAxUpsampler(2*c_mult, 2*c_mult, 2*c_mult)
        self.conv_up1 = ConvAxUpsampler(2*c_mult, 2*c_mult, c_mult)
        self.conv_up2 = ConvAxUpsampler(c_mult, c_mult, c_mult)

        in_c = c_mult
        self.prof = nn.Sequential(
            nn.Conv2d(in_c, 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(2),
            # No ReLU before softmax !!
        )
        self.softmax = nn.Softmax(dim=2)
        self.coord = CoordRegress(h)

    def forward(self, x):
        # TODO: Learnable gamma
        x = self.innorm(x)
        x = torch.cat([x, self.arange.expand(x.shape)], dim=1)
        x00 = self.conv0(x)  # h/2 x w
        x01 = self.conv1(x00)  # h/4 x w
        x02 = self.conv2(x01)  # h/8 x w
        x = self.conv3(x02)  # h/16 x w
        x0 = self.down[0](x)  # h/16 x w (down0 has stride 1)
        x1 = self.down[1](x0)  # h/32 x w/2
        x2 = self.down[2](x1)  # h/64 x w/4
        x12 = self.up[0](x1, x2)  # h/32 x w/2
        x = self.up[1](x0, x12)  # h/16 x w
        x = self.conv_up0(x02, x)  # h/8 x w
        x = self.conv_up1(x01, x)  # h/4 x w
        x = self.conv_up2(x00, x)  # h/2 x w
        if not self.full_image_mode:
            # Take the central slice
            p = self.prof(x[..., self.center, None])  # 2 x h/2 x 1
            p = torch.squeeze(p, dim=3)  # 2 x h/2 x None
            p = interpolate(p, scale_factor=2, mode='linear', align_corners=True)
            p = self.softmax(p)  # 2 x h
            c = self.coord(p)  # 2 x None
            return c, self.regularizer(p, c)
        else:
            p = self.prof(x)  # 2 x h/2 x w
            p = interpolate(p, scale_factor=(2, 1), mode='bilinear', align_corners=True)
            p = self.softmax(p)  # 2 x h x w
            c = self.coord(p.permute(0, 1, 3, 2))  # 2 x w
            return c, p

    def regularizer(self, profile, c):
        """Calculates variance of heatmap as probability distribution"""
        c = c[..., None]
        xrange = self.coord.xrange.repeat(c.shape)
        xrange -= c
        xrange = xrange**2
        sqrerr = torch.mul(profile, xrange)
        return torch.mean(torch.sum(sqrerr, dim=2))


if __name__ == '__main__':
    m = SegNet1D(c_mult=6, e_fact=5)
    n_params = sum(p.numel() for p in m.parameters())
    print(n_params)  # 53590
