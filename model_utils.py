from torch import nn

class Upsample(nn.Module):
    """
    The transpose convolution is reverse of the convolution operation. 
    Here, the kernel is placed over the input image pixels. 
    The pixel values are multiplied successively by the kernel weights to produce the upsampled image. 
    In case of overlapping, the values are summed. 
    The kernel weights in upsampling are learned the same way as in convolutional operation 
    that’s why it’s also called learnable upsampling.
    """
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, dropout=True):
        super(Upsample, self).__init__()
        self.dropout = dropout
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=nn.InstanceNorm2d),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.dropout_layer = nn.Dropout2d(0.5)

    def forward(self, x, shortcut=None):
        x = self.block(x)
        if self.dropout:
            x = self.dropout_layer(x)

        if shortcut is not None:
            x = torch.cat([x, shortcut], dim=1)

        return x

class Downsample(nn.Module):
    """
    The normal convolution (without stride) operation gives the same 
    size output image as input image e.g. 3x3 kernel (filter) convolution on 
    4x4 input image with stride 1 and padding 1 gives the same-size output. 
    But strided convolution results in downsampling i.e. reduction in size of input image e.g. 
    3x3 convolution with stride 2 and padding 1 convert image of size 4x4 to 2x2.
    """
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, apply_instancenorm=True):
        super(Downsample, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=nn.InstanceNorm2d)
        self.norm = nn.InstanceNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.apply_norm = apply_instancenorm

    def forward(self, x):
        x = self.conv(x)
        if self.apply_norm:
            x = self.norm(x)
        x = self.relu(x)

        return x
