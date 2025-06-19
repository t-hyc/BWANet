
import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers

import einops
from einops import rearrange
import numpy as np

##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class BWT(nn.Module):
    def __init__(self):
        super(BWT, self).__init__()

    def forward(self, x):
        x01 = x[:, :, 0::2, :] / 2
        x02 = x[:, :, 1::2, :] / 2
        x1 = x01[:, :, :, 0::2]
        x2 = x02[:, :, :, 0::2]
        x3 = x01[:, :, :, 1::2]
        x4 = x02[:, :, :, 1::2]
        x_LL = x1 + x2 + x3 + x4
        x_HL = -x1 - x2 + x3 + x4
        x_LH = -x1 + x2 - x3 + x4
        x_HH = x1 - x2 - x3 + x4
        x_HD = x1 + x2 - x3 - x4
        x_LV = -x1 + x2 + x3 - x4
        x_LD = x1 - x2 + x3 - x4
        x_HV = -x1 - x2 - x3 - x4
        return torch.cat((x_LL, x_HL, x_LH, x_HH, x_HD, x_LV, x_LD, x_HV), 1)


class IBWT(nn.Module):
    def __init__(self):
        super(IBWT, self).__init__()

    def forward(self, x):
        r = 2
        in_batch, in_channel, in_height, in_width = x.size()
        out_batch, out_channel, out_height, out_width = in_batch, int(
            in_channel / (r ** 3)), r * in_height, r * in_width
        x1 = x[:, 0:out_channel, :, :] / 2
        x2 = x[:, out_channel:out_channel * 2, :, :] / 2
        x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
        x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2
        x5 = x[:, out_channel * 4:out_channel * 5, :, :] / 2
        x6 = x[:, out_channel * 5:out_channel * 6, :, :] / 2
        x7 = x[:, out_channel * 6:out_channel * 7, :, :] / 2
        x8 = x[:, out_channel * 7:out_channel * 8, :, :] / 2
        h = torch.zeros([out_batch, out_channel, out_height, out_width]).cuda()

        h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4 + x5 - x6 + x7 - x8
        h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4 - x5 + x6 - x7 + x8
        h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4 - x5 - x6 + x7 + x8
        h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8

        return h


def conv(in_channels, out_channels, kernel_size, bias=False, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, stride=stride)


class SA(nn.Module):
    def __init__(self, kernel_size=5, bias=False):
        super(SA, self).__init__()
        self.conv_du = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        avg_pool = torch.mean(x, 1, keepdim=True)
        channel_pool = torch.cat([max_pool, avg_pool], dim=1)
        y = self.conv_du(channel_pool)

        return x * y


class CA(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class BWAB(nn.Module):
    def __init__(self, n_feat, o_feat, kernel_size, reduction, bias, act):
        super(BWAB, self).__init__()
        self.bwt = BWT()
        self.ibwt = IBWT()

        modules_body = [
            conv(n_feat * 8, n_feat, kernel_size=1, bias=bias),
            act,
            conv(n_feat, n_feat * 8, kernel_size=1, bias=bias)
        ]
        self.body = nn.Sequential(*modules_body)

        self.SA = SA()
        self.CA = CA(n_feat * 8, reduction, bias=bias)

        self.conv1x1 = nn.Conv2d(n_feat * 8, n_feat * 8, kernel_size=1, bias=bias)
        self.conv3x3 = nn.Conv2d(n_feat, o_feat, kernel_size=3, padding=1, bias=bias)
        self.conv3x31 = nn.Conv2d(n_feat, o_feat, kernel_size=3, padding=1, bias=bias)
        self.activate = act
        self.conv1x1_final = nn.Conv2d(n_feat, o_feat, kernel_size=1, bias=bias)

    def forward(self, x):
        residual = x

        x_bwt = self.bwt(x)

        res = self.body(x_bwt)

        res = self.SA(res)
        res = self.CA(res)

        res = self.conv1x1(res) + x_bwt
        wavelet_path = self.ibwt(res)

        out = self.conv3x31(self.activate(self.conv3x3(wavelet_path)))
        out += self.conv1x1_final(residual)

        return out

class PLKB(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.norm1 = nn.BatchNorm2d(dim)

        self.conv1 = nn.Conv2d(dim, dim, kernel_size=1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=5, padding=2, padding_mode='reflect')

        self.dw_19 = nn.Conv2d(dim, dim, kernel_size=7, padding=9, dilation=3,stride=1, groups=dim, padding_mode='reflect')
        self.dw_13 = nn.Conv2d(dim, dim, kernel_size=5, padding=6,dilation=3, stride=1, groups=dim, padding_mode='reflect')
        self.dw_7 = nn.Conv2d(dim, dim, kernel_size=3, padding=3, stride=1,dilation=3, groups=dim, padding_mode='reflect')
        self.dw_1 = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, groups=dim)

        ### fca ###
        self.fac_conv = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.fac_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp = nn.Sequential(
            nn.Conv2d(dim * 5, dim * 4, 1),
            nn.GELU(),
            # nn.ReLU(True),
            nn.Conv2d(dim * 4, dim, 1)
        )
    def forward(self, x):
        identity = x
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)

        # ### fca ###
        x_att = self.fac_conv(self.fac_pool(x))
        x_fft = torch.fft.fft2(x, norm='backward')
        x_fft = x_att * x_fft
        x_fca = torch.fft.ifft2(x_fft, dim=(-2, -1), norm='backward').real

        x = torch.cat([self.dw_1(x),self.dw_19(x),self.dw_13(x),self.dw_7(x),x_fca], dim=1)
        x = self.mlp(x)
        # x = identity + x
        return x

class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, bias=True,LayerNorm_type='WithBias'):
        super(BasicConv, self).__init__()
        padding = kernel_size // 2
        layers = list()

        layers.append(
            nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        layers.append(LayerNorm(out_channel, LayerNorm_type))
        layers.append(nn.GELU())
        layers.append(
            nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)+x

class ResBlock(nn.Module):
    def __init__(self, dim, num):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=True)
        self.gelu=nn.GELU()
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=True)
        self.bwab=BWAB(n_feat=dim, o_feat=dim, kernel_size=3, reduction=16, bias=False, act=nn.PReLU())
        self.plkb=PLKB(dim)
        self.res = nn.Sequential(*[BasicConv(dim, dim) for i in range(num)])
        self.bn=LayerNorm(dim, LayerNorm_type='WithBias')
    def forward(self, x):
        out = self.res(x)
        rres=out
        out = self.conv1(out)
        out=self.bn(out)
        out = self.gelu(out)
        out = self.plkb(out)
        out = self.bwab(out)

        out = self.conv2(out)
        out+=rres
        return out




##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x



##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

class SKFusion(nn.Module):
    def __init__(self, dim, height=2, reduction=8):
        super(SKFusion, self).__init__()

        self.height = height
        d = max(int(dim / reduction), 4)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, d, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(d, dim * height, 1, bias=False)
        )

        self.softmax = nn.Softmax(dim=1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x, g):
        """
        x: encoder feature map (B, C, H, W)
        g: decoder feature map (B, C, H, W)
        return: fused output, and attention-weighted x & g
        """
        B, C, H, W = x.shape

        # Stack along height dimension
        feats = torch.stack([x, g], dim=1)  # shape: (B, 2, C, H, W)

        # Attention weights
        feats_sum = torch.sum(feats, dim=1)  # (B, C, H, W)
        attn = self.mlp(self.avg_pool(feats_sum))  # (B, C*2, 1, 1)
        attn = attn.view(B, self.height, C, 1, 1)
        attn = self.softmax(attn)  # (B, 2, C, 1, 1)

        # Compute attention-weighted x and g separately
        x_attn = feats[:, 0, :, :, :] * attn[:, 0, :, :, :]
        g_attn = feats[:, 1, :, :, :] * attn[:, 1, :, :, :]

        out = x_attn + g_attn

        # Fused output
        g = self.proj(g)
        return out+g








##########################################################################
##---------- BWANet -----------------------
class BWANet(nn.Module):
    def __init__(self,
        inp_channels=3,
        out_channels=3,
        dim = 16,
        num_blocks = [2,2,4],
        bias = False,
        skip = True
    ):

        super(BWANet, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_1 = nn.Sequential(*[ResBlock(dim=dim,num=1) for i in range(num_blocks[0])])

        self.down_1 = Downsample(int(dim)) ## From Level 0 to Level 1

        self.decoder_level1_0 = nn.Sequential(*[ResBlock(dim=int(int(dim * 2)),num=1) for i in range(num_blocks[1])])

        self.down_2 = Downsample(int(dim *2)) ## From Level 1 to Level 2
        self.decoder_level2_0 = nn.Sequential(*[ResBlock(dim=int(int(dim * 2 * 2)),num=1) for i in range(num_blocks[2])])

        self.up2_1 = Upsample(int(dim * 2 *2)) ## From Level 2 to Level 1
        self.skf1 = SKFusion(dim * 2)

        self.decoder_level1_1 = nn.Sequential(*[ResBlock(dim=int(int(dim * 2)),num=1) for i in range(num_blocks[1])])
        self.up2_0 = Upsample(int(dim * 2))  ## From Level 1 to Level 0
        self.skf0 = SKFusion(dim)

        self.refinement_1 = nn.Sequential(*[ResBlock(dim=int(dim),num=1) for i in range(num_blocks[0])])

        self.output = nn.Conv2d(int(dim), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.skip = skip


    def forward(self, inp_img):

        inp_enc_encoder1 = self.patch_embed(inp_img)
        out_enc_encoder1 = self.encoder_1(inp_enc_encoder1)

        inp_enc_level1_0 = self.down_1(out_enc_encoder1)
        out_enc_level1_0 = self.decoder_level1_0(inp_enc_level1_0)

        inp_enc_level2_0 = self.down_2(out_enc_level1_0)
        out_enc_level2_0 = self.decoder_level2_0(inp_enc_level2_0)

        out_enc_level2_1 = self.up2_1(out_enc_level2_0)

        inp_enc_level1_1 = self.skf1(out_enc_level1_0, out_enc_level2_1)

        out_enc_level1_1 = self.decoder_level1_1(inp_enc_level1_1)

        out_enc_level1_1 = self.up2_0(out_enc_level1_1)

        out = self.skf0(out_enc_encoder1, out_enc_level1_1)

        out = self.refinement_1(out)

        if self.skip:
            out = self.output(out)+ inp_img
        else:
            out = self.output(out)

        return out

