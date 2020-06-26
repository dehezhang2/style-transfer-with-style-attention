import torch
import torch.nn as nn

_R_MEAN = 123.68 / 255.0
_G_MEAN = 116.78 / 255.0
_B_MEAN = 103.94 / 255.0

def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def adaptive_instance_normalization(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)

def adain_normalization(features):
    epsilon = 1e-5
    colorization_kernels, mean_features = torch.std_mean(features, (2, 3), keepdim=True)
    normalized_features = torch.div(features - mean_features, epsilon + colorization_kernels)
    return normalized_features, colorization_kernels, mean_features

def adain_colorization(normalized_features, colorization_kernels, mean_features):
    return colorization_kernels * normalized_features + mean_features

def hw_flatten(x):
    # [b, c, h, w] -> [b, c, h * w]
    return x.view(x.shape[0], x.shape[1], -1)

def batch_mean_image_subtraction(images, means=(_R_MEAN, _G_MEAN, _B_MEAN)):
    if len(images.shape) != 4:
        raise ValueError('Input must be of size [batch, height, width, C>0')
    num_channels = images.shape[1]
    if len(means) != num_channels:
        raise ValueError('len(means) must match the number of channels')
    channels = list(torch.split(images, 1, dim=1))
    # print(images)
    # print(channels)
    for i in range(num_channels):
        channels[i] = channels[i] - means[i]
    return torch.cat(channels, dim=1)
    