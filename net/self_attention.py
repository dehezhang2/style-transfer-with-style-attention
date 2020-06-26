import torch
import torch.nn as nn
import utils

class SelfAttention(nn.Module):
    def __init__(self, channel_num):
        super(SelfAttention, self).__init__()
        self.f = nn.Conv2d(channel_num, channel_num // 2, kernel_size=1) # [b, c', h, w]
        self.g = nn.Conv2d(channel_num, channel_num // 2, kernel_size=1) # [b, c', h, w]
        self.h = nn.Conv2d(channel_num, channel_num, kernel_size=1) # [b, c, h, w]
        self.softmax = nn.Softmax(dim=-1)


    def forward(x):
        x_size = x.shape
        f = utils.hw_flatten(self.f(x)).permute(0, 2, 1) # [b, n, c']
        g = utils.hw_flatten(self.g(x)) # [b, c', n]
        h = utils.hw_flatten(self.h(x)) # [b, c, n]
        energy = torch.bmm(f, g) # [b, n, n]
        attention = self.softmax(energy) # [b, n, n]
        ret = torch.bmm(h, attention.permute(0, 2, 1)) # [b, c, n]
        ret = ret.view(x_size)# [b, c, h, w]
        return ret
        


        

