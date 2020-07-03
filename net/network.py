import torch
import torch.nn as nn
import torchvision
import net.utils as utils

FEATURE_CHANNEL = 512

# Encoder
class Encoder(nn.Module):
    def __init__(self, use_gpu=True):
        super(Encoder, self).__init__()
        vgg = torchvision.models.vgg19(pretrained=True).features
        if use_gpu == True:
            vgg.cuda()
        enc_layers = list(vgg.children())
        self.layers = [nn.Sequential(*enc_layers[:2]),
                      nn.Sequential(*enc_layers[2:7]),
                      nn.Sequential(*enc_layers[7:12]),
                      nn.Sequential(*enc_layers[12:21])]
        self.layer_names = ['conv1', 'conv2', 'conv3', 'conv4']
        
    def forward(self, x):
        out = {}
        for i in range(0, len(self.layers)):
            x = self.layers[i](x)
            out[self.layer_names[i]] = x
        return out
    
    def __str__(self):
        ans = ''
        for layer in self.layers:
            ans += (layer.__str__() + '\n')
        return ans

# Decoder
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.deconv_4 = nn.Sequential(
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(512, 256, (3, 3)),
                nn.ReLU(),
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(256, 256, (3, 3)),
                nn.ReLU(),
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(256, 256, (3, 3)),
                nn.ReLU(),
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(256, 256, (3, 3)),
                nn.ReLU(),
            )
        self.deconv_3 = nn.Sequential(
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(256, 128, (3, 3)),
                nn.ReLU(),
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(128, 128, (3, 3)),
                nn.ReLU(),
            )
        self.deconv_2 = nn.Sequential(
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(128, 64, (3, 3)),
                nn.ReLU(),
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(64, 64, (3, 3)),
                nn.ReLU(),
            )
        self.deconv_1 = nn.Sequential(
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(64, 3, (3, 3)),
            )
        
    def forward(self, x, encode_features):
        x = self.deconv_4(x)
        # print('Shape: input ({}); encode feature ({})'.format(x.size(), encode_features['conv3'].size()))
        x = utils.adaptive_instance_normalization(x, encode_features['conv3'])
        x = self.deconv_3(x)
        # print('Shape: input ({}); encode feature ({})'.format(x.size(), encode_features['conv2'].size()))
        x = utils.adaptive_instance_normalization(x, encode_features['conv2'])
        x = self.deconv_2(x)
        # print('Shape: input ({}); encode feature ({})'.format(x.size(), encode_features['conv1'].size()))
        x = utils.adaptive_instance_normalization(x, encode_features['conv1'])
        x = self.deconv_1(x)
        return x + (127.5/255.0)

# Self Attention Module
class SelfAttention(nn.Module):
    def __init__(self):
        super(SelfAttention, self).__init__()
        self.f = nn.Conv2d(FEATURE_CHANNEL, FEATURE_CHANNEL // 2, kernel_size=1) # [b, c', h, w]
        self.g = nn.Conv2d(FEATURE_CHANNEL, FEATURE_CHANNEL // 2, kernel_size=1) # [b, c', h, w]
        self.h = nn.Conv2d(FEATURE_CHANNEL, FEATURE_CHANNEL, kernel_size=1) # [b, c, h, w]
        self.softmax = nn.Softmax(dim=-1)


    def forward(self, x):
        x_size = x.shape
        f = utils.hw_flatten(self.f(x)).permute(0, 2, 1) # [b, n, c']
        g = utils.hw_flatten(self.g(x)) # [b, c', n]
        h = utils.hw_flatten(self.h(x)) # [b, c, n]
        energy = torch.bmm(f, g) # [b, n, n]
        attention = self.softmax(energy) # [b, n, n]
        ret = torch.bmm(h, attention.permute(0, 2, 1)) # [b, c, n]
        ret = ret.view(x_size)# [b, c, h, w]
        return ret

# Reconstruction Network with Self-attention Module
class AttentionNet(nn.Module):
    def __init__(self, seperate=False, attn=None, decoder=None):
        super(AttentionNet, self).__init__()
        self.perceptual_loss_layers = ['conv1', 'conv2', 'conv3', 'conv4']

        self.recons_weight = 10
        self.perceptual_weight = 1
        self.tv_weight = 10
        self.attention_weight = 6

        self.encode = Encoder()

        self.seperate = seperate
        # self.decode = Decoder()
        
        if self.seperate == True:
            self.self_attn_content = SelfAttention()
            self.self_attn_style = SelfAttention()
            self.content_decode = Decoder()
            self.style_decode = Decoder()
        else:
            self.self_attn = SelfAttention() if attn == None else attn
            self.decode = Decoder() if decoder == None else decoder

        self.mse_loss = nn.MSELoss()

    def get_encoder(self):
        return self.encode

    def self_attention_autoencoder(self, x, cal_self_attn): # in case kernels are not seperated
        input_features = self.encode(x)
        projected_hidden_feature, colorization_kernels, mean_features = utils.adain_normalization(input_features['conv4'])
        attention_feature_map = cal_self_attn(projected_hidden_feature)

        hidden_feature = projected_hidden_feature * attention_feature_map + projected_hidden_feature
        hidden_feature = utils.adain_colorization(hidden_feature, colorization_kernels, mean_features)

        output = self.decode(hidden_feature, input_features)
        return output, attention_feature_map
    
    def calc_recon_loss(self, x, target):
        recons_loss = self.mse_loss(x, target)
        return recons_loss
    
    def calc_perceptual_loss(self, x, target):
        input_feat = self.encode(x)
        output_feat = self.encode(target)
        
        perceptual_loss = 0.0

        for layer in self.perceptual_loss_layers:
            input_per_feat = input_feat[layer]
            output_per_feat = output_feat[layer]
            perceptual_loss += self.mse_loss(input_per_feat, output_per_feat)
        return perceptual_loss

    def calc_tv_loss(self, x):
        tv_loss = torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])) 
        tv_loss += torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
        return tv_loss
    
    def calc_attention_loss(self, att_map):
        return torch.norm(att_map, p=1)

    def forward(self, x):
        # returns loss, output, attention_map
        # seperate must be False in this case
        output, attention_feature_map = self.self_attention_autoencoder(x, self.self_attn)
        output = utils.batch_mean_image_subtraction(output)
        recon_loss = self.calc_recon_loss(x, output) * (255**2 / 4)
        perceptual_loss =  self.calc_perceptual_loss(x, output) * (255**2 / 4)
        tv_loss = self.calc_tv_loss(output) * (255 / 2)
        attention_loss = self.calc_attention_loss(attention_feature_map)
        total_loss = recon_loss * self.recons_weight + perceptual_loss * self.perceptual_weight \
                    + tv_loss * self.tv_weight + attention_loss * self.attention_weight
        loss_dict = {'total': total_loss, 'construct': recon_loss, 'percept': perceptual_loss, 'tv': tv_loss, 'attn': attention_loss}
        return loss_dict, output, attention_feature_map

    
    # def forward(self, content, style):
    #     pass

    
    

