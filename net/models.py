from net.aams import AttentionNet
import torch 
import torch.nn as nn
import torch.nn.functional as F
import net.utils as utils
from net.utils import truncated_normal_, KMeans, project_features

class AAMS():
    def __init__(self, attention_net):
        self.attention_net = attention_net
        self.perceptual_loss_layers = attention_net.get_layers()
        self.encode = attention_net.get_encoder()

    # https://github.com/irasin/Pytorch_Style_Swap
    def style_swap(self, content_feature, style_feature, kernel_size = 5, stride=1):
        # content_feature and style_feature should have shape as (1, C, H, W)
        # kernel_size here is equivalent to extracted patch size

        # extract patches from style_feature with shape (1, C, H, W)
        kh, kw = kernel_size, kernel_size
        sh, sw = stride, stride

        patches = style_feature.unfold(2, kh, sh).unfold(3, kw, sw)

        patches = patches.permute(0, 2, 3, 1, 4, 5)
        patches = patches.reshape(-1, *patches.shape[-3:]) # (patch_numbers, C, kh, kw)

        # calculate Frobenius norm and normalize the patches at each filter
        norm = torch.norm(patches.reshape(patches.shape[0], -1), dim=1).reshape(-1, 1, 1, 1)

        noramalized_patches = patches / norm

        conv_out = F.conv2d(content_feature, noramalized_patches)

        # calculate the argmax at each spatial location, which means at each (kh, kw),
        # there should exist a filter which provides the biggest value of the output
        one_hots = torch.zeros_like(conv_out)
        one_hots.scatter_(1, conv_out.argmax(dim=1, keepdim=True), 1)

        # deconv/transpose conv
        deconv_out = F.conv_transpose2d(one_hots, patches)

        # calculate the overlap from deconv/transpose conv
        overlap = F.conv_transpose2d(one_hots, torch.ones_like(patches))

        # average the deconv result
        res = deconv_out / overlap
        return res

    def multi_scale_style_swap(self, content_feature, style_feature, kernel_size=5, stride=1):
        c_shape = content_feature.shape
        s_shape = style_feature.shape
        assert (c_shape[1] == s_shape[1]) # same along the channel dimension
        
        combined_feature_maps = []
        for beta in [1.1/2, 1.1/(2**0.5), 1.1]:
            new_height = int(float(s_shape[2]) * beta)
            new_width = int(float(s_shape[3]) * beta)

            # reshape the style image and run style swap
            tmp_style_features = F.interpolate(style_feature, \
                size=(new_height, new_width), mode='bilinear', align_corners=True)
            
            combined_feature = self.style_swap(content_feature, \
                tmp_style_features, kernel_size=kernel_size, stride=stride)
            
            combined_feature_maps.append(combined_feature)
        combined_feature_maps.append(content_feature)
        return combined_feature_maps
    
    def multi_stroke_fusion(self, stylized_maps, attention_map, theta=50.0, mode='softmax'):
        # only accept one image, stylized maps and attention map should have same channel size
        stroke_num = len(stylized_maps)
        if stroke_num == 1:
            return stylized_maps[0]
        
        one_channel_attention = torch.mean(attention_map, 1).unsqueeze(1) # 1*1*h*w
        origin_shape = one_channel_attention.shape
        one_channel_attention = one_channel_attention.reshape((-1, 1)) # stretch to tensor (hw)* 1 
        _, centroids = KMeans(one_channel_attention, stroke_num)
        
        one_channel_attention = one_channel_attention.reshape(origin_shape)
        
        saliency_distances = []
        for i in range(stroke_num):
            saliency_distances.append(torch.abs(one_channel_attention - centroids[i]))
        
        multi_channel_saliency = torch.cat(saliency_distances, 1) # 1*cluster_size*h*w
        
        softmax = nn.Softmax(dim=1)
        
        multi_channel_saliency = softmax(theta*(1.0 - multi_channel_saliency)) # 1*cluster_size*h*w
        
        finial_stylized_map = 0
        for i in range(stroke_num):
            temp = multi_channel_saliency[0, i, :, :].unsqueeze(0).unsqueeze(0) # 1*1*h*w
            finial_stylized_map += temp * stylized_maps[i]
        return finial_stylized_map, centroids

    def attention_filter(self, attention_feature_map, kernel_size=3, mean=6, stddev=5):
        attention_map = torch.abs(attention_feature_map)
        
        attention_mask = attention_map > 2 * torch.mean(attention_map)
        attention_mask = attention_mask.float()
        
        w = torch.randn(kernel_size, kernel_size)
        truncated_normal_(w, mean, stddev)
        w = w / torch.sum(w)
        
        # [in_channels, out_channels, filter_height, filter_width]
        w = torch.unsqueeze(w, 0)
        w = w.repeat(attention_mask.shape[1], 1, 1)
        
        w = torch.unsqueeze(w, 0)
        w = w.repeat(attention_mask.shape[1], 1, 1, 1)
        
        gaussian_filter = nn.Conv2d(attention_mask.shape[1], attention_mask.shape[1], (kernel_size, kernel_size))
        gaussian_filter.weight.data = w
        gaussian_filter.weight.requires_grad = False
        pad_filter = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            gaussian_filter
        )
        pad_filter.cuda()
        attention_map = pad_filter(attention_mask)
        attention_map = attention_map - torch.min(attention_map)
        attention_map = attention_map / torch.max(attention_map)
        return attention_map
    
    def transfer(self, contents, styles, inter_weight=1):
        content_features = self.encode(contents)
        style_features = self.encode(styles)

        content_hidden_feature = content_features[self.perceptual_loss_layers[-1]]
        style_hidden_feature = style_features[self.perceptual_loss_layers[-1]]

        projected_content_features, _, _ = utils.project_features(content_hidden_feature, 'ZCA')
        projected_style_features, style_kernels, mean_style_features = utils.project_features(style_hidden_feature, 'ZCA')

        attention_feature_map = self.attention_net.self_attn(projected_content_features)
        projected_content_features = projected_content_features * attention_feature_map + projected_content_features
        
        attention_map = self.attention_filter(attention_feature_map)

        multi_swapped_features = self.multi_scale_style_swap(projected_content_features, projected_style_features)
        outputs = []
        for feature in multi_swapped_features:
            reconstructed_features = utils.reconstruct_features(feature, style_kernels, mean_style_features, 'ZCA')
            outputs.append(self.attention_net.decode(reconstructed_features, style_features))
        fused_features, centroids = self.multi_stroke_fusion(multi_swapped_features, attention_map, theta=50.0, mode='softmax')

        fused_features = inter_weight * fused_features + (1 - inter_weight) * projected_content_features

        reconstructed_features = utils.reconstruct_features(fused_features, style_kernels, mean_style_features, 'ZCA')
        output = self.attention_net.decode(reconstructed_features, style_features)
        return output, attention_map, centroids, outputs
        # return outputs, attention_map



            