from net.network import AttentionNet
import torch 
import torch.nn as nn
import torch.nn.functional as F
import net.utils as utils
from net.utils import truncated_normal_, KMeans, project_features
from net.wct import whitening, coloring

class AAMS():
    def __init__(self, attention_net):
        self.attention_net = attention_net
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
        for beta in [1.0/2, 1.0/(2**0.5), 1.0]:
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
    
    def transfer(self, contents, styles, inter_weight=1, projection_method='ZCA'):
        content_features = self.attention_net.encode(contents)
        style_features = self.attention_net.encode(styles)

        content_hidden_feature = content_features[self.attention_net.perceptual_loss_layers[-1]]
        style_hidden_feature = style_features[self.attention_net.perceptual_loss_layers[-1]]

        projected_content_features, content_kernels, mean_content_features = utils.project_features(content_hidden_feature, projection_method)
        projected_style_features, style_kernels, mean_style_features = utils.project_features(style_hidden_feature, projection_method)

        content_attention_feature_map = self.attention_net.self_attn(projected_content_features)
        style_attention_feature_map = self.attention_net.self_attn(projected_style_features)
        projected_content_features = projected_content_features * content_attention_feature_map + projected_content_features
        
        content_attention_map = self.attention_filter(content_attention_feature_map)
        style_attention_map = self.attention_filter(style_attention_feature_map)

        multi_swapped_features = self.multi_scale_style_swap(projected_content_features, projected_style_features)
        seperate_strokes = []
        for feature in multi_swapped_features:
            reconstructed_features = utils.reconstruct_features(feature, style_kernels, mean_style_features, projection_method)
            temp = self.attention_net.decode(reconstructed_features, style_features)
            temp = utils.batch_mean_image_subtraction(temp)
            seperate_strokes.append(temp)

        fused_features, centroids = self.multi_stroke_fusion(multi_swapped_features, content_attention_map, theta=50.0, mode='softmax')

        fused_features = inter_weight * fused_features + (1 - inter_weight) * projected_content_features

        reconstructed_features = utils.reconstruct_features(fused_features, style_kernels, mean_style_features, projection_method)
        output = self.attention_net.decode(reconstructed_features, style_features)
        output = utils.batch_mean_image_subtraction(output)
        return output, content_attention_map, style_attention_map, centroids, seperate_strokes

class SAVA():
    def __init__(self, attention_net):
        self.attention_net = attention_net
        self.encode = attention_net.get_encoder()

    def attentional_style_swap(self, content_attention_map, style_attention_map, content_feature, style_feature, kernel_size=5, stride=1):
        kh, kw = kernel_size, kernel_size
        sh, sw = stride, stride
        patches = style_feature.unfold(2, kh, sh).unfold(3, kw, sw)

        patches = patches.permute(0, 2, 3, 1, 4 ,5)
        patches = patches.reshape(-1, *patches.shape[-3:])

        norm = torch.norm(patches.reshape(patches.shape[0], -1), dim=1).reshape(-1, 1, 1, 1)
        normalized_patches = patches / norm
        conv_out_feature = F.conv2d(content_feature, normalized_patches)


        attn_patches = style_attention_map.unfold(2, kh, sh).unfold(3, kw, sw)

        attn_patches = attn_patches.permute(0, 2, 3, 1, 4 ,5)
        attn_patches = attn_patches.reshape(-1, *attn_patches.shape[-3:])

        norm = torch.norm(attn_patches.reshape(patches.shape[0], -1), dim=1).reshape(-1, 1, 1, 1)
        normalized_attn_patches = attn_patches / norm
        conv_out_attention = F.conv2d(content_attention_map, normalized_attn_patches)

        conv_out = conv_out_feature + conv_out_attention

        one_hots = torch.zeros_like(conv_out)
        one_hots.scatter_(1, conv_out.argmax(dim=1, keepdim=True), 1)

        deconv_out = F.conv_transpose2d(one_hots, patches)
        overlap = F.conv_transpose2d(one_hots, torch.ones_like(patches))

        res = deconv_out / overlap
        return res




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
        for beta in [1.0/2, 1.0/(2**0.5), 1.0]:
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
    
    def transfer(self, contents, styles, inter_weight=1, projection_method='AdaIN'):
        content_features = self.attention_net.encode(contents)
        style_features = self.attention_net.encode(styles)

        content_hidden_feature = content_features[self.attention_net.perceptual_loss_layers[-1]]
        style_hidden_feature = style_features[self.attention_net.perceptual_loss_layers[-1]]

        projected_content_features, content_kernels, mean_content_features = utils.project_features(content_hidden_feature, projection_method)
        projected_style_features, style_kernels, mean_style_features = utils.project_features(style_hidden_feature, projection_method)

        content_attention_feature_map = self.attention_net.self_attn(projected_content_features)
        style_attention_feature_map = self.attention_net.self_attn(projected_style_features)
        projected_content_features = projected_content_features * content_attention_feature_map + projected_content_features
        
        content_attention_map = self.attention_filter(content_attention_feature_map)
        style_attention_map = self.attention_filter(style_attention_feature_map)

        multi_swapped_features = self.multi_scale_style_swap(projected_content_features, projected_style_features)
        seperate_strokes = []
        for feature in multi_swapped_features:
            reconstructed_features = utils.reconstruct_features(feature, style_kernels, mean_style_features, projection_method)
            temp = self.attention_net.decode(reconstructed_features, style_features)
            temp = utils.batch_mean_image_subtraction(temp)
            seperate_strokes.append(temp)

        fused_features, centroids = self.multi_stroke_fusion(multi_swapped_features, content_attention_map, theta=50.0, mode='softmax')

        fused_features = inter_weight * fused_features + (1 - inter_weight) * projected_content_features

        reconstructed_features = utils.reconstruct_features(fused_features, style_kernels, mean_style_features, projection_method)
        output = self.attention_net.decode(reconstructed_features, style_features)
        output = utils.batch_mean_image_subtraction(output)
        return output, content_attention_map, style_attention_map, centroids, seperate_strokes

class AAMS_test():
    def __init__(self, attention_net):
        self.attention_net = attention_net
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

    def multi_scale_style_swap(self, content_feature, style_feature, kernel_size=3, stride=1):
        c_shape = content_feature.shape
        s_shape = style_feature.shape
        assert (c_shape[1] == s_shape[1]) # same along the channel dimension
        
        combined_feature_maps = []
        for beta in [1.0/2, 1.0/(2**0.5), 1.0]:
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
    
    def transfer(self, contents, styles, inter_weight=1, projection_method='AdaIN'):
        content_features = self.attention_net.encode(contents)
        style_features = self.attention_net.encode(styles)

        content_hidden_feature = content_features[self.attention_net.perceptual_loss_layers[-1]]
        style_hidden_feature = style_features[self.attention_net.perceptual_loss_layers[-1]]

        projected_content_features, content_kernels, mean_content_features = utils.project_features(content_hidden_feature, projection_method)
        projected_style_features, style_kernels, mean_style_features = utils.project_features(style_hidden_feature, projection_method)

        projected_content_features_attn, _, _ = utils.project_features(content_hidden_feature, 'ZCA')
        projected_style_features_attn, _, _ = utils.project_features(style_hidden_feature, 'ZCA')

        content_attention_feature_map = self.attention_net.self_attn(projected_content_features_attn)
        style_attention_feature_map = self.attention_net.self_attn(projected_style_features_attn) 
        # content_attention_feature_map = self.attention_net.self_attn(projected_content_features)
        # style_attention_feature_map = self.attention_net.self_attn(projected_style_features)

        projected_content_features = projected_content_features * content_attention_feature_map + projected_content_features
        content_attention_map = self.attention_filter(content_attention_feature_map)
        style_attention_map = self.attention_filter(style_attention_feature_map)

        multi_swapped_features = self.multi_scale_style_swap(projected_content_features, projected_style_features)
        seperate_strokes = []
        for feature in multi_swapped_features:
            reconstructed_features = utils.reconstruct_features(feature, style_kernels, mean_style_features, projection_method)
            temp = self.attention_net.decode(reconstructed_features, style_features)
            temp = utils.batch_mean_image_subtraction(temp)
            seperate_strokes.append(temp)

        fused_features, centroids = self.multi_stroke_fusion(multi_swapped_features, content_attention_map, theta=50.0, mode='softmax')

        fused_features = inter_weight * fused_features + (1 - inter_weight) * projected_content_features

        reconstructed_features = utils.reconstruct_features(fused_features, style_kernels, mean_style_features, projection_method)
        output = self.attention_net.decode(reconstructed_features, style_features)
        output = utils.batch_mean_image_subtraction(output)
        return output, content_attention_map, style_attention_map, centroids, seperate_strokes, self.attention_net.decode(projected_content_features, style_features)            

def extract_patches(feature, patch_size, stride):
    ph, pw = patch_size
    sh, sw = stride
    
    # padding the feature
    padh = (ph - 1) // 2
    padw = (pw - 1) // 2
    padding_size = (padw, padw, padh, padh)
    feature = F.pad(feature, padding_size, 'constant', 0)

    # extract patches
    patches = feature.unfold(2, ph, sh).unfold(3, pw, sw)
    patches = patches.contiguous().view(*patches.size()[:-2], -1)
    
    return patches

class StyleDecorator(torch.nn.Module):
    
    def __init__(self):
        super(StyleDecorator, self).__init__()

    def kernel_normalize(self, kernel, k=3):
        b, ch, h, w, kk = kernel.size()
        
        # calc kernel norm
        kernel = kernel.view(b, ch, h*w, kk).transpose(2, 1)
        kernel_norm = torch.norm(kernel.contiguous().view(b, h*w, ch*kk), p=2, dim=2, keepdim=True)
        
        # kernel reshape
        kernel = kernel.view(b, h*w, ch, k, k)
        kernel_norm = kernel_norm.view(b, h*w, 1, 1, 1)
        
        return kernel, kernel_norm

    def conv2d_with_style_kernels(self, features, kernels, patch_size, deconv_flag=False):
        output = list()
        b, c, h, w = features.size()
        
        # padding
        pad = (patch_size - 1) // 2
        padding_size = (pad, pad, pad, pad)
        
        # batch-wise convolutions with style kernels
        for feature, kernel in zip(features, kernels):
            feature = F.pad(feature.unsqueeze(0), padding_size, 'constant', 0)
                
            if deconv_flag:
                padding_size = patch_size - 1
                output.append(F.conv_transpose2d(feature, kernel, padding=padding_size))
            else:
                output.append(F.conv2d(feature, kernel))
        
        return torch.cat(output, dim=0)
        
    def binarize_patch_score(self, features):
        outputs= list()
        
        # batch-wise operation
        for feature in features:
            matching_indices = torch.argmax(feature, dim=0)
            one_hot_mask = torch.zeros_like(feature)

            h, w = matching_indices.size()
            for i in range(h):
                for j in range(w):
                    ind = matching_indices[i, j]
                    one_hot_mask[ind, i, j] = 1
            outputs.append(one_hot_mask.unsqueeze(0))
            
        return torch.cat(outputs, dim=0)
   
    def norm_deconvolution(self, h, w, patch_size):
        mask = torch.ones((h, w))
        fullmask = torch.zeros((h + patch_size - 1, w + patch_size - 1))

        for i in range(patch_size):
            for j in range(patch_size):
                pad = (i, patch_size - i - 1, j, patch_size - j - 1)
                padded_mask = F.pad(mask, pad, 'constant', 0)
                fullmask += padded_mask

        pad_width = (patch_size - 1) // 2
        if pad_width == 0:
            deconv_norm = fullmask
        else:
            deconv_norm = fullmask[pad_width:-pad_width, pad_width:-pad_width]

        return deconv_norm.view(1, 1, h, w)

    def reassemble_feature(self, normalized_content_feature, normalized_style_feature, patch_size, patch_stride):
        # get patches of style feature
        style_kernel = extract_patches(normalized_style_feature, [patch_size, patch_size], [patch_stride, patch_stride])

        # kernel normalize
        style_kernel, kernel_norm = self.kernel_normalize(style_kernel, patch_size)
        
        # convolution with style kernel(patch wise convolution)
        patch_score = self.conv2d_with_style_kernels(normalized_content_feature, style_kernel/kernel_norm, patch_size)
        
        # binarization
        binarized = self.binarize_patch_score(patch_score)
        
        # deconv norm
        deconv_norm = self.norm_deconvolution(h=binarized.size(2), w=binarized.size(3), patch_size=patch_size)

        # deconvolution
        output = self.conv2d_with_style_kernels(binarized, style_kernel, patch_size, deconv_flag=True)
        
        return output/deconv_norm.type_as(output)

    def forward(self, content_feature, style_feature, style_strength=1.0, patch_size=3, patch_stride=1): 
        # 1-1. content feature projection
        normalized_content_feature = whitening(content_feature)

        # 1-2. style feature projection
        normalized_style_feature = whitening(style_feature)

        # 2. swap content and style features
        reassembled_feature = self.reassemble_feature(normalized_content_feature, normalized_style_feature, patch_size=patch_size, patch_stride=patch_stride)

        # 3. reconstruction feature with style mean and covariance matrix
        stylized_feature = coloring(reassembled_feature, style_feature)

        # 4. content and style interpolation
        result_feature = (1 - style_strength) * content_feature + style_strength * stylized_feature
        
        return result_feature

class AvatarNet():
    def __init__(self, attention_net):
        self.attention_net = attention_net
        self.encode = attention_net.get_encoder()
        self.decorator = StyleDecorator()

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

    def transfer(self, contents, styles, inter_weight=1, projection_method='ZCA'):
        content_features = self.attention_net.encode(contents)
        style_features = self.attention_net.encode(styles)

        projected_content_features = content_features[self.attention_net.perceptual_loss_layers[-1]]
        projected_style_features = style_features[self.attention_net.perceptual_loss_layers[-1]]

        reconstructed_features = self.decorator(projected_content_features, projected_style_features)

        output = self.attention_net.decode(reconstructed_features, style_features)
        output = utils.batch_mean_image_subtraction(output)
        print("haha")
        return output         