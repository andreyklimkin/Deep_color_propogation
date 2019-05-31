import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


import sys
sys.path.append("src")
from resent import resnet101
from skimage.transform import resize
from skimage import io, img_as_float
from skimage.color import rgb2lab, lab2rgb
from tqdm import tqdm


class NewSepConv(nn.Module):
    def __init__(self):
        super(NewSepConv, self).__init__()
       
    def forward(self, imgs, vers, hors):
        b, c, H, W = imgs.size()
        _, s, _, _ = vers.size()
        
        all_kernels = vers.permute(0,2,3,1).contiguous().view(b, H, W, s, 1) @ hors.permute(0,2,3,1).contiguous().view(b,
                                                                                                                       H,
                                                                                                                       W,
                                                                                                                       1, 
                                                                                                                       s)
        all_kernels = all_kernels.view(b, 1, H, W, s, s)
        imgs = torch.nn.ReplicationPad2d([8,8,8,8]) (imgs)
        all_patches = []
        for i in range(H):
            for j in range(W):
                all_patches.append(imgs[:,:,i:i+17, j:j+17].contiguous().view(b, c, 17, 17, 1))
        all_patches = torch.cat(all_patches, dim=-1).view(b, c, 17, 17, H, W).permute(0,1,4,5,2,3).contiguous()
        #print (124,all_patches.size(), all_kernels.size())
        return (all_patches * all_kernels).sum(dim=-1).sum(dim=-1)
    

class TripleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3,padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3,padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3,padding=1),
            nn.ReLU(),
        )
    def forward(self, x):
        return self.conv_layers(x)


## LOCAL TRANSFER NET    
class LocalTransferNet(nn.Module):
    def __init__(self):

        super().__init__()
        
        self.conv_32 = TripleConv(6, 32) #/1
        self.conv_down_32_64 = nn.Sequential(nn.AvgPool2d(2), TripleConv(32, 64)) #/2
        self.conv_down_64_128 = nn.Sequential(nn.AvgPool2d(2), TripleConv(64, 128)) #/4
        self.conv_down_128_256 = nn.Sequential(nn.AvgPool2d(2), TripleConv(128, 256)) #/8
        self.conv_down_256_512 = nn.Sequential(nn.AvgPool2d(2), TripleConv(256, 512)) #/16
        self.conv_down_512_512 = nn.Sequential(nn.AvgPool2d(2), TripleConv(512, 512)) #/32
        
        self.upsample = nn.Upsample(scale_factor=2) #/16
        self.conv_up_512_256 = nn.Sequential(TripleConv(512, 256), nn.Upsample(scale_factor=2)) #/8
        self.conv_up_256_128 = nn.Sequential(TripleConv(256, 128), nn.Upsample(scale_factor=2)) #/4
        self.conv_up_128_64 = nn.Sequential(TripleConv(128, 64), nn.Upsample(scale_factor=2)) #/2
        
        self.image_h_filter = nn.Sequential(TripleConv(64, 17), nn.Upsample(scale_factor=2)) #/1
        self.image_w_filter = nn.Sequential(TripleConv(64, 17), nn.Upsample(scale_factor=2)) #/1
#         self.image2_h_filter = nn.Sequential(TripleConv(64, 17), nn.Upsample(scale_factor=2)) #/1
#         self.image2_w_filter = nn.Sequential(TripleConv(64, 17), nn.Upsample(scale_factor=2)) #/1
        
        self.image_sepconv = NewSepConv()
        #self.image2_sepconv = NewSepConv()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, G_prev, G_cur, I_prev):
        
        x = self.conv_32(torch.cat((G_prev, G_cur), dim=1))
        
        
        x_down_32_64 = self.conv_down_32_64(x) #/2
        x_down_64_128 = self.conv_down_64_128(x_down_32_64) #/4
        x_down_128_256 = self.conv_down_128_256(x_down_64_128) #/8
        x_down_256_512 = self.conv_down_256_512(x_down_128_256) #/16
        x_down_512_512 = self.conv_down_512_512(x_down_256_512) #/32
        
        x_bottle = self.upsample(x_down_512_512) #/16
        
        x_up = x_bottle + x_down_256_512  
        x_up_512_256 = self.conv_up_512_256(x_up) + x_down_128_256 #/8
        x_up_256_128 = self.conv_up_256_128(x_up_512_256) + x_down_64_128 #/4
        x_up_128_64 = self.conv_up_128_64(x_up_256_128) + x_down_32_64 #/2
        
        image_kh = self.softmax(self.image_h_filter(x_up_128_64))
        image_kw = self.softmax(self.image_w_filter(x_up_128_64))
        
        #image2_kh = self.image2_h_filter(x_up_128_64)
        #image2_kw = self.image2_w_filter(x_up_128_64)
        
        image_sepconv_proceed = self.image_sepconv(I_prev, image_kh, image_kw)
        #image2_sepconv_proceed = self.image2_sepconv(I_prev, image2_kh, image2_kw)
        

        return image_sepconv_proceed
    
    

class RefinementNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(9, 64, 3, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=2, dilation=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=4, dilation=4),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, 3, padding=1, dilation=1),
        )
    def forward(self, x):
        return self.conv_layers(x)


# ##GLOBAL TRANSFER NET
# class ResNet101Extractor:
#     def __init__(self):
#         self.net = resnet101(pretrained=True)
#         self.net.eval()
    
#     def get_features(self, img):
#         tensor = torch.Tensor(img.transpose(2, 0, 1)[None, ...])
#         x = self.net.conv1(tensor)
#         x = self.net.bn1(x)
#         x = self.net.relu(x)
#         x_fine = self.net.layer1(x)
#         x = self.net.layer2(x_fine)
#         x_coarse = self.net.layer3(x)
#         return (x_fine[0].cpu().detach().numpy().transpose(1, 2, 0).astype(np.double),
#                 x_coarse[0].cpu().detach().numpy().transpose(1, 2, 0).astype(np.double))  

class ResNet101Extractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = resnet101(pretrained=True).double()
        self.net.eval()
    
    def forward(self, tensor):
        #tensor = torch.Tensor(img.transpose(0, 3, 1, 2)).cuda()
        x = self.net.conv1(tensor)
        x = self.net.bn1(x)
        x = self.net.relu(x)
        x_fine = self.net.layer1(x)
        x = self.net.layer2(x_fine)
        x_coarse = self.net.layer3(x)
        return (x_fine.cpu().detach().numpy().transpose(0, 2, 3, 1).astype(np.double),
                x_coarse.cpu().detach().numpy().transpose(0, 2, 3, 1).astype(np.double))


class GlobalTransferer(nn.Module):
    def __init__(self, downscale_coef=4):
        super().__init__()
        self.downscale_coef = downscale_coef
        self.feature_extractor_net = ResNet101Extractor().cuda()
    
    def get_coarse_match_old(self, F_G1, F_Gk):
        closest = torch.zeros(F_G1.shape[0], F_G1.shape[1], 2)
        F_G1_tensor = torch.Tensor(F_G1.transpose(2, 0, 1))[None, ...]
        F_Gk_tensor = torch.Tensor(F_Gk)

        f2_g1 = torch.sum(F_G1_tensor ** 2, 1)[:, ...]
        f2_gk = torch.sum(F_Gk_tensor ** 2, -1)

        for i in range(F_Gk.shape[0]):
            fgk_row = F_Gk_tensor[i, :, :]
            fgk_row_g1 = F.conv2d(F_G1_tensor, fgk_row[:, :, None, None])[0]

            f2_gk_row = f2_gk[i, ...].reshape(-1, 1, 1)
            row_mse = ((f2_gk_row -2 * fgk_row_g1 + f2_g1).reshape(fgk_row_g1.shape[0], -1))

            mse_row_closest_indxs = torch.argmin(row_mse, -1)

            closest[i, :, 0] = mse_row_closest_indxs // F_G1.shape[1]
            closest[i, :, 1] = mse_row_closest_indxs % F_G1.shape[1]
        return closest.cpu().detach().numpy().astype(np.uint8)
    
    def get_coarse_match_new(self, F_G1_batch, F_Gk_batch):
        B, H, W, K = F_G1_batch.shape
        closest = torch.zeros(B, H, W, 2).cuda()
        
        F_G1_batch_tensor = torch.Tensor(F_G1_batch).cuda()
        F_Gk_batch_tensor = torch.Tensor(F_Gk_batch).cuda()

        for b in range(B):
            F_G1_tensor = F_G1_batch_tensor[b]
            F_Gk_tensor = F_Gk_batch_tensor[b]
            
            weights = F_G1_tensor.reshape(H * W, K, 1, 1)
            input_tensor = F_Gk_tensor.permute(2, 0, 1).reshape(1, K, H, W)


            f2_g1 = torch.sum(F_G1_tensor ** 2, -1)[None, None, ...].reshape(H * W, 1, 1)
            f2_gk = torch.sum(F_Gk_tensor ** 2, -1)[None, None, ...]
            F_gk_g1 = F.conv2d(input_tensor, weights)[0]

            pathces_mse = (f2_gk + f2_g1 - 2 * F_gk_g1)[0]
            closes_patches_indxs = torch.argmin(pathces_mse, 0)
            closest[b, ..., 0] = closes_patches_indxs // W
            closest[b, ..., 1] = closes_patches_indxs % W

        return closest.cpu().detach().numpy().astype(np.uint8)

    
    def get_fine_match_new(self, F_G1_batch, F_Gk_batch, coarse_match_batch):
        downscale_coef = self.downscale_coef
        batch_size, H, W = F_Gk_batch.shape[:3]

        result_match = np.zeros((batch_size, H, W, 2))
        for batch_element in range(batch_size):
            F_G1, F_Gk, coarse_match = F_G1_batch[batch_element], F_Gk_batch[batch_element], coarse_match_batch[batch_element]
            for I in range(0, H // downscale_coef):
                for J in range(0, W // downscale_coef):
                    Fk_ij_features = F_Gk[downscale_coef * I : downscale_coef * (I + 1),
                                          downscale_coef * J : downscale_coef * (J + 1), :].reshape(downscale_coef ** 2, 
                                                                                                    1, -1)

                    t = max(I - 1, 0)
                    b = min(I + 2, H // downscale_coef)
                    l = max(J - 1, 0)
                    r = min(J + 2, W // downscale_coef)
                    closest_IJ_indeces = coarse_match[t : b, l : r]
                    closest_IJ_indeces = closest_IJ_indeces.transpose(2, 0, 1).reshape(2, (b - t) * (r - l), 
                                                                                       1).astype(np.int)


                    row_indxs_shifts = np.repeat(np.arange(downscale_coef), axis=0, 
                                                 repeats=downscale_coef).reshape(1,downscale_coef ** 2)
                    col_indxs_shifts = np.repeat(np.arange(downscale_coef).reshape(1,downscale_coef), axis=0, 
                                                 repeats=downscale_coef).reshape(-1).reshape(1,  downscale_coef ** 2)

                    res_indxs = np.zeros((2, (b - t) * (r - l)
                                          , downscale_coef ** 2))

                    res_indxs[0] = closest_IJ_indeces[0] * downscale_coef + row_indxs_shifts
                    res_indxs[1] = closest_IJ_indeces[1] * downscale_coef + col_indxs_shifts

                    flatten_res_indxs = (res_indxs[0].reshape(-1).astype(np.int), res_indxs[1].reshape(-1).astype(np.int))



                    F1_ij_candidates = F_G1[flatten_res_indxs].reshape(len(flatten_res_indxs[0]),-1)

                    feature_distance = np.sum((F1_ij_candidates - Fk_ij_features) ** 2, -1)
                    closest_features_indeces = np.argmin(feature_distance, -1)

                    result_IJ_indeces = np.array((flatten_res_indxs[0][closest_features_indeces],  
                                                  flatten_res_indxs[1][closest_features_indeces])).reshape(2,
                                                                                                           downscale_coef,
                                                                                                           downscale_coef)
                    result_match[batch_element, downscale_coef * I : downscale_coef * (I + 1),
                                 downscale_coef * J : downscale_coef * (J + 1)] = result_IJ_indeces.transpose(1, 2, 0)
        return result_match

    def copy_colors(self, rgb_I0, gray_Ik, closest_idxs):
        
        result = np.zeros_like(rgb_I0)
        lab_I0 = rgb2lab(rgb_I0)
        lab_Ik = rgb2lab(gray_Ik)
        
        for i in range(rgb_I0.shape[0]):
            for j in range(rgb_I0.shape[1]):
                lab_Ik[i, j, 1:] = lab_I0[closest_idxs[i, j, 0], closest_idxs[i, j, 1], 1:]
        return lab2rgb(lab_Ik)
    
    def get_features(self, img):
        tensor = torch.tensor(img.transpose(0, 3, 1, 2), dtype=torch.double, requires_grad=False).cuda()
        return self.feature_extractor_net(tensor)
    
    def forward(self, G_1, G_k, I_1):
        F_g1_fine, F_g1_coarse = self.get_features(G_1)
        F_gk_fine, F_gk_coarse = self.get_features(G_k)
        closest_coarse = self.get_coarse_match_new(F_g1_coarse, F_gk_coarse)
        closest_fine = self.get_fine_match_new(F_g1_fine, F_gk_fine, closest_coarse)
        result = []
        for i in range(len(closest_fine)): 
            I_k = self.copy_colors(I_1[i], G_k[i], closest_fine[i].astype(np.uint8))
            result.append(I_k)
        return np.array(result)
    