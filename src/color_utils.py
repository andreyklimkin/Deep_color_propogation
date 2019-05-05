import numpy as np
import numbers
import torch

from torch import nn
from torch.nn import functional as F

mat_rgb2xyz = torch.Tensor(np.array([[0.412453, 0.357580, 0.180423],
                     [0.212671, 0.715160, 0.072169],
                     [0.019334, 0.119193, 0.950227]]).T).double()

mat_xyz2rgb = torch.Tensor(np.array([[ 3.24048134, -1.53715152, -0.49853633],
       [-0.96925495,  1.87599   ,  0.04155593],
       [ 0.05564664, -0.20404134,  1.05731107]])).transpose(0, 1).double()


def clip_lab(lab):
    L = lab[..., 0].clamp(-100, 100)
    A = lab[..., 1].clamp(-128, 128)
    B = lab[..., 2].clamp(-128, 128)

    return torch.cat([L[..., None], A[..., None], B[..., None]], -1)


def rgb2xyz_torch(rgb_image, use_gpu=True):

    rgb_image = rgb_image
    mask = (rgb_image > 0.04045).double()
    
    cond_res = ((rgb_image + 0.055) / 1.055) ** 2.4
    uncond_res = rgb_image / 12.92
    res = cond_res * mask + (1 - mask) * uncond_res
    
    mat = mat_rgb2xyz.cuda() if use_gpu else mat_rgb2xyz
    return res @ mat


def xyz2rgb_torch(xyz_, use_gpu=True):
    c = xyz_
    mat = mat_xyz2rgb.cuda() if use_gpu else mat_xyz2rgb
    v = c @ mat
    
    mask = (v > 0.0031308).double()
    
    cond_res = 1.055 * torch.pow(v * mask + 1e-10, 1. / 2.4) - 0.055
    uncond_res = v * (1 - mask) * 12.92
    
    res = cond_res * mask + uncond_res

    return res.clamp(0, 1)


def xyz2lab_torch(xyz_image, is_cuda=True):
    multiplier = torch.Tensor([0.95047, 1., 1.08883]).double()
    n = xyz_image / (multiplier.cuda() if is_cuda else multiplier)
    
    mask = (n > 0.008856).double()
    
    cond_res = torch.pow(n.clamp(0, 1) + 1e-10, 0.333333)
    uncond_res = n * 7.787 + 16.0 / 116.0

    res = cond_res * mask + (1 - mask) * uncond_res
    x = res[..., 0][..., None]
    y = res[..., 1][..., None]
    z = res[..., 2][..., None]
    
    result = torch.cat([116 * y - 16, 500 * (x - y), 200 * (y - z)], -1)
    result = result.cuda() if is_cuda else result
    return result


def lab2xyz_torch(lab_image, is_cuda=True):
    x = lab_image[..., 0][..., None]
    y = lab_image[..., 1][..., None]
    z = lab_image[..., 2][..., None]
    
    fy = ( x + 16.0 ) / 116.0
    fx = y / 500. + fy
    fz = fy - z / 200.
    
    fz = torch.clamp(fz, min=0)
    
    f = torch.cat([fx, fy, fz], -1)
    
    mask = (f > 0.2068966).double()
    cond_res = torch.pow(f * mask, 3)
    uncond_res = (f - 16.0 / 116.0) / 7.787
    
    res = cond_res + (1 - mask) * uncond_res
    
    multiplier = torch.Tensor([0.95047, 1., 1.08883]).double()
    res = res * (multiplier.cuda() if is_cuda else multiplier)
    return res


def lab2rgb_torch(x, use_gpu=True):
    return xyz2rgb_torch(lab2xyz_torch(x, use_gpu), use_gpu)


def rgb2lab_torch(x, use_gpu=True):
    return xyz2lab_torch(rgb2xyz_torch(x, use_gpu), use_gpu)

