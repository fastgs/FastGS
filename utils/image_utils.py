#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch

def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2, mask=None):
    if mask is None:
        mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    else:
        mask_bin = (mask == 1.)
        mse = (((img1 - img2)[mask_bin]) ** 2).mean()
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def psnr_to_mse(psnr):
    return torch.exp(-0.1 * torch.log(torch.tensor(10.)) * psnr)

def avge(ssim, psnr, lpips):
    ssim = torch.sqrt(1 - ssim)
    psnr = psnr_to_mse(psnr)
    return torch.exp(torch.mean(torch.log(torch.tensor([ssim, psnr, lpips]))))
