#ssim implementation from https://github.com/bonlime/pytorch-tools/blob/master/pytorch_tools/metrics/psnr.py
#
#vgg implementation as vgg_perceptual_loss.py from
#https://gist.github.com/alper111/8233cdb0414b4cb5853f2f730ab95a49#file-vgg_perceptual_loss-py

from math import log10
from skimage import io
import cv2
import numpy as np
import torch
import os
import sys
sys.path.insert(1, os.path.join("Deep-Red-Flash-main", "image_filtering"))
from vgg_perceptual_loss import VGGPerceptualLoss

def get_psnr(image1, image2):
    mse = ((image1 - image2)**2).mean(axis=None)
    psnr = 10 * log10(255 / mse.item())
    return psnr

def get_ssim(image1, image2):
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2

        image1 = image1.astype(np.float64)
        image2 = image2.astype(np.float64)
        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())

        mu1 = cv2.filter2D(image1, -1, window)[5:-5, 5:-5]  # valid
        mu2 = cv2.filter2D(image2, -1, window)[5:-5, 5:-5]
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = cv2.filter2D(image1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
        sigma2_sq = cv2.filter2D(image2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
        sigma12 = cv2.filter2D(image1 * image2, -1, window)[5:-5, 5:-5] - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean()

def get_vgg(image1, image2):
    image1 = torch.tensor(np.array([np.transpose(image1, (2,0,1))]) / 255).float()
    image2 = torch.tensor(np.array([np.transpose(image2, (2,0,1))]) / 255).float()
    vgg_loss = VGGPerceptualLoss()
    if torch.cuda.is_available():
        image1 = image1.cuda()
        image2 = image2.cuda()
        vgg_loss = vgg_loss.cuda()
    loss = vgg_loss.forward(image1,image2).item()
    return loss


def get_metrics(images1, images2):
    total_psnr = 0
    total_ssim = 0
    total_vgg = 0
    for i in range(len(images1)):
        image1 = io.imread(images1[i])
        image2 = io.imread(images2[i])
        total_psnr += get_psnr(image1, image2)
        total_ssim += get_ssim(image1, image2)
        total_vgg += get_vgg(image1, image2)
    return total_psnr / len(images1), total_ssim / len(images1), total_vgg / len(images1)


#usage example
outputfolderpath = os.path.join("Deep-Red-Flash-main", "image_filtering", "output")
goldfolderpath = os.path.join("Deep-Red-Flash-main", "image_filtering", "test")

images1 = []
images2 = []
for i in range(1, 6):
    images1.append(os.path.join(outputfolderpath, f"out_{i}.png"))
    images2.append(os.path.join(goldfolderpath, f"gt_{i}.bmp"))
psnr, ssim, vgg = get_metrics(images1, images2)
print(f"psnr: {psnr}, ssim: {ssim}, vgg: {vgg}")

