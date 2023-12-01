import os.path

import torch
import torch.nn as nn
from torch.autograd import Function
import math
import gauss_psf_cuda_new as gauss_psf_cuda
import cv2
import time
import numpy as np
import PIL.Image as Image
import signal
import scipy.signal as signal


class GaussPSFFunction(Function):
    @staticmethod
    def forward(ctx, input, weights, covariance, kernel_size=7):
        with torch.no_grad():
            x = torch.arange(kernel_size // 2,
                             -kernel_size // 2,
                             -1).view(kernel_size, 1).float().repeat(1, kernel_size).cuda()

            y = torch.arange(kernel_size // 2,
                             -kernel_size // 2,
                             -1).view(1, kernel_size).float().repeat(kernel_size, 1).cuda()
        # print("input",torch.max(input), torch.min(input))
        # print("weights",torch.max(weights), torch.min(weights))
        # print("covariance1",torch.max(covariance[:, 0, :, :]), torch.min(covariance[:, 0, :, :]))
        # print("covariance2", torch.max(covariance[:, 1, :, :]), torch.min(covariance[:, 1, :, :]))
        # print("covariance3", torch.max(covariance[:, 2, :, :]), torch.min(covariance[:, 2, :, :]))
        # print("covariance4", torch.max(covariance[:, 3, :, :]), torch.min(covariance[:, 3, :, :]))
        # print("covariance5", torch.max(covariance[:, 4, :, :]), torch.min(covariance[:, 4, :, :]))
        # print("covariance6", torch.max(covariance[:, 5, :, :]), torch.min(covariance[:, 5, :, :]))
        # print("covariance7", torch.max(covariance[:, 6, :, :]), torch.min(covariance[:, 6, :, :]))
        # print("covariance8", torch.max(covariance[:, 7, :, :]), torch.min(covariance[:, 7, :, :]))
        # print("covariance9", torch.max(covariance[:, 8, :, :]), torch.min(covariance[:, 8, :, :]))
        outputs, wsum = gauss_psf_cuda.forward(input, weights, covariance, x, y)
        # print("outputs", torch.max(outputs), torch.min(outputs))
        # ctx.save_for_backward(input, outputs, weights, covariance, wsum, x, y)
        return outputs

    @staticmethod
    def backward(ctx, grad):
        input, outputs, weights, covariance, wsum, x, y = ctx.saved_variables
        x = -x
        y = -y

        grad_input, grad_weights, grad_covar = gauss_psf_cuda.backward(grad.contiguous(), input, outputs, weights,
                                                                       covariance, wsum,
                                                                       x, y)
        grad_covar = grad_covar * 1e-9
        return grad_input, grad_weights, grad_covar, None


class GaussPSF(nn.Module):
    def __init__(self, kernel_size):
        super(GaussPSF, self).__init__()
        self.kernel_size = kernel_size

    def forward(self, image, weights, gauss_params):
        image = image.contiguous()
        weights = weights.expand_as(image).contiguous()
        covariance = torch.zeros([9, image.shape[1], image.shape[2]], dtype=torch.float)

        d = 2 * (1 - gauss_params[2] * gauss_params[2])
        a = 1 / (d * gauss_params[0] * gauss_params[0])
        b = -2 * gauss_params[2] / (d * gauss_params[0] * gauss_params[1])
        c = 1 / (d * gauss_params[1] * gauss_params[1])

        covariance[0:3, :, :] = a
        covariance[3:6, :, :] = b
        covariance[6:9, :, :] = c
        # print(covariance[0:9,0,0])
        covariance = covariance.cuda().contiguous()

        image=image.unsqueeze(0)
        weights = weights.unsqueeze(0)
        covariance = covariance.unsqueeze(0)
        return GaussPSFFunction.apply(image, weights, covariance, self.kernel_size)

class GaussianBlur(object):
    def __init__(self, kernel_size=7, sigma=20):
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.kernel = self.gaussian_kernel()

    def gaussian_kernel(self):
        kernel = np.zeros(shape=(self.kernel_size, self.kernel_size), dtype=np.float64)
        radius = self.kernel_size // 2
        for y in range(-radius, radius + 1):  # [-r, r]
            for x in range(-radius, radius + 1):
                # 二维高斯函数
                v = 1.0 / (2 * np.pi * self.sigma ** 2) * np.exp(-1.0 / (2 * self.sigma ** 2) * (x ** 2 + y ** 2))
                kernel[y + radius, x + radius] = v  # 高斯函数的x和y值 vs 高斯核的下标值
        kernel2 = kernel / np.sum(kernel)
        # kernel2=np.ones((self.kernel_size,self.kernel_size),np.float64)*1/(self.kernel_size*self.kernel_size)
        # print(kernel2)
        return kernel2

    def filter(self, img: Image.Image):
        img_arr = np.array(img)
        if len(img_arr.shape) == 2:
            new_arr = signal.convolve2d(img_arr, self.kernel, mode="same", boundary="symm")
        else:
            h, w, c = img_arr.shape
            new_arr = np.zeros(shape=(h, w, c), dtype=np.float64)
            for i in range(c):
                new_arr[..., i] = signal.convolve2d(img_arr[..., i], self.kernel, mode="same", boundary="symm")
        new_arr = np.array(new_arr, dtype=np.uint8)
        return new_arr

def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def ssim(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

def laplace_sharpen(input_image, c):
    '''
    拉普拉斯锐化
    :param input_image: 输入图像
    :param c: 锐化系数
    :return: 输出图像
    '''
    input_image_cp = np.copy(input_image)  # 输入图像的副本
    # 拉普拉斯滤波器
    laplace_filter = np.array([
        [1, 1, 1],
        [1, -8, 1],
        [1, 1, 1],
    ])
    input_image_cp = np.pad(input_image_cp, (1, 1), mode='constant', constant_values=0)  # 填充输入图像
    m, n = input_image_cp.shape  # 填充后的输入图像的尺寸
    output_image = np.copy(input_image_cp)  # 输出图像
    for i in range(1, m - 1):
        for j in range(1, n - 1):
            R = np.sum(laplace_filter * input_image_cp[i - 1:i + 2, j - 1:j + 2])  # 拉普拉斯滤波器响应
            output_image[i, j] = input_image_cp[i, j] + c * R
    output_image = output_image[1:m - 1, 1:n - 1]  # 裁剪
    return output_image

def allin3d_gauss(generator):
    num=100
    focus_nums=np.arange(0,20,1) # 0~19
    for focus_num in focus_nums:
        img_focus_num=np.arange(1,200,10)[focus_num]
        AIF_path = "/home/irvlab2/HHL/NeRF/Camera-NeRF/dataset/allin3d/{}/AIF/AIF.tif".format(num)
        focus_path = "/home/irvlab2/HHL/NeRF/Camera-NeRF/dataset/allin3d/{}/focus/foc_{}.bmp".format(num,img_focus_num)
        depth_path = "/home/irvlab2/HHL/NeRF/Camera-NeRF/dataset/allin3d/{}/depth/depth.npy".format(num)
        focus_distance =np.linspace(600,1200,20)[focus_num]
        # focus_distance =600
        ape = 800
        F = 50

        n = 4

        AIF = cv2.imread(AIF_path)
        AIF_r = cv2.resize(AIF, None, fx=1 / n, fy=1 / n)
        focus = cv2.imread(focus_path)
        focus_r = cv2.resize(focus, None, fx=1 / n, fy=1 / n)
        depth = np.load(depth_path)
        depth_r = cv2.resize(depth, None, fx=1 / n, fy=1 / n)

        time0 = time.time()
        CoC = abs(ape * F * (depth_r - focus_distance) / (focus_distance - F) / depth_r)
        time1 = time.time()
        # print(CoC)
        gauss_ps=[np.sqrt(1 / 2),np.sqrt(1 / 2),0]
        AIF_r0 = np.transpose(AIF_r, (2, 0, 1))
        clear_img=torch.Tensor(AIF_r0).float().cuda()
        weight=torch.Tensor(CoC).float().cuda()
        gauss_params=torch.Tensor(gauss_ps).float().cuda()
        time2=time.time()
        out_foucsed_img = generator(clear_img, weight, gauss_params)
        out_cuda_gaussian=np.transpose(out_foucsed_img.squeeze().cpu().numpy().astype('uint8'), (1, 2, 0))
        time3 = time.time()
        out_gaussian = GaussianBlur(kernel_size=19,sigma=1.5).filter(AIF_r)
        time4 = time.time()

        # print("CoC:", time1 - time0)
        print("cuda:",time3-time2)
        print("no cuda:", time4 - time3)
        print("psnr:",psnr(out_cuda_gaussian,focus_r))
        print("ssim:", calculate_ssim(out_cuda_gaussian, focus_r))

        cv2.imwrite("/home/irvlab2/HHL/NeRF/Camera-NeRF/ppt-fig/cuda_gaussian_{}.png".format(focus_num), out_cuda_gaussian)
        # cv2.imwrite("/home/irvlab2/HHL/NeRF/Camera-NeRF/ppt-fig/gaussian_19_1.5.png".format(num), out_gaussian)
        # cv2.imwrite("/home/irvlab2/HHL/NeRF/Camera-NeRF/ppt-fig/focus_{}.png".format(focus_num), focus_r)

        # cv2.imshow("cuda_gaussian",out_cuda_gaussian)
        # cv2.imshow("gaussian", out_gaussian)
        # cv2.imshow("org", focus_r)

        # cv2.waitKey(0)

def nerf_gauss(generator):
    fd_num=20
    focus_nums = np.arange(0, fd_num, 1)  # 0~19

    img_num = 3
    img_nums=np.arange(0, img_num, 1)

    # focus_distance = np.linspace(100, 1000, fd_num)
    focus_distance=np.load("/home/irvlab2/HHL/NeRF/Camera-NeRF/dataset/allin3d/100/focus_4/focus000（复件）/fd.npy")

    # focus_nums=[10]
    for img_num in img_nums:
        save_dir_path = "/home/irvlab2/HHL/NeRF/Camera-NeRF/dataset/allin3d/100/focus_4/focus{0:03d}".format(
            img_num)
        if not os.path.exists(save_dir_path):
            os.mkdir(save_dir_path)
        # np.save(os.path.join(save_dir_path,"fd.npy"),focus_distance/1000.)

        for focus_num in focus_nums:

            AIF_path = "/home/irvlab2/HHL/NeRF/Camera-NeRF/dataset/allin3d/100/AIF_4/image{0:03d}.png".format(img_num)

            depth_path = "/home/irvlab2/HHL/NeRF/Camera-NeRF/dataset/allin3d/100/depth_4/{0:03d}.npy".format(img_num)

            fd=focus_distance[focus_num]
            # focus_distance =600
            ape = 800
            F = 50
            #
            # n = 4

            AIF = cv2.imread(AIF_path)
            # AIF_r = cv2.resize(AIF, None, fx=1 / n, fy=1 / n)
            # focus = cv2.imread(focus_path)
            # focus_r = cv2.resize(focus, None, fx=1 / n, fy=1 / n)
            depth = np.load(depth_path)
            # depth = cv2.imread(depth_path)
            # depth_r = cv2.resize(depth, None, fx=1 / n, fy=1 / n)



            CoC = abs(ape * F * (depth - fd) / (fd - F) / depth)
            gauss_ps = [np.sqrt(1 / 2), np.sqrt(1 / 2), 0]
            AIF = np.transpose(AIF, (2, 0, 1))
            clear_img = torch.Tensor(AIF).float().cuda()
            weight = torch.Tensor(CoC).float().cuda()
            gauss_params = torch.Tensor(gauss_ps).float().cuda()

            out_foucsed_img = generator(clear_img, weight, gauss_params)
            out_cuda_gaussian = np.transpose(out_foucsed_img.squeeze().cpu().numpy().astype('uint8'), (1, 2, 0))

            # out_gaussian = GaussianBlur(kernel_size=19, sigma=1.5).filter(AIF_r)


            cv2.imwrite(os.path.join(save_dir_path,"focus{0:03d}.png".format(focus_num)),out_cuda_gaussian)
            # print(0)
            # cv2.imwrite("/home/irvlab2/HHL/NeRF/Camera-NeRF/ppt-fig/gaussian_19_1.5.png".format(num), out_gaussian)
            # cv2.imwrite("/home/irvlab2/HHL/NeRF/Camera-NeRF/ppt-fig/focus_{}.png".format(focus_num), focus_r)

            # cv2.imshow("aif", AIF)
            # cv2.imshow("depth", depth)
            # cv2.imshow("cuda_gaussian",out_cuda_gaussian)
            # cv2.imshow("gaussian", out_gaussian)
            # cv2.imshow("org", focus_r)

            # cv2.waitKey(0)


if __name__ == '__main__':
    import torch.backends.cudnn as cudnn

    cudnn.benchmark = True
    generator = GaussPSF(kernel_size=7)
    generator = torch.nn.DataParallel(generator)
    generator = generator.cuda()
    torch.set_num_threads(1)

    # allin3d_gauss(generator)
    nerf_gauss(generator)


