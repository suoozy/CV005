# -*- coding: utf-8 -*-
#------------------------------------
# @Time          : 2019.10.26
# @Author        : Su
#------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import random

def median_blur(img, size=3, padding = 'VALID'):
    if size%2 == 0:
        return print('Please choose an odd number!')

    channel1 = img[:, :, 0]
    channel2 = img[:, :, 1]
    channel3 = img[:, :, 2]

    if padding == 'VALID':
        x = y = size//2    # 窗口中心像素的坐标
        for j in range(size//2, img.shape[0]-size//2):
            for i in range(size//2, img.shape[1]-size//2):
                channel1[y+j, x+i] = np.median(channel1[y+j - size//2:y+j + size//2+1, x+i - size//2:x+i + size//2+1])
                channel2[y+j, x+i] = np.median(channel2[y+j - size//2:y+j + size//2+1, x+i - size//2:x+i + size//2+1])
                channel3[y+j, x+i] = np.median(channel3[y+j - size//2:y+j + size//2+1, x+i - size//2:x+i + size//2+1])
        img_median = np.dstack((channel1, channel2, channel3))

    if padding == 'SAME':
        channel1 = np.pad(channel1, size // 2)
        channel2 = np.pad(channel2, size // 2)
        channel3 = np.pad(channel3, size // 2)
        x = y = size//2
        for j in range(0, img.shape[0]):
            for i in range(0, img.shape[1]):
                channel1[y+j, x+i] = np.median(channel1[y+j - size//2:y+j + size//2+1, x+i - size//2:x+i + size//2+1])
                channel2[y+j, x+i] = np.median(channel2[y+j - size//2:y+j + size//2+1, x+i - size//2:x+i + size//2+1])
                channel3[y+j, x+i] = np.median(channel3[y+j - size//2:y+j + size//2+1, x+i - size//2:x+i + size//2+1])
        channel1 = channel1[size // 2:channel1.shape[0] - size // 2, size // 2:channel1.shape[1] - size // 2]
        channel2 = channel2[size // 2:channel2.shape[0] - size // 2, size // 2:channel2.shape[1] - size // 2]
        channel3 = channel3[size // 2:channel3.shape[0] - size // 2, size // 2:channel3.shape[1] - size // 2]
        img_median = np.dstack((channel1, channel2, channel3))

    return img_median


def sp_noise(image, prob=0.05):
    '''
    添加椒盐噪声
    prob:噪声比例
    '''
    output = np.zeros(image.shape, np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output


if __name__ == '__main__':
    img = plt.imread('lenna.jpg', 0)
    img_sp = sp_noise(img)
    img_median = median_blur(img_sp, size=5, padding='SAME')

    plt.subplot(1,3,1)
    plt.imshow(img)
    
    plt.subplot(1,3,2)
    plt.imshow(img_sp)

    plt.subplot(1,3,3)
    plt.imshow(img_median)
    plt.show()