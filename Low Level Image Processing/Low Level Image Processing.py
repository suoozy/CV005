"""
# 使用opencv-python实现对图片的基本操作：
# Part1： 读取图片并显示
# Part2： change color
# Part3： gamma correction
# Part4： image crop
# Part5： histogram
# Part6： similarity transform
# Part7： Affine transform
# Part8: Perspective transform
 """

import cv2
import numpy as np
import matplotlib.pyplot as plt


#====================== PART1：读取图片并显示=====================
# 彩色图像
img = cv2.imread('lenna-color.png', 1)  # 0为灰度图，1为彩色三通道
print(img)
cv2.imshow('lenna', img)

key = cv2.waitKey()    # 等待键盘输入显示毫秒数，否则无法显示
if key == 27:    # 如果输入ESC则退出
    cv2.destroyAllWindows()

# 灰度图像
img_gray = cv2.imread('lenna.bmp', 0)
cv2.imshow('lenna', img_gray)

key = cv2.waitKey()    # 等待键盘输入显示毫秒数，否则无法显示
if key == 27:    # 如果输入ESC则退出
    cv2.destroyAllWindows()

# 使用matplotlib显示图像
plt.imshow(img)
plt.show()           # opencv的彩色三通道的存储顺序是BGR
                     # matplotlib的彩色三通道的存储顺序是RGB
                     # pillow的彩色三通道的存储顺序是RGB

B,G,R = cv2.split(img)   # 通道拆分函数
img_new = cv2.merge([R,G,B])  # 通道合并函数

plt.imshow(img_new)
plt.show()

#======================== PART2：change color=======================
img = cv2.imread('lenna-color.png', 1)
B, G, R = cv2.split(img)

const = 100
B[B > 255-const] = 255
B[B <= 255-const] = B[B <= 255-const] + const

img_new = cv2.merge([R,G,B])

plt.imshow(img_new)
plt.show()

# change color though YUV space
img = cv2.imread('dog.jpg', 1)
img_YUV = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)  # 颜色空间转换函数

cv2.imshow('img_YUV', img_YUV)
key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()

#=================== PART3：Gamma Correction伽马校正================
img = cv2.imread('dog.jpg', 1)  # 0为灰度图，1为彩色三通道
cv2.imshow('dog', img)

key = cv2.waitKey()    # 等待键盘输入显示毫秒数，否则无法显示
if key == 27:    # 如果输入ESC则退出
    cv2.destroyAllWindows()

def gamma_adjust(img, gamma = 1.0):
    """
    伽马校正函数：使得图片的亮度存储更符合人眼特性
    parament:
      img: 伽马校正的图片
      gamma: 伽马函数的参数
    return:
      image_new: 伽马校正后的图片
    """
    table = []
    for i in range(0,256):
        table.append((i/255)**gamma*255)  # 伽马函数
    table = np.array(table).astype('uint8')   # 转换table中的数据类型
    img_gamma = cv2.LUT(img, table)  # look up table查找表，表示两个图片之间转换的映射关系

    return img_gamma

img_gamma = gamma_adjust(img, 2)
cv2.imshow('dog-gamma', img_gamma)

key = cv2.waitKey()    # 等待键盘输入显示毫秒数，否则无法显示
if key == 27:    # 如果输入ESC则退出
    cv2.destroyAllWindows()

#=================== PART4：image crop图像切片================
img = cv2.imread('dog.jpg', 1)  # 0为灰度图，1为彩色三通道
print(img.shape)

img_crop = img[100:300, 200:400]

cv2.imshow('dog-crop', img_crop)

key = cv2.waitKey()    # 等待键盘输入显示毫秒数，否则无法显示
if key == 27:    # 如果输入ESC则退出
    cv2.destroyAllWindows()

#====================== PART5：histogram==================
img_gray = cv2.imread('dog.jpg', 0)  # 0为灰度图，1为彩色三通道
print(img_gray.shape)

plt.hist(img_gray.flatten(), bins=256, range=[0,255], facecolor= 'g')
         # data:必选参数，绘图数据
         # bins:直方图的长条形数目，可选项，默认为10
         # normed:是否将得到的直方图向量归一化，默认为0，代表不归一化，显示频数。normed=1，表示归一化，显示频率。
         # facecolor:长条形的颜色
         # edgecolor:长条形边框的颜色
         # alpha:透明度
plt.show()

# 直方图均衡化
img_eq = cv2.equalizeHist(img_gray)
plt.hist(img_eq.flatten(), bins=256, range=[0,255])
plt.show()
cv2.imshow('dog-hist', img_eq)

key = cv2.waitKey()    # 等待键盘输入显示毫秒数，否则无法显示
if key == 27:    # 如果输入ESC则退出
    cv2.destroyAllWindows()

#====================== PART6：similarity transform==================
#scaling
img = cv2.imread('dog.jpg', 1)  # 0为灰度图，1为彩色三通道
img_scaling = cv2.resize(img, dsize=None, fx=2, fy=2)

cv2.imshow('dog-scaling', img_scaling)

key = cv2.waitKey()    # 等待键盘输入显示毫秒数，否则无法显示
if key == 27:    # 如果输入ESC则退出
    cv2.destroyAllWindows()

#translation
img = cv2.imread('dog.jpg', 1)  # 0为灰度图，1为彩色三通道
M = np.float32([[1,0,100],[0,1,50]])   # 平移矩阵，x方向平移100，y方向平移50
img_translation = cv2.warpAffine(img, M, dsize=None)

cv2.imshow('dog-translation', img_translation)

key = cv2.waitKey()    # 等待键盘输入显示毫秒数，否则无法显示
if key == 27:    # 如果输入ESC则退出
    cv2.destroyAllWindows()

#rotation
img = cv2.imread('dog.jpg', 1)  # 0为灰度图，1为彩色三通道
M = cv2.getRotationMatrix2D(center=(img.shape[0]/2, img.shape[1]/2), angle=30, scale=0.5)
                            # 通过中心点和旋转角度以及缩放比例来确定变换矩阵
img_rotation = cv2.warpAffine(img, M, dsize=None)

cv2.imshow('dog-rotation', img_rotation)

key = cv2.waitKey()    # 等待键盘输入显示毫秒数，否则无法显示
if key == 27:    # 如果输入ESC则退出
    cv2.destroyAllWindows()

#====================== PART7：Affine Transform==================
img = cv2.imread('dog.jpg', 1)  # 0为灰度图，1为彩色三通道
cols, rows,ch = img.shape

pts1 = np.float32([[0,0],[cols-3,0],[0,rows-3]])
pts2 = np.float32([[cols*0.1,rows*0.2],[cols*0.9,rows*0.4],[cols*0.1,rows*0.9]])

M = cv2.getAffineTransform(pts1,pts2)     # 通过三对点来确定变换矩阵
img_affine = cv2.warpAffine(img, M, dsize=None)

cv2.imshow('dog-affine', img_affine)

key = cv2.waitKey()    # 等待键盘输入显示毫秒数，否则无法显示
if key == 27:    # 如果输入ESC则退出
    cv2.destroyAllWindows()

#====================== PART8：Perspective Transform==================
img = cv2.imread('dog.jpg', 1)  # 0为灰度图，1为彩色三通道
cols, rows,ch = img.shape

pts1 = np.float32([[0,0],[cols-3,0],[0,rows-3],[cols-3,rows-3]])
pts2 = np.float32([[cols*0.1,rows*0.2],[cols*0.9,rows*0.4],[cols*0.1,rows*0.9],[cols*0.9,rows*0.9]])

M_warp = cv2.getPerspectiveTransform(pts1,pts2)

img_warp = cv2.warpPerspective(img, M_warp, dsize=None)

cv2.imshow('dog-warp', img_warp)

key = cv2.waitKey()    # 等待键盘输入显示毫秒数，否则无法显示
if key == 27:    # 如果输入ESC则退出
    cv2.destroyAllWindows()
