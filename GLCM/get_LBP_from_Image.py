# -*- coding: utf-8 -*-
    # get_LBP_from_Image.py
    # 2015-7-7
    # github: https://github.com/michael92ht
    #__author__ = 'huangtao'

# import the necessary packages
import math
import numpy as np
import cv2
from PIL import Image
from pylab import*

class LBP:
    def __init__(self):
        #revolve_map为旋转不变模式的36种特征值从小到大进行序列化编号得到的字典
        self.revolve_map={0:0,1:1,3:2,5:3,7:4,9:5,11:6,13:7,15:8,17:9,19:10,21:11,23:12,
                          25:13,27:14,29:15,31:16,37:17,39:18,43:19,45:20,47:21,51:22,53:23,55:24,
                          59:25,61:26,63:27,85:28,87:29,91:30,95:31,111:32,119:33,127:34,255:35}
        #uniform_map为等价模式的58种特征值从小到大进行序列化编号得到的字典
        self.uniform_map={0:0,1:1,2:2,3:3,4:4,6:5,7:6,8:7,12:8,
                          14:9,15:10,16:11,24:12,28:13,30:14,31:15,32:16,
                          48:17,56:18,60:19,62:20,63:21,64:22,96:23,112:24,
                          120:25,124:26,126:27,127:28,128:29,129:30,131:31,135:32,
                          143:33,159:34,191:35,192:36,193:37,195:38,199:39,207:40,
                          223:41,224:42,225:43,227:44,231:45,239:46,240:47,241:48,
                          243:49,247:50,248:51,249:52,251:53,252:54,253:55,254:56,
                          255:57}


     #将图像载入，并转化为灰度图，获取图像灰度图的像素信息
    def describe(self,image):
        image_array=np.array(Image.open(image).convert('L'))
        return image_array

    #图像的LBP原始特征计算算法：将图像指定位置的像素与周围8个像素比较
    #比中心像素大的点赋值为1，比中心像素小的赋值为0，返回得到的二进制序列
    def calute_basic_lbp(self,image_array,i,j):
        sum=[]
        if image_array[i-1,j-1]>image_array[i,j]:
            sum.append(1)
        else:
            sum.append(0)
        if image_array[i-1,j]>image_array[i,j]:
            sum.append(1)
        else:
            sum.append(0)
        if image_array[i-1,j+1]>image_array[i,j]:
            sum.append(1)
        else:
            sum.append(0)
        if image_array[i,j-1]>image_array[i,j]:
            sum.append(1)
        else:
            sum.append(0)
        if image_array[i,j+1]>image_array[i,j]:
            sum.append(1)
        else:
            sum.append(0)
        if image_array[i+1,j-1]>image_array[i,j]:
            sum.append(1)
        else:
            sum.append(0)
        if image_array[i+1,j]>image_array[i,j]:
            sum.append(1)
        else:
            sum.append(0)
        if image_array[i+1,j+1]>image_array[i,j]:
            sum.append(1)
        else:
            sum.append(0)
        return sum

    # 双线性滤波
    @staticmethod
    def bilinear_interpolation(r, c, img):
        x1, y1 = int(r), int(c)
        x2, y2 = math.ceil(r), math.ceil(c)

        r1 = (x2 - r) / (x2 - x1) * img[x1, y1] + (r - x1) / (x2 - x1) * img[x2, y1]
        r2 = (x2 - r) / (x2 - x1) * img[x1, y2] + (r - x1) / (x2 - x1) * img[x2, y2]

        return (y2 - c) / (y2 - y1) * r1 + (c - y1) / (y2 - y1) * r2

    #图像的LBP cicular特征计算方法:将图像指定位置的像素与周围8个方向像素值比较
    #(8个方向像素值一般是插值取出，取决于圆的半径，像素值是插值取出的，因为根据
    #圆的半径与方向角计算出来的像素坐标一般不是整数值，故而均经过通用的插值过程)
    #8个方向像素值比中心像素大的点赋值为1，比中心像素小的赋值为0，返回得到的二进制序列
    def calute_circular_lbp(self,image_array,i,j):
        P = 8 # number of pixels
        R = 1 # radius

        center = image_array[i,j]
        pixels = []
        for point in range(0, P):
            r = i + R * math.cos(2 * math.pi * point / P)
            c = j - R * math.sin(2 * math.pi * point / P)
            if r < 0 or c < 0:
                pixels.append(0)
                continue
            if int(r) == r:
                if int(c) != c:
                    c1 = int(c)
                    c2 = math.ceil(c)
                    w1 = (c2 - c) / (c2 - c1)
                    w2 = (c - c1) / (c2 - c1)

                    #像素值插值
                    pixels.append(int((w1 * image_array[int(r), int(c)] + \
                                   w2 * image_array[int(r), math.ceil(c)]) / (w1 + w2)))
                else:
                    pixels.append(image_array[int(r), int(c)])

            elif int(c) == c:
                r1 = int(r)
                r2 = math.ceil(r)
                w1 = (r2 - r) / (r2 - r1)
                w2 = (r - r1) / (r2 - r1)

                #像素值插值
                pixels.append((w1 * image_array[int(r), int(c)] + \
                               w2 * image_array[math.ceil(r), int(c)]) / (w1 + w2))
            else:
                #像素值插值
                pixels.append(self.bilinear_interpolation(r, c, image_array))

        # 8个方向像素值与中心值进行比较
        out = []
        for a in pixels:
            if a >= center:
                out.append(1)
            else:
                out.append(0)
        return out

    #获取二进制序列进行不断环形旋转得到新的二进制序列的最小十进制值
    def get_min_for_revolve(self,arr):
        values=[]
        circle=arr
        circle.extend(arr)
        for i in range(0,8):
            j=0
            sum=0
            bit_num=0
            while j<8:
                sum+=circle[i+j]<<bit_num
                bit_num+=1
                j+=1
            values.append(sum)
        return min(values)

    #获取值r的二进制中1的位数
    def calc_sum(self,r):
        num=0
        while(r):
            r&=(r-1)
            num+=1
        return num

    #获取图像的LBP原始模式特征
    def lbp_basic(self,image_array):
        basic_array=np.zeros(image_array.shape, np.uint8)
        width=image_array.shape[0]
        height=image_array.shape[1]
        for i in range(1,width-1):
            for j in range(1,height-1):
                sum=self.calute_basic_lbp(image_array,i,j)
                bit_num=0
                result=0
                for s in sum:
                    result+=s<<bit_num
                    bit_num+=1
                basic_array[i,j]=result
        return basic_array

    #获取图像的LBP圆特征
    def lbp_circular(self,image_array):
        circular_array=np.zeros(image_array.shape, np.uint8)
        width=image_array.shape[0]
        height=image_array.shape[1]
        for i in range(1,width-1):
            for j in range(1,height-1):
                sum=self.calute_circular_lbp(image_array,i,j)
                bit_num=0
                result=0
                for s in sum:
                    result+=s<<bit_num
                    bit_num+=1
                circular_array[i,j]=result
        return circular_array

   #获取图像的LBP旋转不变模式特征
    def lbp_revolve(self,image_array):
        revolve_array=np.zeros(image_array.shape, np.uint8)
        width=image_array.shape[0]
        height=image_array.shape[1]
        for i in range(1,width-1):
            for j in range(1,height-1):
                sum=self.calute_basic_lbp(image_array,i,j)
                revolve_key=self.get_min_for_revolve(sum)
                revolve_array[i,j]=self.revolve_map[revolve_key]
        return revolve_array

  #获取图像的LBP等价模式特征
    def lbp_uniform(self,image_array):
        uniform_array=np.zeros(image_array.shape, np.uint8)
        basic_array=self.lbp_basic(image_array)
        width=image_array.shape[0]
        height=image_array.shape[1]

        for i in range(1,width-1):
            for j in range(1,height-1):
                 k= basic_array[i,j]<<1
                 if k>255:
                     k=k-255
                 xor=basic_array[i,j]^k
                 num=self.calc_sum(xor)
                 if num<=2:
                     uniform_array[i,j]=self.uniform_map[basic_array[i,j]]
                 else:
                     uniform_array[i,j]=58
        return uniform_array

    #获取图像的LBP旋转不变等价模式特征
    def lbp_revolve_uniform(self,image_array):
        uniform_revolve_array=np.zeros(image_array.shape, np.uint8)
        basic_array=self.lbp_basic(image_array)
        width=image_array.shape[0]
        height=image_array.shape[1]
        for i in range(1,width-1):
            for j in range(1,height-1):
                 k= basic_array[i,j]<<1
                 if k>255:
                     k=k-255
                 xor=basic_array[i,j]^k
                 num=self.calc_sum(xor)
                 if num<=2:
                     uniform_revolve_array[i,j]=self.calc_sum(basic_array[i,j])
                 else:
                     uniform_revolve_array[i,j]=9
        return uniform_revolve_array

    #绘制指定维数和范围的图像灰度归一化统计直方图
    def show_hist(self,img_array,im_bins,im_range):
        hist = cv2.calcHist([img_array],[0],None,im_bins,im_range)
        hist = cv2.normalize(hist,None).flatten()
        plt.plot(hist,color = 'r')
        plt.xlim(im_range)
        plt.show()

    #绘制图像原始LBP特征的归一化统计直方图
    def show_basic_hist(self,img_array):
        self.show_hist(img_array,[256],[0,256])

    #绘制图像圆LBP特征的归一化统计直方图
    def show_circular_hist(self,img_array):
        self.show_hist(img_array,[256],[0,256])

    #绘制图像旋转不变LBP特征的归一化统计直方图
    def show_revolve_hist(self,img_array):
        self.show_hist(img_array,[36],[0,36])

    #绘制图像等价模式LBP特征的归一化统计直方图
    def show_uniform_hist(self,img_array):
        self.show_hist(img_array,[60],[0,60])

    #绘制图像旋转不变等价模式LBP特征的归一化统计直方图
    def show_revolve_uniform_hist(self,img_array):
        self.show_hist(img_array,[10],[0,10])

    #显示图像
    def show_image(self,image_array):
        cv2.imshow('Image',image_array)
        cv2.waitKey(0)

    #保存图像
    def save_image(self,image_array,save_loc='save_test.jpg'):
        cv2.imwrite(save_loc,image_array)
        cv2.waitKey(0)

if __name__ == '__main__':
    image = './test_0002_aligned.jpg'

    lbp=LBP()
    image_array=lbp.describe(image)

    #获取图像原始LBP特征，并显示其统计直方图与特征图像
    #basic_array=lbp.lbp_basic(image_array)
    #print(basic_array.shape)
    #print(type(basic_array))
    #lbp.show_basic_hist(basic_array)
    #lbp.show_image(basic_array)

    #获取图像circular LBP特征，并显示其统计直方图与特征图像
    # circular_array=lbp.lbp_circular(image_array)
    # print(circular_array.shape)
    # print(type(circular_array))
    # lbp.show_circular_hist(circular_array)
    # lbp.show_image(circular_array)
    # lbp.save_image(circular_array, save_loc='save_test.jpg')

    #获取图像旋转不变LBP特征，并显示其统计直方图与特征图像
    #revolve_array=lbp.lbp_revolve(image_array)
    #lbp.show_revolve_hist(revolve_array)
    #lbp.show_image(revolve_array)

    #获取图像等价模式LBP特征，并显示其统计直方图与特征图像
    uniform_array=lbp.lbp_uniform(image_array)
    lbp.show_uniform_hist(uniform_array)
    lbp.show_image(uniform_array)
    lbp.save_image(uniform_array, save_loc='save_test.jpg')

    #获取图像等价模式LBP特征，并显示其统计直方图与特征图像
    #resolve_uniform_array=lbp.lbp_revolve_uniform(image_array)
    #lbp.show_revolve_uniform_hist(resolve_uniform_array)
    #lbp.show_image(resolve_uniform_array)
