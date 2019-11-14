# -*- coding:utf-8 -*-
"""
Blur and Noise Detection.
"""
import os
import cv2
import sys
import time
import numpy as np
from skimage import filters
from configparser import ConfigParser

import xlwt
# import shutil
# from configparser import ConfigParser # Config file.

def get_file_realpath(src, *tar):
    '''
    返回图片文件的路径
    Parameters
    ----------
    src: sring. 图片root目录
    tar: 图片类型文件后缀. [".jpg",“.png”]
    '''
    for root, _, files in os.walk(src):
        for fn in files:
            fn_name, fn_ext = os.path.splitext(fn)
            if fn_ext.lower() not in tar:
                continue
            
            yield os.path.join(root, fn)

def Write_Excel(input_path, image_name_list, Evaluation_list):
    '''
    将图片质量的检测结果保存到 PictureFilter.xls 文件中
    ----------------
    input_path: str. PictureFilter.xls文件保存的路径
    Image_name: list. 检测图片名字列表
    Image_Evaluation: list. 图片质量评估结果列表
    '''
    workbook = xlwt.Workbook(encoding="ascii")
    worksheet = workbook.add_sheet("Picture Filter")
    
    for i in range(len(image_name_list)):
        worksheet.write(i, 0, image_name_list[i])        
        worksheet.write(i, 1, Evaluation_list[i])        

    workbook.save(os.path.join(input_path, "PictureFilter.xls"))

def Blur_Noise_Tenengrad(image_path):
    '''
    用Sobel算子处理后图像的平均灰度值,作为质量评估参数之一
    ----------------
    input_path: str. 图片的真实路径
    return: folat. Sobel算子处理图像后的平均灰度值
    '''
    image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Sobel算子处理图像，需处理的图像为灰度图像
    tmp = filters.sobel(gray_image)    
    # 计算图像的平均灰度值
    score = np.sum(tmp**2) 
    score = np.sqrt(score)
    return score 
   
def Blur_Noise_Laplacian(image_path):
    '''
    用Laplacian算子处理图像后的方差值作为质量评估参数之一
    ----------------
    input_path: str. 图片的真实路径
    return: folat. Laplacian算子处理图像后的方差值
    '''
    image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
    img2gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)    
    # Sobel算子处理图像，需处理的图像为灰度图像
    # cv2.Laplacian(src, ddepth)  src: 图像 ddepth:图像深度
    score  = cv2.Laplacian(img2gray,cv2.CV_64F).var()
    return score 

def Asses_Image_Quality(blur, noise, class1, class2, class3, cf):
    '''
    结合Blur_Noise_Tenengrad()和Blur_Noise_Laplacian()得分值，返回检测结果
    blur: folat. Blur_Noise_Tenengrad()的得分值
    noise: folat. Blur_Noise_Laplacian()的得分值 
    class1: str. 图像质量类别值1
    class2: str. 图像质量类别值2
    class3: str. 图像质量类别值3
    return: str. 图像质量评估结果类别值             
    '''
    #评估准侧：1) blur1和 blur2都小于某一阈值时为模糊图片
    #2) blur1和 blur2都大于某一阈值时为清晰图片
    #3) blur1和 blur2都介于阈值之间为噪声图片
    #4) 其余图片待质检
    if blur < cf.getint("image_quality","blur_threshold1") and noise < cf.getint("image_quality","noise_threshold1"):              
        Evaluation = class1
    elif blur > cf.getint("image_quality","blur_threshold2") and noise > cf.getint("image_quality","noise_threshold2"):
        Evaluation = class2
    elif (cf.getint("image_quality","blur_threshold1")-10) < blur < (cf.getint("image_quality","blur_threshold2")-15) and (cf.getint("image_quality","noise_threshold1")+5)< noise < (cf.getint("image_quality","noise_threshold2")+5):
        Evaluation = class3
    else:
        Evaluation = '待质检'
        
    return Evaluation
      
if __name__ == '__main__':
     
    '''
    IQA parameters!
    sys.argv[0] is *.py/exe
    '''
   
    # 从命令行获取输入路径和配置文件
    input_path = sys.argv[1]
    config = sys.argv[2] # Config file.
    
    # 读取配置文件内容
    # cf.getint(), return int. If config file has Chinese words, then use cf = read(configencoding="utf-8").
    cf = ConfigParser()
    cf.read(config,encoding="utf-8") 

    # IQA检测结果类别
    class1 = "模糊"
    class2 = "清晰"
    class3 = "有噪声"
    
    # 得到输入路径下的所有图片
    imageNames = get_file_realpath(input_path, *[".jpg", ".png", ".bmp", ".jpeg"])
    start_time1 = time.time()
    #保存图片名称和检测结果
    image_name_list=[]
    Evaluation_list=[]
    # 迭代所有图片，检测得出结果
    for image_path in imageNames:
        # 得到图片的清晰度和噪声
        blur_noise_Tenengrad = Blur_Noise_Tenengrad(image_path)
        blur_noise_Laplacian = Blur_Noise_Laplacian(image_path)
        # 判断图片的质量类别
        Evaluation = Asses_Image_Quality(blur_noise_Tenengrad, blur_noise_Laplacian, class1, class2,class3, cf)

        image_name = os.path.basename(image_path)
        print("{} {} {}.  Evaluation:{}  ".format(image_path, str("%.4f"%blur_noise_Tenengrad), str(blur_noise_Laplacian), Evaluation))        
        image_name_list.append(image_name)
        Evaluation_list.append(Evaluation)
    
    # 将结果写在输入路径下的execl文件中
    Write_Excel(input_path, image_name_list, Evaluation_list)
         
    print("Toal Time cost is: {}".format(str(time.time() - start_time1)))

