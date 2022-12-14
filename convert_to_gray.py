import cv2
import os

root_path = 'C:/Users/Administrator/Desktop/Git/yolov5-5.0/mydataset/images/'  # 要转灰度图像所在的文件夹路径
des_path = 'C:/Users/Administrator/Desktop/Git/yolov5-5.0/mydataset/gray/'  #灰度图像存储路径
filelist = os.listdir(root_path)  # 遍历文件夹
print(filelist)

for item in filelist: #批量处理照片
    if item.endswith('.jpg'):
        img = cv2.imread(root_path+item,cv2.IMREAD_GRAYSCALE)#读入照片，并转灰度
        cv2.imwrite(des_path+item,img)#保存图片
    print(item + '转灰度结束')
print('批量转灰度结束！')
