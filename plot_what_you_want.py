import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy
from datetime import datetime

def plot_labcoat_num_Chinese(source_image,count,labcoat):
    if (isinstance(source_image, numpy.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    textSize = 20
    fontStyle = ImageFont.truetype("simsun.ttc", textSize, encoding="utf-8")
    # 绘制文本
    left = 10
    top = 10
    text = '总人数:' + str(count) + ' 穿工服人数:' + str(labcoat)
    textColor = (0, 225, 255)
    draw.text((left, top), text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(numpy.asarray(img), cv2.COLOR_RGB2BGR)

def plot_labcoat_num_English(source_image, count, labcoat):
    cv2.putText(
        source_image,
        'Toal:' + str(count) + ' Labcoat:' + str(labcoat),
        (10,20),#左上方坐标
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,#字体大小
        (233, 89, 119),
        1,#字体粗细
    )

def plot_craft_work_time_English(source_image,last_time_remark,last_time,frame_per_second):
    # current_time = datetime.now()
    # delta_time = (current_time - last_time).seconds
    delta_time = int((last_time-last_time_remark)/frame_per_second)
    start_minite = int(last_time_remark/frame_per_second/60)
    start_second = int(last_time_remark/frame_per_second % 60)
    #print('start_minite:',start_minite,'start_second:',start_second)
    start_time = str(start_minite) + ':' + str(start_second)

    cv2.putText(
        source_image,
        'craft start at:' + start_time + ' | ' + 'last:' + str(delta_time) + 'seconds',
        (40,20),#左上方坐标
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,#字体大小
        (65, 105, 225),
        2,#字体粗细
    )

def plot_craft_work_time_Chinese(source_image,last_time_remark,last_time,frame_per_second):
    if (isinstance(source_image, numpy.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    textSize = 30
    fontStyle = ImageFont.truetype("simsun.ttc", textSize, encoding="utf-8",)
    # 绘制文本
    left = 20
    top = 20
    delta_time = int((last_time-last_time_remark)/frame_per_second)
    start_minite = int(last_time_remark/frame_per_second/60)
    start_second = int(last_time_remark/frame_per_second % 60)
    start_time = str(start_minite) + ':' + str(start_second)

    text = '起始时间:' + start_time + ' 工艺时长:' + str(delta_time) + '秒'
    print(text)
    textColor = (135, 51, 36)
    draw.text((left, top), text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(numpy.asarray(img), cv2.COLOR_RGB2BGR)

