import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box,plot_one_box_under
from utils.torch_utils import select_device, load_classifier, time_synchronized

#from color_recognition.src.color_classification_image import plot_pre_color
from plot_what_you_want import plot_labcoat_num_English, plot_labcoat_num_Chinese, plot_craft_work_time_English, plot_craft_work_time_Chinese
import json



def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    # 是否需要保存图片,如果nosave(传入的参数)为false且source的结尾不是txt则保存图片
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))
    #判断source是视频/图像路径，还是url

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    # save_dir是保存运行结果的文件夹名，是通过递增的方式来命名的。第一次运行时路径是“runs\detect\exp”，第二次运行时路径是“runs\detect\exp1”
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    # 根据前面生成的路径创建文件夹

    # Initialize
    set_logging()
    device = select_device(opt.device)
    # select_device方法定义在utils.torch_utils模块中，返回值是torch.device对象，也就是推理时所使用的硬件资源。输入值如果是数字，表示GPU序号。也可是输入‘cpu’，表示使用CPU训练，默认是cpu
    half = device.type != 'cpu'  # half precision 半精度浮点数 only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    #是我们要加载的网络，其中weights参数就是输入时指定的权重文件（比如yolov5s.pt）
    stride = int(model.stride.max())  # model stride
    ## stride：推理时所用到的步长，默认为32， 大步长适合于大目标，小步长适合于小目标
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    # 将图片大小调整为步长的整数倍，也就是图像处理中32的源来
    if half:
        model.half()  # to FP16 精度调整

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam: # 使用摄像头作为输入
        view_img = check_imshow() # 检测cv2.imshow()方法是否可以执行，不能执行则抛出异常
        cudnn.benchmark = True  # set True to speed up constant image size inference
        # 该设置可以加速预测
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
        # 加载输入数据流
        # source：输入数据源 image_size 图片识别前被放缩的大小， stride：识别时的步长，
        # auto的作用可以看utils.augmentations.letterbox方法，它决定了是否需要将图片填充为正方形，如果auto=True则不需要
        # auto是v6.1的设置，但是参考图片处理，也是均要缩放并填充为32的整数倍

    else:
        cudnn.benchmark = True
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()

    start_time = 0
    end_time = 0
    last_time = 0
    last_time_remark = 0
    remain_time = 15  # 检测动作有效持续时间
    frame_count = -1  # 计数器

    frame_per_second = 30
    video_result = []
    current_tag = ''

    location_4101 = []
    remain_time_4101 = 0  # 用于绘制大约停针


    label_translate_dict = {
        "1002": "双手取裁片配对",
        "2002_2005": "调整两块裁片/抚平裁片",
        "3001": "单手取放裁片",
        "302": "翻转裁片调整位置",
        "3002": "双手取放裁片",
        "**": "车缝中",
        "4002": "电脑车倒针",
        "4004": "将裁片移至脚下",
        "4101": "大约停针",
        "4102": "准确停针",
    }

    for path, img, im0s, vid_cap in dataset:
        # 在dataset中，每次迭代的返回值是self.sources, img, img0, None, ''
        # path：文件路径（即source）
        # img: 处理后的输入图片列表（经过了放缩操作）
        # im0s: 源输入图片列表
        # vid_cap
        # s： 图片的基本信息，比如路径，大小
        last_time = last_time + 1
        end_time = end_time + 1

        img = torch.from_numpy(img).to(device)
        # 将图片放到指定设备(如GPU)上识别
        img = img.half() if half else img.float()  # uint8 to fp16/32
        # 把输入从整型转化为半精度/全精度浮点数。 根据是否采用cuda
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        #将图片归一化处理（这是图像表示方法的的规范，使用浮点数就要归一化）
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
            # 添加一个第0维。在pytorch的nn.Module的输入中，第0维是batch的大小，这里添加一个1。

        # Inference
        t1 = time_synchronized()# 获取当前时间
        pred = model(img, augment=opt.augment)[0]
        # 推理结果，pred保存的是所有的bound_box（矩形检测）的信息

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        # 执行非极大值抑制，返回值为过滤后的预测框
        # 执行非极大值抑制，返回值为过滤后的预测框
        # conf_thres： 置信度阈值
        # iou_thres： iou阈值
        # classes: 需要过滤的类（数字列表）
        # agnostic_nms： 标记class-agnostic或者使用class-specific方式。默认为class-agnostic
        # max_det: 检测框结果的最大数量 （v6.1中的设置）
        t2 = time_synchronized()# 获取当前时间

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image 每次迭代处理一张图片
            #enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在 for 循环当中
            #eg： seasons = ['Spring', 'Summer', 'Fall', 'Winter']
            #list(enumerate(seasons)) == [(0, 'Spring'), (1, 'Summer'), (2, 'Fall'), (3, 'Winter')]
            #det即待处理图片，数据类型为：<class 'torch.Tensor'>
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
                # frame：此次取的是第几张图片
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            # count = 0   #计数人数总数
            # labcoat = 0     #计数身穿实验服的数量

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg 推理结果图片保存的路径
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt 推理结果文本保存的路径
            s += '%gx%g ' % img.shape[2:]  # print string 显示推理前裁剪后的图像尺寸
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh 得到原图的宽和高


            # det <class 'torch.Tensor'> [xyxy, conf, cls] 1×6
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                # 将标注的bounding_box大小调整为和原图一致（因为训练时原图经过了放缩）
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    # if int(c): #对单张图片的检测结果进行计数，undress标签为0，身着实验服的标签为1,2,3,4，依此计数
                    #     labcoat += int(n)
                    #     count += int(n)
                    # else:
                    #     count += int(n)
                # 打印出所有的预测结果  比如1 person（检测出一个人）

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    #xyxy是个list
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        # line即写入labels中的数据
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')
                        # 写入对应的文件夹里，路径默认为“runs\detect\exp*\labels”

                    #im0为3n，shape为(高，宽，3(RGB))
                    if save_img or view_img:  # Add bbox to image
                        # 给图片添加推理后的bounding_box边框 #如：Labcoat 0.95
                        #label = f'{names[int(cls)]} {conf:.2f}'
                        # 绘制边框
                        craft_name = names[int(cls)]
                        label = f'{craft_name} '
                        #label = f'{craft_name} {conf:.2f}'
                        if current_tag != craft_name:
                            # 切换动作,current_tag为旧动作，craft_name为更新动作
                            # 旧动作持续时长若大于1s则保存json
                            if (last_time - last_time_remark) > remain_time:
                                # 保存旧动作的json
                                if craft_name != "4004":
                                    current_tag = craft_name
                                    craft_json_info = {"craft": None, "start_time": 0, "last_time": 0, }
                                    craft_json_info["craft"] = label_translate_dict.get(current_tag)
                                    temp = int(last_time_remark / frame_per_second)
                                    craft_json_info["start_time"] = temp
                                    craft_json_info["last_time"] = int(last_time / frame_per_second) - temp
                                    video_result.append(craft_json_info)
                                    print('craft_name:',craft_name,type(craft_name))
                                    if craft_name == "1002":
                                        #"4004": "将裁片移至脚下"
                                        craft_json_info2 = {"craft": "将裁片移至脚下", "start_time": 0, "last_time": 1, }
                                        craft_json_info2["start_time"] = int(last_time / frame_per_second)
                                        video_result.append(craft_json_info2)
                            # 更新当前动作
                            current_tag = craft_name
                            # 新动作开始
                            last_time_remark = last_time
                            # 刷新持续时间
                            frame_count = remain_time
                            if craft_name == "**":
                                location_4101 = xyxy
                                remain_time_4101 = 15
                            else:
                                remain_time_4101 = remain_time_4101 - 1
                                plot_one_box_under(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                                im0 = plot_craft_work_time_Chinese(im0,
                                                                   last_time_remark,
                                                                   last_time,
                                                                   frame_per_second)
                            # print("last_time:", last_time, "last_time_remark:", last_time_remark)
                            # 显示当前时间，并根据label，传入的开始时间，决定是否显示持续时间
                        else:
                            # 动作正常延续
                            if craft_name == "**":
                                location_4101 = xyxy
                                remain_time_4101 = 15
                            #如果要车缝正常识别显示，那就把这段删了
                            else:
                                frame_count = remain_time
                                plot_one_box_under(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                                im0 = plot_craft_work_time_Chinese(im0,
                                                                   last_time_remark,
                                                                   last_time,
                                                                   frame_per_second)
                            # print("last_time:", last_time, "last_time_remark:", last_time_remark)

            else:
                #无工艺、上一工艺动作结束处理
                frame_count = frame_count - 1
                remain_time_4101 = remain_time_4101 - 1
                if remain_time_4101 > 0 and current_tag =="**":
                    plot_one_box_under(location_4101, im0, label="4101", color=colors[int(9)], line_thickness=3)

                if frame_count == 0:
                    if craft_name != "4004":
                        craft_json_info = {"craft": None, "start_time": 0, "last_time": 0, }
                        craft_json_info["craft"] = label_translate_dict.get(craft_name)
                        temp = int(last_time_remark / frame_per_second)
                        craft_json_info["start_time"] = temp
                        craft_json_info["last_time"] = int(last_time / frame_per_second) - temp
                        temp_str = str(craft_json_info)
                        video_result.append(craft_json_info)
                        print('craft_name:', craft_name,type(craft_name))
                        if craft_name == "1002":
                            # "4004": "将裁片移至脚下"
                            craft_json_info2 = {"craft": "将裁片移至脚下", "start_time": 0, "last_time": 1, }
                            craft_json_info2["start_time"] = int(last_time / frame_per_second)
                            video_result.append(craft_json_info2)
                    # 要用到的量：last_time、frame_per_second、last_time_remark
                    # 在这写json
                    #若一直无新的1帧，则一直减少，直到int位数减为正

                        # 识别图片主体颜色
                        #plot_pre_color(im0,xyxy)
                        #im0 = plot_labcoat_num_Chinese(im0,count,labcoat)
            key = cv2.waitKey(1)
            if key == 27:
                exit()

            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')

            # Stream results
            if view_img:# 如果view_img为true,则显示该图片
                cv2.imshow(str(p), im0)# 预览图片
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img: # 如果save_img为true,则保存绘制完的图
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video 说明这张图片属于一段新的视频,需要重新创建视频文件
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)
                    # 以上的部分是保存视频文件

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    #total time compute
    craft_json_info = {"craft": None, "start_time": 0, "last_time": 0, }
    craft_json_info["craft"] = "工艺总时长"
    temp = int(start_time / frame_per_second)
    craft_json_info["start_time"] = temp
    craft_json_info["last_time"] = int(end_time / frame_per_second) - temp
    video_result.insert(0, craft_json_info)
    with open(str(save_dir) + '/results.txt', 'w') as obj:
        json.dump(video_result, obj, ensure_ascii=False)


    print(f'Done. ({time.time() - t0:.3f}s)')
    # 打印耗时

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='runs/train/norm_craft_yolo5s_50epochs/weights/last.pt', help='model.pt path(s)')  #即pt文件，
    parser.add_argument('--source', type=str, default='mydataset/5.mp4', help='source')
    # detect.py - -source
    # 0  # 网络摄像头
    # img.jpg  # 图像
    # vid.mp4  # 视频 写在source中不加引号
    # path /  # 文件夹
    # 'path/*.jpg'  # glob
    # 'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP 流
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.50, help='object confidence threshold')
    #执行度，大于default参数的概率都会进行标记
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    #NMS置信度，采用IOU，解决一目标多类问题，预测框与真实框的交集与并集的取值
    #越大，则容易将对于同一个物品的不同预测结果当成对多个物品的多个预测结果，导致一个物品出现了多个预测结果。
    #越小，则容易将对于多个物品的不同预测结果当成对同一个物品的不同预测结果，导致多个物品只出现了一个预测结果
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    #选择CUDA设备
    parser.add_argument('--view-img', action='store_true', help='display results')
    #命令行指定 --view-img参数可以在程序运行显示结果
    parser.add_argument('--save-txt', default=False, action='store_true', help='save results to *.txt')
    #识别结果保存成txt文件
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    #保存置信度
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    #不保存图片或视频
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    #parser.add_argument('--classes', nargs='+', type=int, default='0',help='filter by class: --class 0, or --class 0 2 3')
    #分类标记的筛选器，比如人就是0标签，在设置只要人或是其他标签
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    #增强nms，交集计算框
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    #增强检测参数，增强置信概率
    parser.add_argument('--update', action='store_true', help='update all models')
    #去掉不必要的更新和优化器等多余参数
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    #保存结果的默认位置
    parser.add_argument('--name', default='exp', help='save results to project/name')
    #输出结果的名字exp
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    #如果设置该参数的退出，文件的默认保存位置不会建立新的
    opt = parser.parse_args()
    print(opt)
    check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
