# -*- coding: utf-8 -*-
# 载入所需库
import cv2
import numpy as np
import os
import time

YOLO_OBJECT = 'CAMERA' # IMAGE or VIDEO or CAMERA

'''
pathIn：原始图片的路径
pathOut：结果图片的路径
label_path：类别标签文件的路径
config_path：模型配置文件的路径
weights_path：模型权重文件的路径
confidence_thre：0-1，置信度（概率/打分）阈值，即保留概率大于这个值的边界框，默认为0.5
nms_thre：非极大值抑制的阈值，默认为0.3
jpg_quality：设定输出图片的质量，范围为0到100，默认为80，越大质量越好
'''
def yolo_detect_image(pathIn='',
                pathOut=None,
                label_path='./cfg/coco.names',
                config_path='./cfg/yolov3.cfg',
                weights_path='./cfg/yolov3.weights',
                confidence_thre=0.5,
                nms_thre=0.3,
                jpg_quality=100):
    
    # 加载类别标签文件
    LABELS = open(label_path).read().strip().split("\n")
    nclass = len(LABELS)

    # 为每个类别的边界框随机匹配相应颜色
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(nclass, 3), dtype='uint8')

    # 载入图片并获取其维度
    base_path = os.path.basename(pathIn)
    img = cv2.imread(pathIn)
    (H, W) = img.shape[:2]

    # 加载模型配置和权重文件
    print('从硬盘加载YOLO......')
    net = cv2.dnn.readNetFromDarknet(config_path, weights_path)

    # 获取YOLO输出层的名字
    ln = net.getLayerNames()
    # 下面两行如果第一行报错就执行第二行。原因是不同版本的OPENCV引起的输出格式不同
    #ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    ln = [ln[i- 1] for i in net.getUnconnectedOutLayers()]

    # 将图片构建成一个blob，设置图片尺寸，然后执行一次
    # YOLO前馈网络计算，最终获取边界框和相应概率
    blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()

    # 显示预测所花费时间
    print('YOLO模型花费 {:.2f} 秒来预测一张图片'.format(end - start))

    # 初始化边界框，置信度（概率）以及类别
    boxes = []
    confidences = []
    classIDs = []

    # 迭代每个输出层，总共三个
    for output in layerOutputs:
        # 迭代每个检测
        for detection in output:
            # 提取类别ID和置信度
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # 只保留置信度大于某值的边界框
            if confidence > confidence_thre:
                # 将边界框的坐标还原至与原图片相匹配，记住YOLO返回的是
                # 边界框的中心坐标以及边界框的宽度和高度
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # 计算边界框的左上角位置
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # 更新边界框，置信度（概率）以及类别
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # 使用非极大值抑制方法抑制弱、重叠边界框
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence_thre, nms_thre)

    # 确保至少一个边界框
    if len(idxs) > 0:
        # 迭代每个边界框
        for i in idxs.flatten():
            # 提取边界框的坐标
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # 绘制边界框以及在左上角添加类别标签和置信度
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            text = '{}: {:.3f}'.format(LABELS[classIDs[i]], confidences[i])
            (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(img, (x, y - text_h - baseline), (x + text_w, y), color, -1)
            cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    # 输出结果图片
    if pathOut is None:
        cv2.imwrite('with_box_' + base_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality])
    else:
        cv2.imwrite(pathOut, img, [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality])
    print('YOLO识别成功，图片生成成功')


'''---------------------------------------------'''

'''
pathIn：原始视频的路径
pathOut：结果视频的路径
label_path：类别标签文件的路径
config_path：模型配置文件的路径
weights_path：模型权重文件的路径
confidence_thre：0-1，置信度（概率/打分）阈值，即保留概率大于这个值的边界框，默认为0.5
nms_thre：非极大值抑制的阈值，默认为0.3
'''


def yolo_detect_video(pathIn='',
                       pathOut=None,
                       label_path='./cfg/coco.names',
                       config_path='./cfg/yolov3.cfg',
                       weights_path='./cfg/yolov3.weights',
                       confidence_thre=0.5,
                       nms_thre=0.3):
    net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
    print("[INFO] loading YOLO from disk...")  # # 可以打印下信息

    clicked = False

    def onMouse(event, x, y, flags, param):
        global clicked
        if event == cv2.EVENT_LBUTTONUP:
            clicked = True

    cameraCapture = cv2.VideoCapture(pathIn)  # 打开视频
    cv2.namedWindow('detected image')  # 给视频框命名
    cv2.setMouseCallback('detected image', onMouse)
    print('显示摄像头图像，点击鼠标左键或按任意键退出')
    success, frame = cameraCapture.read()

    # 保存视频设置
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    fps = 28
    savedPath = pathOut  # 保存地址
    videoWriter = cv2.VideoWriter(savedPath, fourcc, fps, (frame.shape[1], frame.shape[0]))  # 最后为视频图片的形状

    while success and cv2.waitKey(1) == -1 and not clicked:  # 当循环没结束，并且剩余的帧数大于零时进行下面的程序
        # 加载图片、转为blob格式、送入网络输入层
        blobImg = cv2.dnn.blobFromImage(frame, 1.0 / 255.0, (416, 416), None, True,
                                        False)  # # net需要的输入是blob格式的，用blobFromImage这个函数来转格式
        net.setInput(blobImg)  # # 调用setInput函数将图片送入输入层

        # 获取网络输出层信息（所有输出层的名字），设定并前向传播
        outInfo = net.getUnconnectedOutLayersNames()  # # 前面的yolov3架构也讲了，yolo在每个scale都有输出，outInfo是每个scale的名字信息，供net.forward使用
        start = time.time()
        layerOutputs = net.forward(outInfo)  # 得到各个输出层的、各个检测框等信息，是二维结构。
        end = time.time()
        print("[INFO] YOLO took {:.6f} seconds".format(end - start))  # # 可以打印下信息

        # 拿到图片尺寸
        (H, W) = frame.shape[:2]
        # 过滤layerOutputs
        # layerOutputs的第1维的元素内容: [center_x, center_y, width, height, objectness, N-class score data]
        # 过滤后的结果放入：
        boxes = []  # 所有边界框（各层结果放一起）
        confidences = []  # 所有置信度
        classIDs = []  # 所有分类ID

        # # 1）过滤掉置信度低的框框
        for out in layerOutputs:  # 各个输出层
            for detection in out:  # 各个框框
                # 拿到置信度
                scores = detection[5:]  # 各个类别的置信度
                classID = np.argmax(scores)  # 最高置信度的id即为分类id
                confidence = scores[classID]  # 拿到置信度

                # 根据置信度筛查
                if confidence > confidence_thre:
                    box = detection[0:4] * np.array([W, H, W, H])  # 将边界框放会图片尺寸
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        # # 2）应用非最大值抑制(non-maxima suppression，nms)进一步筛掉
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence_thre, nms_thre)  # boxes中，保留的box的索引index存入idxs
        # 得到labels列表
        with open(label_path, 'rt') as f:
            labels = f.read().rstrip('\n').split('\n')
        # 应用检测结果
        np.random.seed(42)
        COLORS = np.random.randint(0, 255, size=(len(labels), 3),
                                   dtype="uint8")  # 框框显示颜色，每一类有不同的颜色，每种颜色都是由RGB三个值组成的，所以size为(len(labels), 3)
        if len(idxs) > 0:
            for i in idxs.flatten():  # indxs是二维的，第0维是输出层，所以这里把它展平成1维
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

                color = [int(c) for c in COLORS[classIDs[i]]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)  # 线条粗细为2px
                text = "{}: {:.4f}".format(labels[classIDs[i]], confidences[i])
                cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color,
                            2)  # cv.FONT_HERSHEY_SIMPLEX字体风格、0.5字体大小、粗细2px
        cv2.imshow('detected image', frame)
        videoWriter.write(frame)  # 每次循环，写入该帧
        success, frame = cameraCapture.read()  # 摄像头获取下一帧
    cv2.destroyWindow('detected image')# 关闭窗口
    cameraCapture.release()# 释放资源
    videoWriter.release()  # 结束循环的时候释放保存
    print('YOLO识别成功，视频生成成功')

'''---------------------------------------------'''
# 摄像头对比视频只修改了 1.删除pathIn 2.修改cameraCapture = cv2.VideoCapture(pathIn) 为 cameraCapture = cv2.VideoCapture(0)
'''
pathIn：原始图片的路径
pathOut：结果图片的路径
label_path：类别标签文件的路径
config_path：模型配置文件的路径
weights_path：模型权重文件的路径
confidence_thre：0-1，置信度（概率/打分）阈值，即保留概率大于这个值的边界框，默认为0.5
nms_thre：非极大值抑制的阈值，默认为0.3
'''
def yolo_detect_camera( pathOut=None,
                        label_path='./cfg/coco.names',
                        config_path='./cfg/yolov3.cfg',
                        weights_path='./cfg/yolov3.weights',
                        confidence_thre=0.5,
                        nms_thre=0.3):
    
    net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
    print("[INFO] loading YOLO from disk...")  # # 可以打印下信息
    
    clicked = False
    
    
    def onMouse(event, x, y, flags, param):
        global clicked
        if event == cv2.EVENT_LBUTTONUP:
            clicked = True

    cameraCapture = cv2.VideoCapture(0)  # 打开编号为0的摄像头
    cv2.namedWindow('detected image')  # 给视频框命名
    cv2.setMouseCallback('detected image', onMouse)
    print ('显示摄像头图像，点击鼠标左键或按任意键退出')
    success, frame = cameraCapture.read()

    #保存视频设置
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    fps = 5
    savedPath = pathOut  # 保存地址
    videoWriter = cv2.VideoWriter(savedPath, fourcc, fps, (frame.shape[1], frame.shape[0]))  # 最后为视频图片的形状

    while success and cv2.waitKey(1) == -1 and not clicked:  # 当循环没结束，并且剩余的帧数大于零时进行下面的程序
        # 加载图片、转为blob格式、送入网络输入层
        blobImg = cv2.dnn.blobFromImage(frame, 1.0 / 255.0, (416, 416), None, True,
                                    False)  # # net需要的输入是blob格式的，用blobFromImage这个函数来转格式
        net.setInput(blobImg)  # # 调用setInput函数将图片送入输入层
    
        # 获取网络输出层信息（所有输出层的名字），设定并前向传播
        outInfo = net.getUnconnectedOutLayersNames()  # # 前面的yolov3架构也讲了，yolo在每个scale都有输出，outInfo是每个scale的名字信息，供net.forward使用
        start = time.time()
        layerOutputs = net.forward(outInfo)  # 得到各个输出层的、各个检测框等信息，是二维结构。
        end = time.time()
        print("[INFO] YOLO took {:.6f} seconds".format(end - start))  # # 可以打印下信息
    
        # 拿到图片尺寸
        (H, W) = frame.shape[:2]
        # 过滤layerOutputs
        # layerOutputs的第1维的元素内容: [center_x, center_y, width, height, objectness, N-class score data]
        # 过滤后的结果放入：
        boxes = []  # 所有边界框（各层结果放一起）
        confidences = []  # 所有置信度
        classIDs = []  # 所有分类ID
    
        # # 1）过滤掉置信度低的框框
        for out in layerOutputs:  # 各个输出层
            for detection in out:  # 各个框框
                # 拿到置信度
                scores = detection[5:]  # 各个类别的置信度
                classID = np.argmax(scores)  # 最高置信度的id即为分类id
                confidence = scores[classID]  # 拿到置信度
    
                # 根据置信度筛查
                if confidence > confidence_thre:
                    box = detection[0:4] * np.array([W, H, W, H])  # 将边界框放会图片尺寸
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)
    
        # # 2）应用非最大值抑制(non-maxima suppression，nms)进一步筛掉
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence_thre, nms_thre)  # boxes中，保留的box的索引index存入idxs
        # 得到labels列表
        with open(label_path, 'rt') as f:
            labels = f.read().rstrip('\n').split('\n')
        # 应用检测结果
        np.random.seed(42)
        COLORS = np.random.randint(0, 255, size=(len(labels), 3),
                                      dtype="uint8")  # 框框显示颜色，每一类有不同的颜色，每种颜色都是由RGB三个值组成的，所以size为(len(labels), 3)
        if len(idxs) > 0:
            for i in idxs.flatten():  # indxs是二维的，第0维是输出层，所以这里把它展平成1维
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
    
                color = [int(c) for c in COLORS[classIDs[i]]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)  # 线条粗细为2px
                text = "{}: {:.4f}".format(labels[classIDs[i]], confidences[i])
                cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color,
                            2)  # cv.FONT_HERSHEY_SIMPLEX字体风格、0.5字体大小、粗细2px
        cv2.imshow('detected image', frame)
        videoWriter.write(frame)  # 每次循环，写入该帧
        success, frame = cameraCapture.read()  # 摄像头获取下一帧
    cv2.destroyWindow('detected image')# 关闭窗口
    cameraCapture.release()# 释放资源
    videoWriter.release()  # 结束循环的时候释放保存
    print('YOLO识别成功，视频生成成功')

if __name__ == '__main__':
    # 方式1：图像检测
    if YOLO_OBJECT == 'IMAGE':
        pathIn = './image/test5_input.png'
        pathOut = './image/test5_output.png'
        yolo_detect_image(pathIn, pathOut)
    #方式2：视频检测
    elif YOLO_OBJECT == 'VIDEO':
        pathIn = './video/input_video.mp4'
        pathOut = './video/output_video.avi'
        yolo_detect_video(pathIn, pathOut)
    #方式3：摄像头检测
    elif YOLO_OBJECT == 'CAMERA':
        pathOut = './video/output_video.avi'
        yolo_detect_camera(pathOut)