import os
import subprocess
import argparse
import csv
import os
import platform
import sys
import threading
import time
from pathlib import Path
import torch
import numpy as np


from django.shortcuts import render
from django.http import StreamingHttpResponse, HttpResponse, JsonResponse, HttpResponseBadRequest

import sys
import cv2
import json

import sendMessage
from regional import detect
from regional.detect import parse_opt
from userLogin.models import warn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ultralytics.utils.plotting import Annotator, colors, save_one_box

from regional.models.common import DetectMultiBackend
from regional.utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from regional.utils.general import (
    LOGGER,
    Profile,
    check_file,
    check_img_size,
    check_imshow,
    check_requirements,
    colorstr,
    cv2,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    strip_optimizer,
    xyxy2xywh,
)
from regional.utils.torch_utils import select_device, smart_inference_mode

thread_save = False
generate_frames_active = True
person_num_now = 0

global number
global times
# Create your views here.

def save(cam, name):
    print("in save")
    frame_width = int(cam.get(3))
    frame_height = int(cam.get(4))

    duration = 10
    t = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
    print(t)
    print(f'yolov5/warning/{name}-{t}.mp4')

    #设置编码器的格式为 H264-MPEG-4 AVC
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(f'yolov5/warning/{name}-{t}.mp4', fourcc, 10, (frame_width, frame_height))
    new_warn = warn(warningname=name, warningtime=t, savepath= f'yolov5/warning/{name}-{t}.mp4')
    new_warn.save()

    start_time = time.time()
    while True:
        ret, frame = cam.read()
        if ret:
            out.write(frame)
            if time.time() - start_time > duration:
                global thread_save
                thread_save = False
                print("finish save")
                break
        else:
            break


def save_video_thread(cam, name):
    save(cam, name)
def regional(request):

    return render(request, 'regional.html')

def video_feed(request):
    cap = cv2.VideoCapture("rtmp://116.62.245.164:1935/live/2")
    return StreamingHttpResponse(generate_frames1(cap), content_type='multipart/x-mixed-replace; boundary=frame')

def video_feed2(request):
    opt = parse_opt()
    return StreamingHttpResponse(generate_frames(**vars(opt)), content_type='multipart/x-mixed-replace; boundary=frame')


def generate_frames1(cap):
    global generate_frames_active
    global person_num_now

    while generate_frames_active:
        ret, frame = cap.read()
        if not ret:
            break
        _, jpeg = cv2.imencode('.jpeg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

    while True:
        ret, frame = cap.read()

        text = f'person: {person_num_now}'
        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        cv2.putText(frame, text, (640 - text_size[0] - 10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 255), 2)
        if not ret:
            break
        _, jpeg = cv2.imencode('.jpeg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')





def generate_frames(
    weights=ROOT / "yolov5s.pt",  # model path or triton URL
    source=ROOT / "data/images",  # file/dir/URL/glob/screen/0(webcam)
    data=ROOT / "data/coco128.yaml",  # fire.yaml path
    imgsz=(640, 640),  # inference size (height, width)
    conf_thres=0.25,  # confidence threshold
    iou_thres=0.45,  # NMS IOU threshold
    max_det=1000,  # maximum detections per image
    device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    view_img=False,  # show results
    save_txt=False,  # save results to *.txt
    save_csv=False,  # save results in CSV format
    save_conf=False,  # save confidences in --save-txt labels
    save_crop=False,  # save cropped prediction boxes
    nosave=False,  # do not save images/videos
    classes=None,  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms=False,  # class-agnostic NMS
    augment=False,  # augmented inference
    visualize=False,  # visualize features
    update=False,  # update all models
    project=ROOT / "runs/detect",  # save results to project/name
    name="exp",  # save results to project/name
    exist_ok=False,  # existing project/name ok, do not increment
    line_thickness=3,  # bounding box thickness (pixels)
    hide_labels=False,  # hide labels
    hide_conf=False,  # hide confidences
    half=False,  # use FP16 half-precision inference
    dnn=False,  # use OpenCV DNN for ONNX inference
    vid_stride=1,  # video frame-rate stride
    person_num=[0],# 视频中人数
    point=[] # 目前的坐标点
):
    number = 0
    times = 0
    source = str(source)
    save_img = not nosave and not source.endswith(".txt")  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
    webcam = source.isnumeric() or source.endswith(".streams") or (is_url and not is_file)
    screenshot = source.lower().startswith("screen")
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(device=device), Profile(device=device), Profile(device=device))
    for path, im, im0s, vid_cap, s in dataset:
        #添加点处理
        # mask for certain region
        #1,2,3,4 分别对应左上，右上，右下，左下四个点
        wl1 = point[0] / 640.0 #监测区域高度距离图片顶部比例
        hl1 = point[1] / 480.0 #监测区域高度距离图片左部比例
        wl2 = point[2] / 640.0 # 监测区域高度距离图片顶部比例
        hl2 = point[3] / 480.0 # 监测区域高度距离图片左部比例
        wl3 = point[4] / 640.0 # 监测区域高度距离图片顶部比例
        hl3 = point[5] / 480.0 # 监测区域高度距离图片左部比例
        wl4 = point[6] / 640.0 # 监测区域高度距离图片顶部比例
        hl4 = point[7] / 480.0 # 监测区域高度距离图片左部比例

        if webcam:
            for b in range(0,im.shape[0]):
                mask = np.zeros([im[b].shape[1], im[b].shape[2]], dtype=np.uint8)
                #mask[round(im[b].shape[1] * hl1):im[b].shape[1], round(im[b].shape[2] * wl1):im[b].shape[2]] = 255
                pts = np.array([[int(im[b].shape[2] * wl1), int(im[b].shape[1] * hl1)],  # pts1
                                [int(im[b].shape[2] * wl2), int(im[b].shape[1] * hl2)],  # pts2
                                [int(im[b].shape[2] * wl3), int(im[b].shape[1] * hl3)],  # pts3
                                [int(im[b].shape[2] * wl4), int(im[b].shape[1] * hl4)]], np.int32)
                mask = cv2.fillPoly(mask,[pts],(255,255,255))
                imgc = im[b].transpose((1, 2, 0))
                imgc = cv2.add(imgc, np.zeros(np.shape(imgc), dtype=np.uint8), mask=mask)
                #cv2.imshow('1',imgc) 展示蒙版的图片
                im[b] = imgc.transpose((2, 0, 1))

        else:
            mask = np.zeros([im.shape[1], im.shape[2]], dtype=np.uint8)
            #mask[round(img.shape[1] * hl1):img.shape[1], round(img.shape[2] * wl1):img.shape[2]] = 255
            pts = np.array([[int(im[b].shape[2] * wl1), int(im[b].shape[1] * hl1)],  # pts1
                                [int(im[b].shape[2] * wl2), int(im[b].shape[1] * hl2)],  # pts2
                                [int(im[b].shape[2] * wl3), int(im[b].shape[1] * hl3)],  # pts3
                                [int(im[b].shape[2] * wl4), int(im[b].shape[1] * hl4)]], np.int32)
            mask = cv2.fillPoly(mask, [pts], (255,255,255))
            im = im.transpose((1, 2, 0))
            im = cv2.add(im, np.zeros(np.shape(im), dtype=np.uint8), mask=mask)
            im = im.transpose((2, 0, 1))


        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            if model.xml and im.shape[0] > 1:
                ims = torch.chunk(im, im.shape[0], 0)

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            if model.xml and im.shape[0] > 1:
                pred = None
                for image in ims:
                    if pred is None:
                        pred = model(image, augment=augment, visualize=visualize).unsqueeze(0)
                    else:
                        pred = torch.cat((pred, model(image, augment=augment, visualize=visualize).unsqueeze(0)), dim=0)
                pred = [pred, None]
            else:
                pred = model(im, augment=augment, visualize=visualize)
        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Define the path for the CSV file
        csv_path = save_dir / "predictions.csv"

        # Create or append to the CSV file
        def write_to_csv(image_name, prediction, confidence):
            """Writes prediction data for an image to a CSV file, appending if the file exists."""
            data = {"Image Name": image_name, "Prediction": prediction, "Confidence": confidence}
            with open(csv_path, mode="a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=data.keys())
                if not csv_path.is_file():
                    writer.writeheader()
                writer.writerow(data)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f"{i}: "
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, "frame", 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / "labels" / p.stem) + ("" if dataset.mode == "image" else f"_{frame}")  # im.txt
            s += "%gx%g " % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                global person_num_now
                person_num_now = 0
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    if names[c] == 'person':
                        person_num_now += 1
                        print(time.localtime)

                    label = names[c] if hide_conf else f"{names[c]}"
                    confidence = float(conf)
                    confidence_str = f"{confidence:.2f}"

                    if save_csv:
                        write_to_csv(p.name, label, confidence_str)

                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f"{txt_path}.txt", "a") as f:
                            f.write(("%g " * len(line)).rstrip() % line + "\n")

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f"{names[c]} {conf:.2f}")
                        annotator.box_label(xyxy, label, color=colors(c, True))

                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / "crops" / names[c] / f"{p.stem}.jpg", BGR=True)

                text = f'person: {person_num_now}'
                text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                cv2.putText(im0, text, (640 - text_size[0] - 10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 0, 255), 2)
                print(person_num_now)

                # cv2.imshow('1', im0)
                if person_num_now > 0 and names[c] == 'person':
                    cam = cv2.VideoCapture(source)
                    global thread_save
                    if not thread_save:
                        thread_save = True
                        threading.Thread(target=save_video_thread, args=(cam, names[c])).start()
                    if number == 0:
                        number = number + 1
                        sendMessage.send_message()  # 发现有人摔倒报警

            # Stream results
            im0 = annotator.result()
            if view_img:
                retval, buffer = cv2.imencode('.jpg', im0)

                if not retval:
                    raise RuntimeError('Could not encode image')

                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def handle_points(request):
    global generate_frames_active

    try:
        data_list = json.loads(request.body)

        points = []
        for data in data_list:
            points.append(data['x'])
            points.append(data['y'])

        print(points)

        opt = parse_opt(points)

        current_video_stream = cv2.VideoCapture("rtmp://116.62.245.164:1935/live/2")
        responce = StreamingHttpResponse(generate_frames(**vars(opt)),  content_type='multipart/x-mixed-replace; boundary=frame')
        generate_frames_active = False

        return responce
        #return StreamingHttpResponse(generate_frames1(current_video_stream),
        #                         content_type='multipart/x-mixed-replace; boundary=frame')
    except json.JSONDecodeError as e:

        current_video_stream = cv2.VideoCapture("rtmp://116.62.245.164:1935/live/2")

        return StreamingHttpResponse(generate_frames1(current_video_stream),
                                             content_type='multipart/x-mixed-replace; boundary=frame')
