import json
from pathlib import Path
import threading
import os
import sys
import pandas as pd
import pyodbc

import cv2
import torch
import torch.backends.cudnn as cudnn

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadWebcam
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors
from utils.torch_utils import select_device, time_sync

import CONFIG as cfg

CONNECTION = pyodbc.connect(
    f'Driver={{SQL Server}};\
    Server={cfg.SERVER_IP};\
    Database={cfg.DATABASE};\
    UID={cfg.USER};\
    PWD={cfg.PASSWORD};'
)
CURSER = CONNECTION.cursor()


def update_db(query, records):
    records = list(zip(*records.values()))
    CURSER.executemany(query, records)
    CONNECTION.commit()
    LOGGER.info(f'DB updated successfully')


def measure(bbox, w, h):
    return (
        (cfg.FOA_HORIZONTAL * abs(float(bbox[2]) - float(bbox[0]))) / w,
        (cfg.FOA_VERTICAL * abs(float(bbox[3]) - float(bbox[1]))) / h
    )


def initialize_dict():
    annotations_dict = {
        "IMAGE NAME": [],
        "IMAGE WIDTH": [],
        "IMAGE HEIGHT": [],
        "DEFECT TYPE": [],
        "DEFECT WIDTH (cm)": [],
        "DEFECT HEIGHT (cm)": [],
        "CONFIDENCE SCORE": [],
        "NORMALIZED BOX CENTER X": [],
        "NORMALIZED BOX CENTER Y": [],
        "NORMALIZED BOX WIDTH": [],
        "NORMALIZED BOX HEIGHT": []
    }
    return annotations_dict


@torch.no_grad()
def run():
    source = str(cfg.SOURCE)
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # make necessary Directories
    main_dir = increment_path(cfg.RESULTS_PATH, exist_ok=False)  # increment run
    visualizations_dir = os.path.join(main_dir, "visualiztions")
    images_dir = os.path.join(main_dir, "images")
    labels_dir = os.path.join(main_dir, "labels")
    os.makedirs(visualizations_dir, exist_ok=True)  # make dir
    os.makedirs(labels_dir, exist_ok=True)  # make dir
    os.makedirs(images_dir, exist_ok=True)  # make dir

    # Load YOLO model
    device = select_device()
    model = DetectMultiBackend(cfg.PRETRAINED_WEIGHT_PATH, device=device, dnn=False, data=None)
    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
    imgsz = check_img_size(cfg.IMAGE_SIZE, s=stride)  # check image size
    crop_params = [cfg.LEFT_CROP_PERCENTAGE, cfg.TOP_CROP_PERCENTAGE, cfg.RIGHT_CROP_PERCENTAGE, cfg.BOTTOM_CROP_PERCENTAGE]

    # Dataloader
    if webcam:
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadWebcam(source, img_size=cfg.IMAGE_SIZE, stride=stride, crop_params=crop_params)
    else:
        dataset = LoadImages(source, img_size=cfg.IMAGE_SIZE, stride=stride, auto=pt, crop_params=None)

    vis_delay = 1 if dataset.mode in ["webcam", "video"] else 0     # in ms
    bs, vid_writer = 1, None

    # create a classes mapping json file
    classes_json = os.path.join(main_dir, "classes.json")
    with open(classes_json, "w") as f:
        classes_dict = {index: name for index, name in enumerate(names)}
        json.dump(classes_dict, f)

    # generate CSV for resutls
    results_csv = os.path.join(main_dir, "annotations.csv")

    # initialize dataframe to store records
    annotations_dict = initialize_dict()
    df_main = pd.DataFrame(annotations_dict)

    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *cfg.IMAGE_SIZE), half=False)  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0
    detections_counter = 1
    for path, im, im0s, vid_cap, s in dataset:
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0

        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        pred = model(im, augment=False, visualize=False)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, cfg.CONFIDENCE_THRESHOLD, cfg.IOU_THRESHOLD, classes=None, agnostic=False, max_det=1000)
        det = pred[0]       # take first image
        dt[2] += time_sync() - t3

        # Process predictions
        seen += 1
        p, im0 = path, im0s.copy()
        frame = dataset.count if webcam else getattr(dataset, 'frame', 0)

        p = Path(p)  # to Path
        ann_path = os.path.join(visualizations_dir, p.name)  # im.jpg
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        img_height, img_width, _ = im0.shape

        annotator = Annotator(im0, line_width=cfg.BBOX_LINE_THICKNESS, example=str(names))
        file_name = p.stem + ('' if dataset.mode == 'image' else f'_{frame}')
        img_path = os.path.join(images_dir, file_name + '.jpg')  # im.txt

        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
            # Write results
            for *xyxy, conf, cls in reversed(det):
                if cfg.SAVE_ANNOTATIONS:  # Write to file
                    # save label
                    txt_path = os.path.join(labels_dir, file_name + '.txt')  # im.txt
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    line = (cls, *xywh)  # label format
                    with open(txt_path, 'a') as f:
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    # measure the actual height and width on the detected defect in centimeters
                    width_actual, height_actual = measure(list(xyxy), img_width, img_height)

                    # add data record to annotations dictionary
                    annotations_dict["IMAGE NAME"].append(file_name + ".jpg")
                    annotations_dict['IMAGE WIDTH'].append(img_width)
                    annotations_dict['IMAGE HEIGHT'].append(img_height)
                    annotations_dict['DEFECT TYPE'].append(names[int(cls)])
                    annotations_dict['DEFECT WIDTH (cm)'].append(width_actual)
                    annotations_dict['DEFECT HEIGHT (cm)'].append(height_actual)
                    annotations_dict['CONFIDENCE SCORE'].append(float(conf))
                    annotations_dict['NORMALIZED BOX CENTER X'].append(float(xywh[0]))
                    annotations_dict['NORMALIZED BOX CENTER Y'].append(float(xywh[1]))
                    annotations_dict['NORMALIZED BOX WIDTH'].append(float(xywh[2]))
                    annotations_dict['NORMALIZED BOX HEIGHT'].append(float(xywh[3]))

                # draw bouding boxes
                if cfg.SAVE_VISUALIZATIONS:
                    c = int(cls)  # integer class
                    label = f'{names[c]} {conf:.2f}'
                    annotator.box_label(xyxy, label, color=colors(c, True))

            # save image
            if cfg.SAVE_ANNOTATIONS:
                cv2.imwrite(img_path, im0s)

            # increase the images counter only if detections are made
            detections_counter += 1

        # Stream results
        im0 = annotator.result()

        # Save results (image with detections)
        if cfg.SAVE_VISUALIZATIONS:
            if dataset.mode == 'image':
                cv2.imwrite(ann_path, im0)
            else:
                if vid_writer is None:
                    if vid_cap:
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:
                        fps, w, h = 20, im0.shape[1], im0.shape[0]
                    ann_path = str(Path(ann_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                    vid_writer = cv2.VideoWriter(ann_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer.write(im0)

                # also save images
                if len(det):
                    os.makedirs(os.path.join(visualizations_dir, "images"), exist_ok=True)
                    cv2.imwrite(os.path.join(visualizations_dir, "images", file_name + '.jpg'), im0)

        if cfg.VISUALIZE:
            if im0.shape[1] > 1440:
                im0 = cv2.resize(im0, (0, 0), fx=0.75, fy=0.75)

            cv2.imshow('detection', im0)
            if cv2.waitKey(vis_delay) == ord('q'):
                break

        # update the CSV and DB after 100 images
        if detections_counter % 100 == 0:
            LOGGER.info(f'DB update in process')

            # save the strored records to Hard Drive in the form of a CSV
            df_ = pd.DataFrame(annotations_dict)
            df_main = df_main.append(df_, ignore_index=True)
            df_main.to_csv(results_csv, index=False)

            # update Database in separate thread
            t1 = threading.Thread(target=update_db, args=(cfg.INSERT_QUERY, annotations_dict))
            t1.start()

            # reset the annotations dictionary
            annotations_dict = initialize_dict()

    LOGGER.info(f'DB update in process')
    # save the strored records to Hard Drive in the form of a CSV
    df_ = pd.DataFrame(annotations_dict)
    df_main = df_main.append(df_, ignore_index=True)
    df_main.to_csv(results_csv, index=False)
    # update Database in separate thread
    t1 = threading.Thread(target=update_db, args=(cfg.INSERT_QUERY, annotations_dict))
    t1.start()
    
    # release vid_writer
    if vid_writer is not None:
        vid_writer.release()
    cv2.destroyAllWindows()

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if cfg.SAVE_ANNOTATIONS or cfg.SAVE_VISUALIZATIONS:
        LOGGER.info(f"Results saved to {colorstr('bold', main_dir)}")


if __name__ == "__main__":
    run()
