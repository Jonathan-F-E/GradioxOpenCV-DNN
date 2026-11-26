import cv2
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

def load_yolo():
    net = cv2.dnn.readNetFromONNX(str(BASE_DIR / "yolov5s.onnx"))
    classes = []
    with open(str(BASE_DIR / "coco.names"), "r") as f:
        classes = [line.strip() for line in f.readlines()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    return net, classes, colors

def resize_max_height(img, max_height=500):
    h, w = img.shape[:2]
    if h <= max_height:
        return img
    scale = max_height / h
    new_w = int(w * scale)
    return cv2.resize(img, (new_w, max_height), interpolation=cv2.INTER_AREA)

def load_image(img_path, max_height=500):
    img = cv2.imread(img_path)
    img = resize_max_height(img, max_height=max_height)
    height, width, channels = img.shape
    return img, height, width, channels

def detect_objects_yolo(img, net):
    blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(640, 640), mean=(0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward()
    return blob, outputs

def get_box_dimensions_yolo(outputs, height, width):
    boxes = []
    confs = []
    class_ids = []
    scale_x = width / 640
    scale_y = height / 640
    for row in outputs[0]:
        obj_conf = float(row[4])
        if obj_conf <= 0:
            continue
        scores = row[5:]
        class_id = int(np.argmax(scores))
        conf = obj_conf * float(scores[class_id])
        if conf > 0.3:
            center_x = row[0]
            center_y = row[1]
            w = row[2]
            h = row[3]
            x = int((center_x - w / 2) * scale_x)
            y = int((center_y - h / 2) * scale_y)
            boxes.append([x, y, int(w * scale_x), int(h * scale_y)])
            confs.append(conf)
            class_ids.append(class_id)
    return boxes, confs, class_ids

def draw_labels_yolo(boxes, confs, colors, class_ids, classes, img):
    indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.2, 0.5)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in np.array(indexes).flatten() if len(indexes) else []:
        x, y, w, h = boxes[i]
        label = f"{classes[class_ids[i]]}: {confs[i]:.2f}"
        color = colors[class_ids[i]]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label, (x, y - 5), font, .7, color, 2)
    return img