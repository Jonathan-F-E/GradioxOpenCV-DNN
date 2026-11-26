import cv2
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

def load_model():
    model_path = str(BASE_DIR / "frozen_inference_graph.pb")
    config_path = str(BASE_DIR / "graph.pbtxt")
    if hasattr(cv2.dnn, "readNetFromTensorflow"):
        model = cv2.dnn.readNetFromTensorflow(model_path, config_path)
    else:
        model = cv2.dnn.readNet(model_path, config_path)
    with open(str(BASE_DIR / "object_detection_classes_coco.txt"), "r") as f:
        class_names = f.read().split("\n")
    COLORS = np.random.uniform(2, 255, size=(len(class_names), 3))
    return model, class_names, COLORS

def resize_max_height(img, max_height=500):
    h, w = img.shape[:2]
    if h <= max_height:
        return img
    scale = max_height / h
    new_w = int(w * scale)
    return cv2.resize(img, (new_w, max_height), interpolation=cv2.INTER_AREA)

def load_img(img_path, max_height=500):
    img = cv2.imread(img_path)
    img = resize_max_height(img, max_height=max_height)
    height, width, channels = img.shape
    return img, height, width, channels

def detect_objects(img, net):
    blob = cv2.dnn.blobFromImage(img, size=(300, 300), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward()
    return blob, outputs

def get_box_dimensions(outputs, height, width):
    boxes = []
    class_ids = []
    confs = []
    for detection in outputs[0, 0, :, :]:
        score = float(detection[2])
        class_id = int(detection[1])
        if score > 0.5:
            w = int(detection[5] * width)
            h = int(detection[6] * height)
            x = int(detection[3] * width)
            y = int(detection[4] * height)
            boxes.append([x, y, w, h])
            class_ids.append(class_id)
            confs.append(score)
    return boxes, confs, class_ids

def draw_labels(boxes, confs, colors, class_ids, classes, img):
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        x, y, w, h = boxes[i]
        label = f"{classes[class_ids[i]]}: {confs[i]:.2f}"
        color = colors[class_ids[i]]
        cv2.rectangle(img, (x, y), (w, h), color, 2)
        cv2.putText(img, label, (x, y - 5), font, .9, color, 1.3)
    return img
