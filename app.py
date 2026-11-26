import cv2
import numpy as np
import gradio as gr
from fasterrcnnnet import load_model, load_img, detect_objects, get_box_dimensions, draw_labels
from yolov5 import load_yolo, load_image, detect_objects_yolo, get_box_dimensions_yolo, draw_labels_yolo
from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent
SAMPLE_IMG = str(BASE_DIR / "sample/dog.jpg")

def image_detect(img, model, net, classes, colors):
    height, width = img.shape[:2]
    if model == "Faster R-CNN":
        _, outputs = detect_objects(img, net)
        boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
        return draw_labels(boxes, confs, colors, class_ids, classes, img)
    elif model == "YOLOv5":
        _, outputs = detect_objects_yolo(img, net)
        boxes, confs, class_ids = get_box_dimensions_yolo(outputs, height, width)
        return draw_labels_yolo(boxes, confs, colors, class_ids, classes, img)

def load_selected(model):
    return load_model() if model == "Faster R-CNN" else load_yolo()

def img_inf(img, model):
    if img is None:
        return None
    net, classes, colors = load_selected(model)
    if model == "Faster R-CNN":
        image, _, _, _ = load_img(img)
    else:
        image, _, _, _ = load_image(img)
    image = image_detect(image, model, net, classes, colors)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

model_name_img = gr.Radio(["Faster R-CNN", "YOLOv5"], value="YOLOv5", label="Model", info="choose your model")
inputs_image = gr.Image(type="filepath", label="Input Image")
outputs_image = gr.Image(type="numpy", label="Output Image")
interface_image = gr.Interface(
    fn=img_inf,
    inputs=[inputs_image, model_name_img],
    outputs=outputs_image,
    title="Image Inference",
    description="Upload your photo and select one model and see the results!",
    examples=[[SAMPLE_IMG]],
    cache_examples=False,
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    gr.TabbedInterface(
        [interface_image], tab_names=["Image"], title="GradioxOpenCV-DNN - JFE"
    ).queue().launch(server_port=port, server_name="0.0.0.0")