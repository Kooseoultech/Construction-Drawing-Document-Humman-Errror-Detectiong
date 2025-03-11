import streamlit as st
import os
import tempfile
import subprocess
from pdf2image import convert_from_path
import torch
from huggingface_hub import hf_hub_download
from doclayout_yolo import YOLOv10
import cv2
import pandas as pd

def convert_pdf_to_images(uploaded_file):
    with tempfile.TemporaryDirectory() as temp_dir:
        pdf_path = os.path.join(temp_dir, uploaded_file.name)
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        images = convert_from_path(pdf_path, dpi=500)
        image_paths = []
        for i, image in enumerate(images):
            image_path = os.path.join(temp_dir, f'page_{i + 1}.jpg')
            image.save(image_path, 'JPEG')
            image_paths.append(image_path)
        return image_paths

def apply_yolo_on_images(image_paths):
    repo_id = "juliozhao/DocLayout-YOLO-DocStructBench"
    filename = "doclayout_yolo_docstructbench_imgsz1024.pt"
    model_path = hf_hub_download(repo_id=repo_id, filename=filename)
    model = YOLOv10(model_path)
    results = {}
    
    for img_path in image_paths:
        det_res = model.predict(img_path, imgsz=1024, conf=0.2, iou=0.1, device="cpu")
        boxes = det_res[0].boxes
        names = det_res[0].names
        
        results[img_path] = []
        for box in boxes:
            cls_id = int(box.cls.item())
            conf = box.conf.item()
            cls_name = names[cls_id]
            if cls_name in ["figure", "plain text"] and conf >= 0.2:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                results[img_path].append((cls_name, conf, (x1, y1, x2, y2)))
    
    return results

def process_uploaded_pdf():
    uploaded_file = st.file_uploader("Upload PDF Document", type=["pdf"])
    if uploaded_file:
        st.write("Processing PDF to Images...")
        image_paths = convert_pdf_to_images(uploaded_file)
        st.success(f"Extracted {len(image_paths)} images from PDF.")
        
        st.write("Applying YOLO Model...")
        results = apply_yolo_on_images(image_paths)
        st.success("YOLO Processing Completed.")
        
        for img_path, detections in results.items():
            st.image(img_path, caption="Processed Image")
            st.write("Detections:")
            for det in detections:
                st.write(f"{det[0]} - Confidence: {det[1]:.2f} - Box: {det[2]}")

st.title("Structural Document Processing App")
st.sidebar.header("Navigation")
menu = st.sidebar.radio("Choose a Process", ["Upload & Process PDF"])

if menu == "Upload & Process PDF":
    process_uploaded_pdf()
