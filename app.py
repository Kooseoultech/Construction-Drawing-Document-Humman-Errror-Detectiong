# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 17:05:26 2025

@author: S3PARC
"""
import sys
sys.path.append(r"C:\Users\S3PARC\AppData\Local\Programs\Python\Python310\lib\site-packages")

import streamlit as st
from PIL import Image
import numpy as np
from surya_ocr import run_ocr
from surya_ocr.models import load_detection_model, load_recognition_model

@st.cache_resource
def load_models():
    det_model = load_detection_model()
    rec_model = load_recognition_model()
    return det_model, rec_model

st.title("📑 Surya OCR Streamlit 앱")

# 모델 로드
det_model, rec_model = load_models()

# 이미지 업로드
uploaded_image = st.file_uploader("OCR할 이미지를 업로드하세요", type=["png", "jpg", "jpeg"])

if uploaded_image:
    image = Image.open(uploaded_image).convert("RGB")
    image_np = np.array(image)

    # OCR 수행
    result = run_ocr(image_np, det_model, rec_model)

    # OCR 결과 출력
    st.subheader("📝 OCR 결과 텍스트")
    st.write(result)

    st.subheader("🖼️ OCR 결과 이미지")
    st.image(image, use_column_width=True)
