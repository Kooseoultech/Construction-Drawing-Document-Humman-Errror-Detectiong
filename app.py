import streamlit as st
import os
import subprocess
import pandas as pd
from pdf2image import convert_from_path

def convert_pdf_to_images(pdf_path, output_folder):
    """Convert PDF to images and store in the output folder"""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    images = convert_from_path(pdf_path, dpi=500)
    image_paths = []
    for i, image in enumerate(images):
        image_path = os.path.join(output_folder, f'page_{i + 1}.jpg')
        image.save(image_path, 'JPEG')
        image_paths.append(image_path)
    return image_paths

def run_surya_ocr(input_folder):
    """Run Surya OCR on extracted images"""
    command = ["surya_ocr", input_folder]
    subprocess.run(command, capture_output=True, text=True)

def validate_results(input_csv, output_csv):
    """Perform validation of extracted results"""
    df = pd.read_csv(input_csv)
    duplicates = df[df.duplicated(subset=['Member code', 'Direction'], keep=False)]
    duplicates.to_csv(output_csv, index=False, encoding='utf-8-sig')
    return duplicates

def compare_csv_files(sc_csv, scd_csv):
    """Compare SCD and SD extracted data"""
    df_sc = pd.read_csv(sc_csv)
    df_scd = pd.read_csv(scd_csv)
    merged = df_sc.merge(df_scd, on=['Member code', 'Direction'], suffixes=('_SC', '_SCD'))
    discrepancies = merged[df_sc.columns[df_sc.ne(df_scd)].any(1)]
    return discrepancies

st.title("Structural Document Processing and Validation")

st.sidebar.header("Upload Files")
scd_pdf = st.sidebar.file_uploader("Upload SCD PDF", type=['pdf'])
sd_pdf = st.sidebar.file_uploader("Upload SD PDF", type=['pdf'])

if st.sidebar.button("Process Files"):
    if scd_pdf:
        with open("scd_temp.pdf", "wb") as f:
            f.write(scd_pdf.getbuffer())
        st.write("Processing SCD PDF...")
        scd_images = convert_pdf_to_images("scd_temp.pdf", "scd_images")
        st.write(f"Extracted {len(scd_images)} images from SCD PDF")
        run_surya_ocr("scd_images")
        st.success("SCD OCR Completed")
    
    if sd_pdf:
        with open("sd_temp.pdf", "wb") as f:
            f.write(sd_pdf.getbuffer())
        st.write("Processing SD PDF...")
        sd_images = convert_pdf_to_images("sd_temp.pdf", "sd_images")
        st.write(f"Extracted {len(sd_images)} images from SD PDF")
        run_surya_ocr("sd_images")
        st.success("SD OCR Completed")

st.sidebar.header("Validation")
validate_button = st.sidebar.button("Validate Extracted Data")
if validate_button:
    scd_csv = "scd_results.csv"
    sd_csv = "sd_results.csv"
    validation_csv = "validation_results.csv"
    validate_results(scd_csv, validation_csv)
    st.write("Validation Completed. Download results below:")
    st.download_button("Download Validation Results", validation_csv)

st.sidebar.header("Comparison")
compare_button = st.sidebar.button("Compare SCD and SD Data")
if compare_button:
    sc_csv = "sc_results.csv"
    scd_csv = "scd_results.csv"
    discrepancies = compare_csv_files(sc_csv, scd_csv)
    st.write("Discrepancies Found:")
    st.dataframe(discrepancies)
