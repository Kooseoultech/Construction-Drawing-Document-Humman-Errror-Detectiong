import streamlit as st
import pandas as pd
import os
import re
import numpy as np
import tempfile
from pdf2image import convert_from_path

def find_duplicate_member_direction(input_csv):
    df = pd.read_csv(input_csv)
    dup_group = df.groupby(["Member code", "Direction"]).size().reset_index(name="Count")
    duplicates = dup_group[dup_group["Count"] > 1]
    return duplicates

def find_duplicates_in_csv(input_csv):
    df = pd.read_csv(input_csv)
    subset_cols = ["Page", "Member code", "Direction", "Top_rebar", "Bot_rebar", "Stirrups", "Width", "Height", "Top_Rebar_Img", "Bot_Rebar_Img"]
    existing_cols = [c for c in subset_cols if c in df.columns]
    duplicates_mask = df.duplicated(subset=existing_cols, keep=False)
    return df[duplicates_mask]

def compare_rows(sc_row, scd_row):
    errors = []
    for col in ["Top_rebar", "Bot_rebar", "Stirrups", "Width", "Height"]:
        if sc_row.get(col) != scd_row.get(col):
            errors.append(f"{col} mismatch")
    return errors

def compare_csv_files(sc_csv, scd_csv):
    df_sc = pd.read_csv(sc_csv)
    df_scd = pd.read_csv(scd_csv)
    sc_dict = df_sc.set_index(["Member code", "Direction"]).to_dict(orient="index")
    scd_dict = df_scd.set_index(["Member code", "Direction"]).to_dict(orient="index")
    
    all_keys = set(sc_dict.keys()).union(set(scd_dict.keys()))
    error_records = []
    
    for key in all_keys:
        sc_data = sc_dict.get(key)
        scd_data = scd_dict.get(key)
        
        if sc_data and scd_data:
            errors = compare_rows(sc_data, scd_data)
            if errors:
                error_records.append({"Member code": key[0], "Direction": key[1], "Error Detail": ', '.join(errors)})
        else:
            error_records.append({"Member code": key[0], "Direction": key[1], "Error Detail": "Missing in one file"})
    
    return pd.DataFrame(error_records)

def process_pdf(uploaded_pdf):
    with tempfile.TemporaryDirectory() as temp_dir:
        pdf_path = os.path.join(temp_dir, uploaded_pdf.name)
        with open(pdf_path, "wb") as f:
            f.write(uploaded_pdf.read())
        
        images = convert_from_path(pdf_path, dpi=300)
        image_paths = []
        for i, image in enumerate(images):
            img_path = os.path.join(temp_dir, f"page_{i+1}.jpg")
            image.save(img_path, "JPEG")
            image_paths.append(img_path)
        return image_paths

def main():
    st.title("Multi-Function Data Processing App")
    
    tab1, tab2, tab3 = st.tabs(["PDF Processing", "Duplicate Detection", "CSV Comparison"])
    
    with tab1:
        st.header("PDF to Image Processing")
        uploaded_pdf = st.file_uploader("Upload a PDF file", type=["pdf"], key="pdf")
        if uploaded_pdf is not None:
            images = process_pdf(uploaded_pdf)
            st.write(f"Converted {len(images)} pages to images")
            for img_path in images:
                st.image(img_path, caption=os.path.basename(img_path), use_column_width=True)
    
    with tab2:
        st.header("Find Duplicate Member Codes and Directions")
        uploaded_file1 = st.file_uploader("Upload CSV file", type=["csv"], key="file1")
        if uploaded_file1 is not None:
            duplicates = find_duplicate_member_direction(uploaded_file1)
            st.write("### Duplicate Member Codes and Directions")
            st.write(duplicates)
    
    with tab3:
        st.header("CSV File Comparison")
        uploaded_file3 = st.file_uploader("Upload First CSV file", type=["csv"], key="file3")
        uploaded_file4 = st.file_uploader("Upload Second CSV file", type=["csv"], key="file4")
        
        if uploaded_file3 is not None and uploaded_file4 is not None:
            error_df = compare_csv_files(uploaded_file3, uploaded_file4)
            st.write("### Error Report")
            st.write(error_df)
            
            error_file_path = "error_report.csv"
            error_df.to_csv(error_file_path, index=False)
            st.download_button(label="Download Error Report", data=open(error_file_path, "rb").read(), file_name="error_report.csv", mime="text/csv")

if __name__ == "__main__":
    main()
