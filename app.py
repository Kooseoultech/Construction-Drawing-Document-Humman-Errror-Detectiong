import streamlit as st
import pandas as pd
import os
import re
import numpy as np

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

def remove_duplicates(df):
    df_unique = df.drop_duplicates(subset=["Member code", "Direction"], keep='first')
    return df_unique

def compare_rows(sc_row, scd_row, direction_exists):
    errors = []
    for col in ["Top_rebar", "Bot_rebar", "Stirrups", "Width", "Height"]:
        if sc_row.get(col) != scd_row.get(col):
            errors.append(f"{col} mismatch")
    return errors

def compare_csv_files(sc_csv, scd_csv):
    df_sc = pd.read_csv(sc_csv)
    df_scd = pd.read_csv(scd_csv)
    sc_dict = df_sc.set_index(['Member code', 'Direction']).to_dict(orient='index')
    scd_dict = df_scd.set_index(['Member code', 'Direction']).to_dict(orient='index')
    
    all_keys = set(sc_dict.keys()).union(set(scd_dict.keys()))
    error_records = []
    
    for key in all_keys:
        sc_data = sc_dict.get(key)
        scd_data = scd_dict.get(key)
        
        if sc_data and scd_data:
            errors = compare_rows(sc_data, scd_data, direction_exists=True)
            if errors:
                error_records.append({"Member code": key[0], "Direction": key[1], "Error Detail": ', '.join(errors)})
        else:
            error_records.append({"Member code": key[0], "Direction": key[1], "Error Detail": "Missing in one file"})
    
    return pd.DataFrame(error_records)

def main():
    st.title("CSV Data Processing and Error Detection")
    
    tab1, tab2, tab3 = st.tabs(["Code 1", "Code 2", "Code 3"])
    
    with tab1:
        st.header("Duplicate Member Code and Direction Finder")
        uploaded_file1 = st.file_uploader("Upload CSV file", type=["csv"], key="file1")
        if uploaded_file1 is not None:
            duplicates = find_duplicate_member_direction(uploaded_file1)
            st.write("### Duplicate Member Codes and Directions")
            st.write(duplicates)
    
    with tab2:
        st.header("Find Fully Duplicate Rows")
        uploaded_file2 = st.file_uploader("Upload CSV file", type=["csv"], key="file2")
        if uploaded_file2 is not None:
            full_duplicates = find_duplicates_in_csv(uploaded_file2)
            st.write("### Fully Duplicate Rows")
            st.write(full_duplicates)
    
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
