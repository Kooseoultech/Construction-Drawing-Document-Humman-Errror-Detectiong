# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 10:15:43 2025

@author: S3PARC
"""

##For Streamlit

import streamlit as st
import tempfile
import os
import pandas as pd

# 모듈 import (GitHub에 업로드한 모듈들이 동일 폴더 혹은 패키지로 설치되어 있어야 함)
import SCD_complete      # SCD 관련 코드
import SD_complete       # SD 관련 코드
import Validation_Code   # Validation 관련 코드

st.set_page_config(page_title="문서 처리 및 검증 웹앱", layout="wide")
st.title("문서 처리 및 검증 웹앱")

# 사이드바에서 기능 선택 (탭 방식)
tab = st.sidebar.radio("기능 선택", ("SCD 처리", "SD 처리", "Validation 검증"))

if tab == "SCD 처리":
    st.header("SCD 처리")
    uploaded_file = st.file_uploader("SCD 관련 파일(PDF, 이미지, JSON 등)을 업로드하세요.", type=["pdf", "png", "jpg", "jpeg", "json"])
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        st.info("SCD 처리를 시작합니다. 잠시 기다려 주세요...")
        try:
            # SCD_complete 모듈의 처리 함수 호출 (반환값: 결과 CSV 파일 경로 또는 DataFrame)
            scd_result = SCD_complete.process_scd(tmp_path)
            st.success("SCD 처리 완료!")
            # 결과가 CSV 파일 경로라면 해당 파일을 읽어서 다운로드 버튼 제공
            if isinstance(scd_result, str) and os.path.exists(scd_result):
                with open(scd_result, "rb") as f:
                    st.download_button("SCD 결과 CSV 다운로드", f, file_name=os.path.basename(scd_result))
                st.write("SCD 결과:", pd.read_csv(scd_result).head())
            # 혹은 DataFrame이라면 바로 보여주기 및 다운로드
            elif isinstance(scd_result, pd.DataFrame):
                st.dataframe(scd_result.head())
                csv_bytes = scd_result.to_csv(index=False).encode("utf-8")
                st.download_button("SCD 결과 CSV 다운로드", csv_bytes, file_name="SCD_result.csv")
            else:
                st.warning("SCD 처리 결과를 확인할 수 없습니다.")
        except Exception as e:
            st.error(f"SCD 처리 중 오류 발생: {e}")

elif tab == "SD 처리":
    st.header("SD 처리")
    uploaded_file = st.file_uploader("SD 관련 파일(PDF, 이미지, JSON 등)을 업로드하세요.", type=["pdf", "png", "jpg", "jpeg", "json"])
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        st.info("SD 처리를 시작합니다. 잠시 기다려 주세요...")
        try:
            # SD_complete 모듈의 처리 함수 호출 (반환값: 결과 CSV 파일 경로 또는 DataFrame)
            sd_result = SD_complete.process_sd(tmp_path)
            st.success("SD 처리 완료!")
            if isinstance(sd_result, str) and os.path.exists(sd_result):
                with open(sd_result, "rb") as f:
                    st.download_button("SD 결과 CSV 다운로드", f, file_name=os.path.basename(sd_result))
                st.write("SD 결과:", pd.read_csv(sd_result).head())
            elif isinstance(sd_result, pd.DataFrame):
                st.dataframe(sd_result.head())
                csv_bytes = sd_result.to_csv(index=False).encode("utf-8")
                st.download_button("SD 결과 CSV 다운로드", csv_bytes, file_name="SD_result.csv")
            else:
                st.warning("SD 처리 결과를 확인할 수 없습니다.")
        except Exception as e:
            st.error(f"SD 처리 중 오류 발생: {e}")

elif tab == "Validation 검증":
    st.header("Validation 검증")
    st.write("두 CSV 파일을 업로드하여 검증을 진행합니다.")
    uploaded_csv1 = st.file_uploader("CSV 파일 1 업로드", type=["csv"], key="csv1")
    uploaded_csv2 = st.file_uploader("CSV 파일 2 업로드", type=["csv"], key="csv2")
    
    if uploaded_csv1 is not None and uploaded_csv2 is not None:
        try:
            df1 = pd.read_csv(uploaded_csv1)
            df2 = pd.read_csv(uploaded_csv2)
            st.dataframe(df1.head())
            st.dataframe(df2.head())
            
            st.info("Validation 검증을 시작합니다...")
            # Validation_Code 모듈의 함수 호출 (예: validate_csv_files(df1, df2) → Excel 파일 바이트 스트림 반환)
            validation_result = Validation_Code.validate_csv_files(df1, df2)
            if validation_result:
                st.success("Validation 검증 완료!")
                st.download_button("검증 결과 Excel 다운로드", validation_result, file_name="Validation_Result.xlsx")
            else:
                st.warning("검증 결과를 확인할 수 없습니다.")
        except Exception as e:
            st.error(f"Validation 검증 중 오류 발생: {e}")
