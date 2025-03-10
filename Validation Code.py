# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 16:51:24 2025

@author: S3PARC
"""

###############################################################################
#추출 정확도 확인






################################################
#중복확인

import pandas as pd

def find_duplicate_member_direction(input_csv, output_csv):
    # CSV 파일 읽기
    df = pd.read_csv(input_csv)
    
    # "Member code"와 "Direction" 열이 동일한 행들의 개수를 그룹화하여 계산
    dup_group = df.groupby(["Member code", "Direction"]).size().reset_index(name="Count")
    
    # Count가 2 이상인 그룹만 추출 (즉, 중복이 있는 경우)
    duplicates = dup_group[dup_group["Count"] > 1]
    
    # 결과를 CSV 파일로 저장 (UTF-8 BOM 포함)
    duplicates.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print("중복된 Member code와 Direction이 저장된 CSV 파일:", output_csv)
    return duplicates

# 사용 예시:
input_csv = r"D:\LLLast\SD\merged_results_CONVERT2.csv"  # 분석할 CSV 파일 경로 (Member code와 Direction 열이 있어야 함)
output_csv = r"D:\LLLast\SD\duplicate_sd_direction2.csv"

find_duplicate_member_direction(input_csv, output_csv)

########################################


import pandas as pd

def find_duplicates_in_csv(input_csv, output_csv_duplicates):
    """
    주어진 CSV에서, Member code와 Direction을 포함한 나머지 열이 모두 동일한 경우를 중복으로 간주하여
    그 중복 행들을 별도의 CSV로 저장하는 함수.
    
    input_csv: 중복 검사할 원본 CSV 파일 경로
    output_csv_duplicates: 중복 행을 저장할 CSV 파일 경로
    """
    # CSV 읽기
    df = pd.read_csv(input_csv)
    
    # 중복 판단에 사용할 열 리스트
    # 필요에 따라 "Page"를 제외할 수도 있음
    subset_cols = [
        "Page", "Member code", "Direction", 
        "Top_rebar", "Bot_rebar", "Stirrups", 
        "Width", "Height", "Top_Rebar_Img", "Bot_Rebar_Img"
    ]
    
    # 실제 CSV에 이 열들이 모두 존재하는지 확인 (없으면 제거)
    existing_cols = [c for c in subset_cols if c in df.columns]
    
    # duplicated(keep=False)를 사용하면, 중복 그룹 내 모든 행이 True가 됨
    duplicates_mask = df.duplicated(subset=existing_cols, keep=False)
    
    # 중복 행만 추출
    df_duplicates = df[duplicates_mask].copy()
    
    if df_duplicates.empty:
        print(f"[INFO] {input_csv}에는 중복된 행이 없습니다.")
    else:
        df_duplicates.to_csv(output_csv_duplicates, index=False, encoding="utf-8-sig")
        print(f"[INFO] {input_csv}의 중복 행이 {output_csv_duplicates}에 저장되었습니다. "
              f"(총 {len(df_duplicates)}행)")
    
    return df_duplicates

# 사용 예시
csv_file_1 = r"D:\LLLast\SCD\transformed_results_last.csv"
csv_file_2 = r"D:\LLLast\SD\merged_results_CONVERT2.csv"

duplicates_csv_1 = r"D:\LLLast\check\SCD_duplicates.csv"
duplicates_csv_2 = r"D:\LLLast\check\SC_duplicates.csv"

# 첫 번째 CSV에서 중복 확인
find_duplicates_in_csv(csv_file_1, duplicates_csv_1)
# 두 번째 CSV에서 중복 확인
find_duplicates_in_csv(csv_file_2, duplicates_csv_2)

###########################################################################



import pandas as pd
import re
import numpy as np

##############################
# 키 생성 및 파싱 함수
##############################
def create_key(row):
    """
    각 행에서 "Member code"와 "Direction"을 추출하여 key를 생성합니다.
    Direction이 없으면 빈 문자열로 처리합니다.
    """
    member = str(row["Member code"]).strip()
    direction = str(row["Direction"]).strip() if ('Direction' in row and pd.notnull(row["Direction"])) else ""
    return (member, direction)

##############################
# 치수 표준화 및 비교 함수 (Width, Height)
##############################
def standardize_dimension_pair(val1, val2):
    try:
        v1 = str(val1).strip().replace(",", "")
        v2 = str(val2).strip().replace(",", "")
        n1 = float(v1)
        n2 = float(v2)
    except:
        return None, None
    if n1 <= 10 or n2 <= 10:
        if n1 < n2:
            n1 = n1 * 1000
        elif n2 < n1:
            n2 = n2 * 1000
    return int(n1), int(n2)

##############################
# 철근 및 Stirrup 파싱 함수
##############################
def parse_rebar_value(value):
    cleaned = re.sub(r"\s+", "", value)
    pattern = re.compile(r'(\d+)-[A-Za-z]+(\d+)', re.IGNORECASE)
    match = pattern.search(cleaned)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None

def parse_rebar_count(value):
    """
    Direction이 없는 경우 Bot_rebar 비교를 위해,
    하이픈(-) 앞의 숫자만 추출합니다.
    예: "40-12-D25" → 40, "34-UHD25" → 34
    """
    try:
        cleaned = re.sub(r"\s+", "", value)
        count_str = cleaned.split("-")[0]
        return int(count_str)
    except:
        return None

def parse_stirrup(value):
    pattern = re.compile(r'@(\d+)', re.IGNORECASE)
    match = pattern.search(value)
    if match:
        return int(match.group(1))
    return None

##############################
# 행 비교 함수
##############################
def compare_rows(sc_row, scd_row, direction_exists):
    errors = []
    if direction_exists:
        for col in ["Top_rebar", "Bot_rebar"]:
            sc_val = str(sc_row.get(col, "")).strip()
            scd_val = str(scd_row.get(col, "")).strip()
            if sc_val == scd_val:
                continue
            sc_count, sc_dia = parse_rebar_value(sc_val)
            scd_count, scd_dia = parse_rebar_value(scd_val)
            sub_errs = []
            if sc_count is None or scd_count is None:
                sub_errs.append(f"{col}: '{sc_val}' vs '{scd_val}'")
            else:
                if sc_count != scd_count:
                    sub_errs.append(f"{col} count error")
                if sc_dia != scd_dia:
                    sub_errs.append(f"{col} diameter error")
            if sub_errs:
                errors.append("; ".join(sub_errs))
        
        col = "Stirrups"
        sc_val = str(sc_row.get(col, "")).strip()
        scd_val = str(scd_row.get(col, "")).strip()
        if sc_val != scd_val:
            sc_spacing = parse_stirrup(sc_val)
            scd_spacing = parse_stirrup(scd_val)
            if sc_spacing is None or scd_spacing is None:
                errors.append(f"{col}: '{sc_val}' vs '{scd_val}'")
            elif sc_spacing != scd_spacing:
                errors.append(f"{col} spacing error")
        
        for col in ["Top_Rebar_Img", "Bot_Rebar_Img"]:
            sc_val = str(sc_row.get(col, "")).strip()
            scd_val = str(scd_row.get(col, "")).strip()
            try:
                sc_int = int(float(sc_val))
                scd_int = int(float(scd_val))
                if sc_int != scd_int:
                    errors.append(f"{col} error")
            except:
                if sc_val != scd_val:
                    errors.append(f"{col} error")
    else:
        # Direction이 없는 경우: Bot_rebar 비교 -> 하이픈 앞 숫자만 비교
        col = "Bot_rebar"
        sc_val = str(sc_row.get(col, "")).strip()
        scd_val = str(scd_row.get(col, "")).strip()
        sc_count = parse_rebar_count(sc_val)
        scd_count = parse_rebar_count(scd_val)
        if sc_count is None or scd_count is None:
            errors.append(f"{col}: '{sc_val}' vs '{scd_val}'")
        elif sc_count != scd_count:
            errors.append(f"{col} count error")
        
        col = "Bot_Rebar_Img"
        sc_val = str(sc_row.get(col, "")).strip()
        scd_val = str(scd_row.get(col, "")).strip()
        try:
            sc_int = int(float(sc_val))
            scd_int = int(float(scd_val))
            if sc_int != scd_int:
                errors.append(f"{col} error")
        except:
            if sc_val != scd_val:
                errors.append(f"{col} error")
    
    for col in ["Width"]:
        sc_val = str(sc_row.get(col, "")).strip()
        scd_val = str(scd_row.get(col, "")).strip()
        try:
            sc_width, scd_width = standardize_dimension_pair(sc_val, scd_val)
            if sc_width is None or scd_width is None or sc_width != scd_width:
                errors.append(f"{col} error")
        except:
            errors.append(f"{col} error")
    
    for col in ["Height"]:
        sc_val = str(sc_row.get(col, "")).strip()
        scd_val = str(scd_row.get(col, "")).strip()
        try:
            sc_height, scd_height = standardize_dimension_pair(sc_val, scd_val)
            if sc_height is None or scd_height is None or sc_height != scd_height:
                errors.append(f"{col} error")
        except:
            errors.append(f"{col} error")
    
    return errors

##############################
# 중복 제거 함수 (Member code, Direction 기준)
##############################
def remove_duplicates(df, file_type_prefix):
    duplicate_errors = []
    dup_mask = df.duplicated(subset=["Member code", "Direction"], keep=False)
    dup_groups = df[dup_mask].groupby(["Member code", "Direction"])
    for (member, direction), group in dup_groups:
        if len(group) > 1:
            duplicate_errors.append({
                "Member code": member,
                "Direction": direction,
                "Error Detail": f"[{file_type_prefix} Duplicate]"
            })
    df_unique = df.drop_duplicates(subset=["Member code", "Direction"], keep='first')
    return df_unique, duplicate_errors

##############################
# 오류 메시지 포맷팅 함수
##############################
def format_error_message(err_str):
    parts = [p.strip() for p in err_str.split(";")]
    formatted = []
    for part in parts:
        part_clean = re.sub(r"\(.*?\)", "", part)
        formatted.append(f"[{part_clean.strip()}]")
    return " ".join(formatted)

##############################
# CSV 비교 및 Excel 저장 함수 (세 번째 시트: 오류별 집계)
##############################
def compare_csv_files_to_excel(sc_csv, scd_csv, output_excel, file_type_prefix_sc="SD", file_type_prefix_scd="SCD"):
    df_sc = pd.read_csv(sc_csv)
    df_scd = pd.read_csv(scd_csv)
    
    df_sc_unique, sc_dup_errors = remove_duplicates(df_sc, file_type_prefix_sc)
    df_scd_unique, scd_dup_errors = remove_duplicates(df_scd, file_type_prefix_scd)
    
    df_sc_unique.loc[:, 'key'] = df_sc_unique.apply(lambda row: create_key(row), axis=1)
    df_scd_unique.loc[:, 'key'] = df_scd_unique.apply(lambda row: create_key(row), axis=1)
    
    sc_dict = df_sc_unique.set_index('key').to_dict(orient='index')
    scd_dict = df_scd_unique.set_index('key').to_dict(orient='index')
    
    all_keys = set(sc_dict.keys()).union(set(scd_dict.keys()))
    
    error_records = []
    summary_records = []
    error_counts = {}
    
    # 중복 오류 추가 및 집계
    for err in sc_dup_errors + scd_dup_errors:
        error_records.append(err)
        err_detail = re.sub(r"\(.*?\)", "", err["Error Detail"]).strip()
        error_counts[err_detail] = error_counts.get(err_detail, 0) + 1
    
    for key in all_keys:
        sc_data = sc_dict.get(key)
        scd_data = scd_dict.get(key)
        member, direction = key
        direction_exists = bool(direction.strip())
        
        if sc_data is None:
            err = f"[{file_type_prefix_sc} No exist]"
            summary_records.append({"Member code": member, "Direction": direction, "Status": False})
            error_records.append({"Member code": member, "Direction": direction, "Error Detail": err})
            error_counts[err] = error_counts.get(err, 0) + 1
            continue
        if scd_data is None:
            err = f"[{file_type_prefix_scd} No exist]"
            summary_records.append({"Member code": member, "Direction": direction, "Status": False})
            error_records.append({"Member code": member, "Direction": direction, "Error Detail": err})
            error_counts[err] = error_counts.get(err, 0) + 1
            continue
        
        if direction_exists:
            errors = compare_rows(sc_data, scd_data, direction_exists=True)
        else:
            errors = compare_rows(sc_data, scd_data, direction_exists=False)
        
        if errors:
            summary_records.append({"Member code": member, "Direction": direction, "Status": False})
            formatted_errors = []
            for e in errors:
                if e.startswith("[") and ("Duplicate" in e or "No exist" in e):
                    formatted_errors.append(e)
                else:
                    formatted_errors.append(format_error_message(e))
                for err_token in re.findall(r"\[([^\]]+)\]", formatted_errors[-1]):
                    error_counts[err_token] = error_counts.get(err_token, 0) + 1
            error_records.append({"Member code": member, "Direction": direction, "Error Detail": ", ".join(formatted_errors)})
        else:
            summary_records.append({"Member code": member, "Direction": direction, "Status": True})
    
    # 동일한 Member code별로 Direction과 Error Detail을 통합
    merged_summary = {}
    for rec in summary_records:
        key = rec["Member code"]
        if key not in merged_summary:
            merged_summary[key] = {"Member code": key, "Direction": rec["Direction"], "Status": rec["Status"]}
        else:
            merged_summary[key]["Direction"] += f", {rec['Direction']}"
    df_summary = pd.DataFrame(list(merged_summary.values()), columns=["Member code", "Direction", "Status"])
    
    merged_errors = {}
    for rec in error_records:
        key = rec["Member code"]
        if key not in merged_errors:
            merged_errors[key] = {"Member code": key, "Direction": rec["Direction"], "Error Detail": rec["Error Detail"]}
        else:
            existing = set(merged_errors[key]["Error Detail"].split(", "))
            new_errors = set(rec["Error Detail"].split(", "))
            merged = existing.union(new_errors)
            merged_errors[key]["Direction"] += f", {rec['Direction']}"
            merged_errors[key]["Error Detail"] = ", ".join(sorted(merged))
    df_errors = pd.DataFrame(list(merged_errors.values()), columns=["Member code", "Direction", "Error Detail"])
    
    df_error_counts = pd.DataFrame(list(error_counts.items()), columns=["Error Type", "Count"])
    df_error_counts.sort_values(by="Count", ascending=False, inplace=True)
    
    with pd.ExcelWriter(output_excel, engine="xlsxwriter") as writer:
        df_summary.to_excel(writer, sheet_name="Error Summary", index=False)
        df_errors.to_excel(writer, sheet_name="Error Details", index=False)
        df_error_counts.to_excel(writer, sheet_name="Error Count Summary", index=False)
    
    print(f"비교 결과 Excel 파일이 저장되었습니다: {output_excel}")
    print("\n오류 유형별 집계:")
    print(df_error_counts)

##############################
# 사용 예시
##############################
sc_csv_path = r"D:\LLLast\SCD_TRUE_DATA.csv"
scd_csv_path = r"D:\LLLast\SD_TRUE_DATA.csv"
output_excel = r"D:\LLLast\SD\HUMAN ERROR.xlsx"

compare_csv_files_to_excel(sc_csv_path, scd_csv_path, output_excel)
