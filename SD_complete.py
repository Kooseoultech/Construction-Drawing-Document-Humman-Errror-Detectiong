# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 16:51:24 2025

@author: S3PARC
"""


#1) pdf to Img

import os
from pdf2image import convert_from_path

# PDF 파일이 들어 있는 폴더 경로
pdf_folder = r'D:\LLLast\SD\raw data'
output_folder = r'D:\LLLast\SD\raw_split'

# 출력 폴더 생성 (없는 경우)
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# PDF 파일 리스트 가져오기
pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith('.pdf')]

# 각 PDF 파일 처리
for pdf_file in pdf_files:
    pdf_path = os.path.join(pdf_folder, pdf_file)
    pdf_name = os.path.splitext(pdf_file)[0]  # 파일 이름만 추출 (확장자 제외)
    
    # PDF를 이미지로 변환
    images = convert_from_path(pdf_path, dpi=500)
    
    # 각 페이지를 저장
    for i, image in enumerate(images):
        output_path = os.path.join(output_folder, f'{pdf_name}_page_{i + 1}.jpg')
        image.save(output_path, 'JPEG')
    
    print(f'{pdf_file} 처리 완료!')

print("모든 PDF 파일 변환이 완료되었습니다!")
################################################################################
#1-2) 테이블 추출하고 OCR하게 테이블 추출(옵션)


import cv2
import numpy as np
import os

# 최외각선 검출 함수
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 그레이스케일 변환
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)  # 블러링으로 노이즈 제거
    edges = cv2.Canny(blurred, 50, 150)  # Canny Edge Detection
    return edges

def detect_outer_contour(image):
    processed_image = preprocess_image(image)
    
    # 외곽선 검출
    contours, _ = cv2.findContours(processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 가장 큰 외곽선 반환
    outer_contour = max(contours, key=cv2.contourArea) if contours else None
    return outer_contour

# 입력 폴더 및 출력 폴더 설정
input_folder = r"D:\LLLast\SD\raw_split"  # 입력 이미지 폴더
output_folder = r"D:\LLLast\SD\raw_split\table_extraction"  # 결과 저장 폴더
os.makedirs(output_folder, exist_ok=True)

# 폴더 내 모든 이미지 처리
for file_name in os.listdir(input_folder):
    if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        image_path = os.path.join(input_folder, file_name)
        image = cv2.imread(image_path)

        # 1단계: 원본 이미지에서 최외각선 검출 및 크롭
        outer_contour = detect_outer_contour(image)

        if outer_contour is not None:
            x, y, w, h = cv2.boundingRect(outer_contour)  # 최외각선 경계 박스 계산
            cropped_image = image[y:y+h, x:x+w]  # 이미지 크롭

            # 1단계 크롭된 이미지 저장
            cropped_path = os.path.join(output_folder, f"{os.path.splitext(file_name)[0]}_cropped.png")
            cv2.imwrite(cropped_path, cropped_image)

            # 2단계: 크롭된 이미지에서 다시 최외각선 검출
            cropped_outer_contour = detect_outer_contour(cropped_image)

            if cropped_outer_contour is not None:
                # 2단계 최외각선 시각화
                output_image = cropped_image.copy()
                cv2.drawContours(output_image, [cropped_outer_contour], -1, (0, 255, 0), 2)  # 초록색 외곽선

                x2, y2, w2, h2 = cv2.boundingRect(cropped_outer_contour)
                final_cropped_image = cropped_image[y2:y2+h2, x2:x2+w2]  # 2단계 크롭

                # 2단계 최종 크롭된 이미지 저장
                final_output_path = os.path.join(output_folder, f"{os.path.splitext(file_name)[0]}_final_cropped.png")
                cv2.imwrite(final_output_path, final_cropped_image)

                # 3단계: 최종 크롭된 이미지에서 최외각선 검출
                third_outer_contour = detect_outer_contour(final_cropped_image)

                if third_outer_contour is not None:
                    # 3단계 최외각선 시각화
                    final_output_image = final_cropped_image.copy()
                    cv2.drawContours(final_output_image, [third_outer_contour], -1, (0, 0, 255), 2)  # 빨간색 외곽선

                    x3, y3, w3, h3 = cv2.boundingRect(third_outer_contour)
                    third_cropped_image = final_cropped_image[y3:y3+h3, x3:x3+w3]  # 최종 크롭

                    # 최종 크롭된 이미지 저장
                    third_output_path = os.path.join(output_folder, f"{os.path.splitext(file_name)[0]}_third_final_cropped.png")
                    cv2.imwrite(third_output_path, third_cropped_image)

                    print(f"Processed {file_name}: 최종 크롭된 이미지 저장됨: {third_output_path}")
                else:
                    print(f"3단계: {file_name}에서 최종 크롭된 이미지에서 최외각선을 찾을 수 없습니다.")
            else:
                print(f"2단계: {file_name}에서 크롭된 이미지에서 최외각선을 찾을 수 없습니다.")
        else:
            print(f"1단계: {file_name}에서 원본 이미지에서 최외각선을 찾을 수 없습니다.")

print("모든 이미지 처리가 완료되었습니다.")

################################################################################
#2) surya ocr 바로 적용

import subprocess
from tqdm import tqdm
import re

# surya_ocr 명령어 및 인자 설정
command = "surya_ocr"
argument = r"D:\LLLast\SD\raw_split\table_extraction"

# surya_ocr 명령어 실행 (진행 정보가 stdout에 출력되어야 함)
process = subprocess.Popen([command, argument],
                           stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT,
                           text=True,
                           shell=True)

# 예를 들어, surya_ocr가 "Progress: 45%" 형식의 문자열을 출력한다고 가정하면,
# tqdm 진행률 표시를 위해 초기 total은 100(%)로 설정할 수 있습니다.
pbar = tqdm(total=100)

# 출력 라인을 실시간으로 읽으며 진행률 업데이트
while True:
    line = process.stdout.readline()
    if not line:
        break
    print(line, end="")  # 터미널에 출력 (원하는 경우)
    # "Progress: XX%"를 찾아서 진행률 업데이트
    match = re.search(r"Progress:\s*(\d+)%", line)
    if match:
        progress = int(match.group(1))
        pbar.n = progress
        pbar.refresh()

process.wait()
pbar.close()
###################################################################################
###################################################################################
#3) 파일별로 suryaocr 쪼개기

import json
import os

def split_surya_ocr_result(combined_json_path, output_folder):
    # 통합 결과 파일 읽기
    with open(combined_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 출력 폴더 생성
    os.makedirs(output_folder, exist_ok=True)

    # data가 dict 형태라고 가정 (각 key가 파일명 또는 식별자)
    for key, value in data.items():
        # key를 기반으로 파일명 생성 (예: key가 "page_1_filename"이면 "page_1_filename.json")
        output_path = os.path.join(output_folder, f"{key}.json")
        with open(output_path, "w", encoding="utf-8") as out_f:
            json.dump(value, out_f, ensure_ascii=False, indent=4)
        print(f"저장됨: {output_path}")

# 사용 예시:
if __name__ == "__main__":
    combined_json_path = r"D:\LLLast\SD\surya\raw\results.json"
    output_folder = r"D:\LLLast\SD\surya\raw_split"
    split_surya_ocr_result(combined_json_path, output_folder)



###############################################################################
#############################


###################################################################################################################
#자 그럼이제 기준 높이 땄으니, 이제 내가 원하는 텍스트 셀만 남기면 되는거임 

import cv2
import json
import os
import re
import numpy as np

########################################
# [A] 텍스트 박스 및 Member Code 추출 (JSON 기반)
########################################
def get_center(bbox):
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

def extract_all_text_boxes(json_file):
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    boxes = []
    if isinstance(data, dict):
        iterable = data.items()
    elif isinstance(data, list):
        iterable = enumerate(data)
    else:
        iterable = []
    for key, results in iterable:
        if isinstance(results, list):
            for result in results:
                if "text_lines" in result:
                    for line in result["text_lines"]:
                        if "bbox" in line and "text" in line:
                            bbox = list(map(int, line["bbox"]))
                            text = line["text"]
                            boxes.append({
                                "text": text,
                                "bbox": bbox,
                                "center": get_center(bbox)
                            })
        elif isinstance(results, dict):
            if "text_lines" in results:
                for line in results["text_lines"]:
                    if "bbox" in line and "text" in line:
                        bbox = list(map(int, line["bbox"]))
                        text = line["text"]
                        boxes.append({
                            "text": text,
                            "bbox": bbox,
                            "center": get_center(bbox)
                        })
    return boxes

def is_member_code(text):
    # 멤버 코드로 인식하기 위해 금지 단어가 없어야 함
    text_lower = text.lower()
    forbidden = ["@", "hd", "suhd", "shd", "d", "center", "cen", "both", "ext.", "int.", "mid", "all"]
    for word in forbidden:
        if word in text_lower:
            return False
    return True

def extract_member_code_boxes(json_file):
    # 전체 텍스트 박스 중 멤버 코드 조건을 만족하는 박스 추출
    boxes = extract_all_text_boxes(json_file)
    member_boxes = [box for box in boxes if is_member_code(box["text"])]
    return member_boxes

########################################
# [B] TC/C 텍스트 추출 (JSON 기반, 전체 OCR 결과에서)
########################################
def extract_tc_c_entries(json_file):
    with open(json_file, "r", encoding="utf-8") as f:
        json_data = json.load(f)
    tc_c_entries = []
    pattern = re.compile(r'^(TC\d+[A-Za-z0-9]*|C\d+[A-Za-z0-9]*|C\d+(,\s*C\d+)*)$', re.IGNORECASE)
    boxes = extract_all_text_boxes(json_file)
    for item in boxes:
        text = item.get("text", "").strip()
        bbox = item.get("bbox", [])
        if pattern.match(text) and len(bbox) == 4:
            tc_c_entries.append({
                "text": text,
                "bbox": bbox,
                "center": get_center(bbox)
            })
    return tc_c_entries

########################################
# [C] Member Code와 TC/C 결합 (Association)
########################################
def associate_member_and_tc(member_boxes, tc_c_entries, y_margin):
    """
    각 member box(멤버 코드 박스)의 하단(y_max) 좌표를 기준으로,
    그 아래 y_margin 이내에 있는 TC/C 텍스트 항목을 찾아서 결합합니다.
    여러 member box가 있을 경우 각각 독립적으로 계산하여,
    한 멤버 박스에 여러 TC/C 텍스트가 매칭될 수 있습니다.
    """
    associations = []  # 각 항목은 dict: {"member_box": box, "tc_entries": [텍스트, ...]}
    for box in member_boxes:
        bx_min, by_min, bx_max, by_max = box["bbox"]
        baseline = by_max  # member code 박스의 하단 좌표
        matched = []
        for entry in tc_c_entries:
            tc_y_min = entry["bbox"][1]
            if baseline < tc_y_min <= baseline + y_margin:
                matched.append(entry["text"])
        associations.append({
            "member_box": box,
            "matched_tc": matched  # 매칭된 TC/C 텍스트 리스트 (없을 수도 있음)
        })
    return associations

########################################
# 예제: JSON 파일별로 Member Code와 TC/C 추출 및 결합 결과 출력
########################################
def process_json_file(json_file, y_margin=900):
    print(f"\n--- Processing JSON file: {json_file} ---")
    member_boxes = extract_member_code_boxes(json_file)
    tc_c_entries = extract_tc_c_entries(json_file)
    
    print("Extracted Member Code Boxes:")
    for mb in member_boxes:
        print(f"Text: '{mb['text']}', BBox: {mb['bbox']}")
    
    print("\nExtracted TC/C Entries:")
    for tc in tc_c_entries:
        print(f"Text: '{tc['text']}', BBox: {tc['bbox']}")
    
    associations = associate_member_and_tc(member_boxes, tc_c_entries, y_margin)
    
    print("\nAssociation Results:")
    for assoc in associations:
        member_text = assoc["member_box"]["text"]
        matched = assoc["matched_tc"]
        print(f"Member '{member_text}' -> Matched TC/C: {matched}")
    
    return associations

########################################
# [D] Main 실행: 폴더 내의 모든 JSON 파일 처리
########################################
def main():
    json_folder = r"D:\LLLast\SD\surya\raw_split"  # JSON 파일 폴더 경로 (필요에 맞게 수정)
    
    json_files = [os.path.join(json_folder, f) for f in os.listdir(json_folder) if f.lower().endswith('.json')]
    for jf in json_files:
        associations = process_json_file(jf, y_margin=900)
        # 여기서 associations 결과를 이용해 파일명에 TC/C 텍스트를 반영하는 등 후속 처리를 할 수 있습니다.
        # 예를 들어, 각 member 박스별로 출력된 matched_tc 리스트를 파일명에 붙이거나 저장하는 로직을 추가하면 됩니다.
        
if __name__ == "__main__":
    main()






######################################################################## 밑에코드  tc/c가 안붙는다 싀발

import cv2
import json
import os
import re
import numpy as np
import math

########################################
# [A] 텍스트 박스 추출 및 글로벌 기준 계산
########################################
def get_center(bbox):
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

def extract_all_text_boxes(json_file):
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    boxes = []
    if isinstance(data, dict):
        iterable = data.items()
    elif isinstance(data, list):
        iterable = enumerate(data)
    else:
        iterable = []
    for key, results in iterable:
        if isinstance(results, list):
            for result in results:
                if "text_lines" in result:
                    for line in result["text_lines"]:
                        if "bbox" in line and "text" in line:
                            bbox = list(map(int, line["bbox"]))
                            text = line["text"]
                            boxes.append({
                                "text": text,
                                "bbox": bbox,
                                "center": get_center(bbox)
                            })
        elif isinstance(results, dict):
            if "text_lines" in results:
                for line in results["text_lines"]:
                    if "bbox" in line and "text" in line:
                        bbox = list(map(int, line["bbox"]))
                        text = line["text"]
                        boxes.append({
                            "text": text,
                            "bbox": bbox,
                            "center": get_center(bbox)
                        })
    return boxes

def extract_keyword_y_positions(json_file, keywords=["부호", "구분"]):
    boxes = extract_all_text_boxes(json_file)
    y_positions = []
    for box in boxes:
        normalized = re.sub(r"\s+", "", box["text"])
        if any(kw in normalized for kw in keywords):
            y_positions.append(box["center"][1])
    return y_positions

def compute_global_reference_and_gap_advanced(json_folder, keywords=["부호", "구분"],
                                              tolerance_ratio=0.02, baseline_height=1000,
                                              same_line_threshold=20):
    """
    - 모든 JSON 파일에서 '부호','구분' 텍스트 박스의 y좌표를 모아 글로벌 시작 y와 gap을 계산
    """
    tolerance = baseline_height * tolerance_ratio
    all_y = []
    json_files = [os.path.join(json_folder, f) for f in os.listdir(json_folder)
                  if f.lower().endswith('.json')]
    for json_file in json_files:
        ys = extract_keyword_y_positions(json_file, keywords)
        all_y.extend(ys)
    if not all_y:
        print("키워드가 포함된 텍스트 박스가 하나도 없습니다.")
        return None, None
    
    all_y = sorted(all_y)
    min_y = all_y[0]
    similar_y = [y for y in all_y if (y - min_y) <= tolerance]
    global_start_y = np.mean(similar_y)
    
    clusters = []
    current_cluster = [all_y[0]]
    for y in all_y[1:]:
        if y - current_cluster[-1] <= same_line_threshold:
            current_cluster.append(y)
        else:
            clusters.append(np.mean(current_cluster))
            current_cluster = [y]
    if current_cluster:
        clusters.append(np.mean(current_cluster))
    
    if len(clusters) < 2:
        gap = tolerance
    else:
        diffs = [clusters[i] - clusters[i-1] for i in range(1, len(clusters))
                 if (clusters[i] - clusters[i-1]) >= same_line_threshold]
        gap = min(diffs) if diffs else tolerance
    return global_start_y, gap

def generate_vertical_points_from_reference(global_start_y, gap, image_height):
    points = [global_start_y]
    y = global_start_y - gap
    while y >= 0:
        points.append(y)
        y -= gap
    y = global_start_y + gap
    while y < image_height:
        points.append(y)
        y += gap
    return sorted(points)

def filter_text_boxes_by_vertical_points(boxes, vertical_points, tolerance_margin=10):
    filtered = []
    for box in boxes:
        cy = box["center"][1]
        for vp in vertical_points:
            if abs(cy - vp) <= tolerance_margin:
                filtered.append(box)
                break
    return filtered

def is_member_code(text):
    text_lower = text.lower()
    # 금지 단어가 포함되면 멤버 코드가 아님
    forbidden = ["@", "hd", "suhd", "shd", "d", "center", "cen", "both", "ext.", "int.", "mid", "all"]
    for word in forbidden:
        if word in text_lower:
            return False
    return True

def filter_member_code_boxes(boxes):
    return [box for box in boxes if is_member_code(box["text"])]

def annotate_image_with_boxes(image, boxes, box_color=(0,255,255), thickness=2,
                              font_scale=0.5, text_color=(0,255,0)):
    annotated = image.copy()
    for box in boxes:
        x_min, y_min, x_max, y_max = map(int, box["bbox"])
        cv2.rectangle(annotated, (x_min, y_min), (x_max, y_max), box_color, thickness)
        cv2.putText(annotated, box["text"], (x_min, y_min-10),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness, cv2.LINE_AA)
    return annotated

########################################
# [B] 테이블 셀(후보 사각형) 검출
########################################
def is_connected(p1, p2, orientation, binary_image):
    if orientation == "horizontal":
        y = int(p1[1])
        x1, x2 = sorted([int(p1[0]), int(p2[0])])
        if y < 0 or y >= binary_image.shape[0]:
            return False
        line = binary_image[y, x1:x2+1]
    elif orientation == "vertical":
        x = int(p1[0])
        y1, y2 = sorted([int(p1[1]), int(p2[1])])
        if x < 0 or x >= binary_image.shape[1]:
            return False
        line = binary_image[y1:y2+1, x]
    else:
        return False
    return np.all(line > 0)

def find_valid_rectangles(points, horizontal_image, vertical_image):
    rectangles = []
    # 교차점들을 y,x 순으로 정렬
    points = sorted(points, key=lambda p: (p[1], p[0]))
    for p1 in points:
        for p2 in points:
            if p1[0] == p2[0] and p1[1] < p2[1]:
                if is_connected(p1, p2, "vertical", vertical_image):
                    for p3 in points:
                        if p2[1] == p3[1] and p2[0] < p3[0]:
                            if is_connected(p2, p3, "horizontal", horizontal_image):
                                for p4 in points:
                                    if p1[1] == p4[1] and p1[0] < p4[0] and p3[0] == p4[0]:
                                        if (is_connected(p4, p1, "horizontal", horizontal_image) and 
                                            is_connected(p1, p2, "vertical", vertical_image) and 
                                            is_connected(p2, p3, "horizontal", horizontal_image) and 
                                            is_connected(p3, p4, "vertical", vertical_image)):
                                            rectangles.append((p1, p2, p3, p4))
    return rectangles

########################################
# filter_rectangles_by_member_boxes
# 후보 사각형 중, 멤버 박스를 포함하고 높이 제한 이하인 사각형만 반환
########################################
def filter_rectangles_by_member_boxes(rectangles, vertical_points, member_boxes,
                                      max_cell_height, tolerance=10):
    filtered_rectangles = []
    for mb in member_boxes:
        mb_xmin, mb_ymin, mb_xmax, mb_ymax = mb["bbox"]
        smallest_rect = None
        smallest_area = float("inf")
        for rect in rectangles:
            rx_min = min(pt[0] for pt in rect)
            ry_min = min(pt[1] for pt in rect)
            rx_max = max(pt[0] for pt in rect)
            ry_max = max(pt[1] for pt in rect)
            # 멤버 박스가 후보 셀 내부에 있는지 (약간의 tolerance)
            inside = (
                rx_min - tolerance <= mb_xmin <= rx_max + tolerance and
                rx_min - tolerance <= mb_xmax <= rx_max + tolerance and
                ry_min - tolerance <= mb_ymin <= ry_max + tolerance and
                ry_min - tolerance <= mb_ymax <= ry_max + tolerance
            )
            cell_height = ry_max - ry_min
            if inside and cell_height <= max_cell_height:
                area = (rx_max - rx_min) * (ry_max - ry_min)
                if area < smallest_area:
                    smallest_rect = rect
                    smallest_area = area
        if smallest_rect and smallest_rect not in filtered_rectangles:
            filtered_rectangles.append(smallest_rect)
    return filtered_rectangles

########################################
# 'TC'/'C' 텍스트 항목 추출
########################################
def extract_tc_c_entries(json_data):
    tc_c_entries = []
    pattern = re.compile(r'^(TC\d+[A-Za-z0-9]*|C\d+[A-Za-z0-9]*|C\d+(,\s*C\d+)*)$', re.IGNORECASE)
    for item in json_data:
        text = item.get("text", "").strip()
        bbox = item.get("bbox", [])
        if pattern.match(text) and len(bbox) == 4:
            x_min, y_min, x_max, y_max = bbox
            center_x = (x_min + x_max) / 2
            center_y = (y_min + y_max) / 2
            tc_c_entries.append({
                "text": text,
                "bbox": bbox,
                "center": (center_x, center_y)
            })
    return tc_c_entries

########################################
# 다수의 멤버 코드 박스 baseline 사용:
# 하나의 후보 셀 내부에 여러 member box가 있으면, 각 box의 y_max로 TC/C 텍스트 매칭
########################################
def associate_with_rectangles(tc_c_entries, filtered_rectangles, member_boxes, y_margin):
    associations = {}
    for idx, rect in enumerate(filtered_rectangles):
        # 후보 셀 좌표
        rx_min = min(pt[0] for pt in rect)
        ry_min = min(pt[1] for pt in rect)
        rx_max = max(pt[0] for pt in rect)
        ry_max = max(pt[1] for pt in rect)
        
        # 후보 셀 내부 모든 멤버 박스의 하단 좌표를 수집
        all_baselines = []
        for mb in member_boxes:
            bx_min, by_min, bx_max, by_max = mb["bbox"]
            # 셀 내부인지 간단히 검사
            if (bx_min >= rx_min and bx_max <= rx_max and
                by_min >= ry_min and by_max <= ry_max):
                all_baselines.append(by_max)
        
        # 멤버 박스가 하나도 없으면 후보 셀 자체의 y_max 사용
        if not all_baselines:
            all_baselines = [ry_max]
        
        matched_texts = set()  # 중복 제거
        for baseline in all_baselines:
            print(f"Candidate {idx+1} baseline: {baseline}")
            for entry in tc_c_entries:
                tc_y_min = entry["bbox"][1]
                # baseline < tc_y_min <= baseline + y_margin
                if baseline < tc_y_min <= baseline + y_margin:
                    matched_texts.add(entry["text"])
        
        if matched_texts:
            associations[idx] = list(matched_texts)
            print(f"  => Candidate {idx+1} matched texts: {associations[idx]}")
        else:
            print(f"  => Candidate {idx+1} no match found.")
    return associations

########################################
# 후보 셀 크롭 이미지 저장 (파일명에 매칭된 텍스트 반영)
########################################
def save_images_with_tc_c(output_base_folder, json_filename, filtered_rectangles, associations, original_image):
    output_folder = os.path.join(output_base_folder, os.path.splitext(json_filename)[0])
    os.makedirs(output_folder, exist_ok=True)
    for idx, rect in enumerate(filtered_rectangles):
        rx_min = int(min(pt[0] for pt in rect))
        ry_min = int(min(pt[1] for pt in rect))
        rx_max = int(max(pt[0] for pt in rect))
        ry_max = int(max(pt[1] for pt in rect))
        cropped_image = original_image[ry_min:ry_max, rx_min:rx_max]
        
        matched = associations.get(idx, None)
        if matched:
            text_suffix = "_".join([t.strip() for t in matched])
            output_filename = f"filtered_rectangle_{idx + 1}_{text_suffix}.png"
            print(f"Candidate {idx+1}: File will be saved as {output_filename}")
        else:
            output_filename = f"filtered_rectangle_{idx+1}.png"
        
        out_path = os.path.join(output_folder, output_filename)
        cv2.imwrite(out_path, cropped_image)
        print(f"{json_filename} - 저장된 최종 이미지: {out_path}")

########################################
# 후보 셀(빨간 사각형)만 별도로 추출 (파일명/오버레이에 매칭 텍스트 반영)
########################################
def save_candidate_cells_with_associations(output_base_folder, json_filename,
                                           filtered_rectangles, original_image, associations):
    candidate_folder = os.path.join(output_base_folder, os.path.splitext(json_filename)[0], "CandidateCells")
    os.makedirs(candidate_folder, exist_ok=True)
    
    if not filtered_rectangles:
        print("  => 후보 셀이 없습니다.")
        return
    
    for idx, rect in enumerate(filtered_rectangles):
        rx_min = int(min(pt[0] for pt in rect))
        ry_min = int(min(pt[1] for pt in rect))
        rx_max = int(max(pt[0] for pt in rect))
        ry_max = int(max(pt[1] for pt in rect))
        
        cropped_image = original_image[ry_min:ry_max, rx_min:rx_max].copy()
        
        matched = associations.get(idx, None)
        if matched:
            text_str = ", ".join([t.strip() for t in matched])
            cv2.putText(cropped_image, text_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
            candidate_filename = f"candidate_cell_{idx+1}_{text_str}.png"
            print(f"Candidate {idx+1}: File will be saved as {candidate_filename}")
        else:
            candidate_filename = f"candidate_cell_{idx+1}.png"
        
        candidate_path = os.path.join(candidate_folder, candidate_filename)
        print(f"  => 후보 셀 {idx+1} 좌표: x_min={rx_min}, x_max={rx_max}, y_min={ry_min}, y_max={ry_max}")
        cv2.imwrite(candidate_path, cropped_image)
        print(f"  => Candidate cell {idx+1} 저장됨: {candidate_path}")

########################################
# [C] 메인 실행
########################################
def main():
    # 경로 설정
    json_folder = r"D:\LLLast\SD\surya\raw_split"       # JSON 파일 폴더
    image_folder = r"D:\LLLast\SD\raw_split"       # 이미지 폴더
    output_base_folder = r"D:\LLLast\SD\Member code_extraction"
    os.makedirs(output_base_folder, exist_ok=True)
    
    # 글로벌 기준 y와 gap 계산
    global_start_y, gap = compute_global_reference_and_gap_advanced(
        json_folder, keywords=["부호", "구분"],
        tolerance_ratio=0.02, baseline_height=1000, same_line_threshold=20
    )
    if global_start_y is None or gap is None:
        print("글로벌 기준 값을 계산할 수 없습니다.")
        return
    print(f"[글로벌 기준] 시작 y: {global_start_y:.2f}, gap: {gap:.2f}")
    
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png','.jpg','.jpeg'))]
    if not image_files:
        print("이미지 파일이 없습니다.")
        return
    
    for img_file in image_files:
        base_name = os.path.splitext(img_file)[0]
        json_file = os.path.join(json_folder, base_name + ".json")
        if not os.path.exists(json_file):
            print(f"[{base_name}] JSON 파일이 없습니다.")
            continue
        
        image_path = os.path.join(image_folder, img_file)
        image = cv2.imread(image_path)
        if image is None:
            print(f"[{base_name}] 이미지를 로드할 수 없습니다: {image_path}")
            continue
        
        print(f"\n=== 처리 중: {base_name} ===")
        
        # 1) 텍스트 박스 추출 + 멤버 박스 필터링
        boxes = extract_all_text_boxes(json_file)
        vertical_points = generate_vertical_points_from_reference(global_start_y, gap, image.shape[0])
        filtered_boxes = filter_text_boxes_by_vertical_points(boxes, vertical_points, tolerance_margin=50)
        member_boxes = filter_member_code_boxes(filtered_boxes)
        print(f"  => 멤버 코드 박스 수: {len(member_boxes)}")
        
        # (옵션) 멤버 박스 정보 저장
        member_json_out = os.path.join(output_base_folder, base_name + '_member_boxes.json')
        with open(member_json_out, 'w', encoding='utf-8') as f:
            json.dump(member_boxes, f, ensure_ascii=False, indent=4)
        
        # 2) 이미지 이진화 + 교차점 추출
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
        vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
        intersections = cv2.bitwise_and(horizontal_lines, vertical_lines)
        contours, _ = cv2.findContours(intersections, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        intersection_points = [(x + w // 2, y + h // 2)
                               for x, y, w, h in [cv2.boundingRect(cnt) for cnt in contours]]
        
        # 3) 후보 사각형 찾기 + 멤버 박스 포함 셀 필터
        valid_rectangles = find_valid_rectangles(intersection_points, horizontal_lines, vertical_lines)
        print(f"  => 후보 사각형(셀) 수: {len(valid_rectangles)}")
        
        max_cell_height = 150
        filtered_rectangles = filter_rectangles_by_member_boxes(
            valid_rectangles, vertical_points, member_boxes, max_cell_height, tolerance=10
        )
        print(f"  => 멤버 박스 포함 후보 셀 수: {len(filtered_rectangles)}")
        
        # 4) 전체 OCR에서 TC/C 텍스트 추출 + association
        with open(json_file, "r", encoding="utf-8") as f:
            json_data = json.load(f)
        tc_c_entries = extract_tc_c_entries(json_data)
        y_margin = 3900
        associations = associate_with_rectangles(tc_c_entries, filtered_rectangles, member_boxes, y_margin)
        
        # 5) 디버깅용: 전체 이미지에 후보 셀 빨간 사각형 + 매칭 텍스트 표시
        annotated_candidate = image.copy()
        for idx, rect in enumerate(filtered_rectangles):
            rx_min = int(min(pt[0] for pt in rect))
            ry_min = int(min(pt[1] for pt in rect))
            rx_max = int(max(pt[0] for pt in rect))
            ry_max = int(max(pt[1] for pt in rect))
            cv2.rectangle(annotated_candidate, (rx_min, ry_min), (rx_max, ry_max), (0, 0, 255), 3)
            if idx in associations:
                text_str = ", ".join(associations[idx])
                cv2.putText(annotated_candidate, text_str, (rx_min, ry_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2, cv2.LINE_AA)
        
        candidate_out = os.path.join(output_base_folder, base_name + "_candidate_cells.png")
        cv2.imwrite(candidate_out, annotated_candidate)
        print(f"  => 전체 이미지(빨간 사각형 + 매칭 텍스트) 저장: {candidate_out}")
        
        # 6) 후보 셀만 별도 크롭: 오버레이 + 파일명에 매칭 텍스트 반영
        save_candidate_cells_with_associations(
            output_base_folder, base_name + ".json",
            filtered_rectangles, image, associations
        )
        
        # 7) 최종 결과: 셀 크롭 (파일명에 매칭 텍스트)
        save_images_with_tc_c(
            output_base_folder, base_name + ".json",
            filtered_rectangles, associations, image
        )
        
        # (디버깅) 수직 점 + 멤버 박스 표시
        annotated_image = annotate_image_with_boxes(
            image.copy(), member_boxes,
            box_color=(0,255,255), thickness=2, font_scale=0.5, text_color=(0,255,0)
        )
        cx = image.shape[1] // 2
        for vp in vertical_points:
            cv2.circle(annotated_image, (cx, int(vp)), 5, (255,0,0), -1)
            cv2.putText(annotated_image, str(int(vp)), (cx+5, int(vp)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1, cv2.LINE_AA)
        annotated_out = os.path.join(output_base_folder, base_name + "_vertical_textboxes.png")
        cv2.imwrite(annotated_out, annotated_image)
        print(f"  => 수직 점 및 멤버 박스 표시 이미지 저장: {annotated_out}")
    
    print("\n모든 JSON 파일 처리가 완료되었습니다.")

if __name__ == "__main__":
    main()
##################################################################################################################'
#이게 지금 매번 인식이 다르게 나와서 문제임=> 일단 테이블 추출 





##############################################################################################################


import cv2
import json
import os
import re
import numpy as np
import math
import csv

########################################
# [NEW] 전역으로 "부호"/"구분" x좌표 범위를 수집
########################################
def gather_global_keyword_xrange(json_folder, keywords=["부호", "구분"]):
    """
    모든 JSON 파일을 읽어, '부호' 또는 '구분'이 포함된 박스의 x_min, x_max를 전역으로 수집하여
    전체 최소/최대 x 좌표를 반환한다.
    """
    global_min_x = None
    global_max_x = None
    
    json_files = [f for f in os.listdir(json_folder) if f.lower().endswith('.json')]
    for jf in json_files:
        path = os.path.join(json_folder, jf)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # 텍스트 박스 추출
        boxes = []
        if isinstance(data, dict):
            iterable = data.items()
        elif isinstance(data, list):
            iterable = enumerate(data)
        else:
            iterable = []
        for key, results in iterable:
            if isinstance(results, list):
                for result in results:
                    if "text_lines" in result:
                        for line in result["text_lines"]:
                            if "bbox" in line and "text" in line:
                                try:
                                    bbox = list(map(int, line["bbox"]))
                                except:
                                    continue
                                text = line["text"]
                                normalized = re.sub(r"\s+", "", text)
                                if any(kw in normalized for kw in keywords):
                                    x0, y0, x1, y1 = bbox
                                    if global_min_x is None or x0 < global_min_x:
                                        global_min_x = x0
                                    if global_max_x is None or x1 > global_max_x:
                                        global_max_x = x1
            elif isinstance(results, dict):
                if "text_lines" in results:
                    for line in results["text_lines"]:
                        if "bbox" in line and "text" in line:
                            try:
                                bbox = list(map(int, line["bbox"]))
                            except:
                                continue
                            text = line["text"]
                            normalized = re.sub(r"\s+", "", text)
                            if any(kw in normalized for kw in keywords):
                                x0, y0, x1, y1 = bbox
                                if global_min_x is None or x0 < global_min_x:
                                    global_min_x = x0
                                if global_max_x is None or x1 > global_max_x:
                                    global_max_x = x1
    return global_min_x, global_max_x

def filter_boxes_by_global_xrange(boxes, global_min_x, global_max_x):
    """
    전역으로 수집된 [global_min_x, global_max_x] 범위에 박스의 중심 x가 속하면 제거.
    """
    if global_min_x is None or global_max_x is None:
        return boxes  # 수집된 범위가 없으면 그대로 반환
    filtered = []
    for box in boxes:
        cx = box["center"][0]
        if not (global_min_x <= cx <= global_max_x):
            filtered.append(box)
    return filtered


########################################
# [A] 문자열 교정: "TCS" -> "TC5", "CS" -> "C5"
########################################
def correct_tcs_to_tc5(text):
    upper_txt = text.upper()
    if upper_txt.startswith("TCS"):
        return "TC5" + text[3:]
    elif upper_txt.startswith("CS"):
        return "C5" + text[2:]
    else:
        return text

########################################
# [B] 텍스트 박스 추출 등 (기존 코드)
########################################
def get_center(bbox):
    return ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)

def extract_all_text_boxes(json_file):
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    boxes = []
    if isinstance(data, dict):
        iterable = data.items()
    elif isinstance(data, list):
        iterable = enumerate(data)
    else:
        iterable = []
    for key, results in iterable:
        if isinstance(results, list):
            for result in results:
                if "text_lines" in result:
                    for line in result["text_lines"]:
                        if "bbox" in line and "text" in line:
                            try:
                                bbox = list(map(int, line["bbox"]))
                            except:
                                continue
                            text = line["text"]
                            boxes.append({
                                "text": text,
                                "bbox": bbox,
                                "center": get_center(bbox)
                            })
        elif isinstance(results, dict):
            if "text_lines" in results:
                for line in results["text_lines"]:
                    if "bbox" in line and "text" in line:
                        try:
                            bbox = list(map(int, line["bbox"]))
                        except:
                            continue
                        text = line["text"]
                        boxes.append({
                            "text": text,
                            "bbox": bbox,
                            "center": get_center(bbox)
                        })
    return boxes

def extract_keyword_y_positions(json_file, keywords=["부호", "구분"]):
    boxes = extract_all_text_boxes(json_file)
    y_positions = []
    for box in boxes:
        normalized = re.sub(r"\s+", "", box["text"])
        if any(kw in normalized for kw in keywords):
            y_positions.append(box["center"][1])
    return y_positions

def compute_global_reference_and_gap_advanced(json_folder, keywords=["부호", "구분"],
                                              tolerance_ratio=0.02, baseline_height=1000,
                                              same_line_threshold=20):
    tolerance = baseline_height * tolerance_ratio
    all_y = []
    json_files = [os.path.join(json_folder, f) for f in os.listdir(json_folder)
                  if f.lower().endswith('.json')]
    for json_file in json_files:
        ys = extract_keyword_y_positions(json_file, keywords)
        all_y.extend(ys)
    if not all_y:
        print("키워드가 포함된 텍스트 박스가 하나도 없습니다.")
        return None, None
    all_y = sorted(all_y)
    min_y = all_y[0]
    similar_y = [y for y in all_y if (y - min_y) <= tolerance]
    global_start_y = np.mean(similar_y)
    
    clusters = []
    current_cluster = [all_y[0]]
    for y in all_y[1:]:
        if y - current_cluster[-1] <= same_line_threshold:
            current_cluster.append(y)
        else:
            clusters.append(np.mean(current_cluster))
            current_cluster = [y]
    if current_cluster:
        clusters.append(np.mean(current_cluster))
    
    if len(clusters) < 2:
        gap = tolerance
    else:
        diffs = [clusters[i] - clusters[i-1] for i in range(1, len(clusters))
                 if (clusters[i] - clusters[i-1]) >= same_line_threshold]
        gap = min(diffs) if diffs else tolerance
    return global_start_y, gap

def generate_vertical_points_from_reference(global_start_y, gap, image_height):
    points = [global_start_y]
    y = global_start_y - gap
    while y >= 0:
        points.append(y)
        y -= gap
    y = global_start_y + gap
    while y < image_height:
        points.append(y)
        y += gap
    return sorted(points)

def filter_text_boxes_by_vertical_points(boxes, vertical_points, tolerance_margin=10):
    filtered = []
    for box in boxes:
        cy = box["center"][1]
        for vp in vertical_points:
            if abs(cy - vp) <= tolerance_margin:
                filtered.append(box)
                break
    return filtered

########################################
# [C] Member Code 추출
########################################
def is_member_code(text):
    text_lower = text.lower()
    forbidden = ["@", "hd", "suhd", "shd", "d", "center", "cen", "both", "ext.", "int.", "mid", "all"]
    for word in forbidden:
        if word in text_lower:
            return False
    return True

def filter_member_code_boxes(boxes):
    return [box for box in boxes if is_member_code(box["text"])]

########################################
# [D] TC/C 텍스트 항목 추출 (교정 포함)
########################################
def extract_tc_c_entries(json_file):
    with open(json_file, "r", encoding="utf-8") as f:
        json_data = json.load(f)
    tc_c_entries = []
    pattern = re.compile(r'^(TC\d+[A-Za-z0-9]*|C\d+[A-Za-z0-9]*|C\d+(,\s*C\d+)*)$', re.IGNORECASE)
    boxes = extract_all_text_boxes(json_file)
    for item in boxes:
        original_text = item.get("text", "").strip()
        corrected_text = correct_tcs_to_tc5(original_text)
        bbox = item.get("bbox", [])
        if len(bbox) == 4 and pattern.match(corrected_text):
            tc_c_entries.append({
                "text": corrected_text,
                "bbox": bbox,
                "center": get_center(bbox)
            })
    return tc_c_entries

########################################
# [E] 테이블 셀(후보 사각형) 검출
########################################
def is_connected(p1, p2, orientation, binary_image):
    if orientation == "horizontal":
        y = int(p1[1])
        x1, x2 = sorted([int(p1[0]), int(p2[0])])
        if y < 0 or y >= binary_image.shape[0]:
            return False
        line = binary_image[y, x1:x2+1]
    elif orientation == "vertical":
        x = int(p1[0])
        y1, y2 = sorted([int(p1[1]), int(p2[1])])
        if x < 0 or x >= binary_image.shape[1]:
            return False
        line = binary_image[y1:y2+1, x]
    else:
        return False
    return np.all(line > 0)

def find_valid_rectangles(points, horizontal_image, vertical_image):
    rectangles = []
    points = sorted(points, key=lambda p: (p[1], p[0]))
    for p1 in points:
        for p2 in points:
            if p1[0] == p2[0] and p1[1] < p2[1]:
                if is_connected(p1, p2, "vertical", vertical_image):
                    for p3 in points:
                        if p2[1] == p3[1] and p2[0] < p3[0]:
                            if is_connected(p2, p3, "horizontal", horizontal_image):
                                for p4 in points:
                                    if p1[1] == p4[1] and p1[0] < p4[0] and p3[0] == p4[0]:
                                        if (is_connected(p4, p1, "horizontal", horizontal_image) and 
                                            is_connected(p1, p2, "vertical", vertical_image) and 
                                            is_connected(p2, p3, "horizontal", horizontal_image) and 
                                            is_connected(p3, p4, "vertical", vertical_image)):
                                            rectangles.append((p1, p2, p3, p4))
    return rectangles

########################################
# [F] 후보 셀 필터링: 멤버 박스를 포함하는 (가장 작은) 셀
########################################
def filter_rectangles_by_member_boxes(rectangles, vertical_points, member_boxes, max_cell_height, tolerance=10):
    filtered_rectangles = []
    for mb in member_boxes:
        mb_xmin, mb_ymin, mb_xmax, mb_ymax = mb["bbox"]
        smallest_rect = None
        smallest_area = float("inf")
        for rect in rectangles:
            rx_min = min(pt[0] for pt in rect)
            ry_min = min(pt[1] for pt in rect)
            rx_max = max(pt[0] for pt in rect)
            ry_max = max(pt[1] for pt in rect)
            inside = (
                rx_min - tolerance <= mb_xmin <= rx_max + tolerance and
                rx_min - tolerance <= mb_xmax <= rx_max + tolerance and
                ry_min - tolerance <= mb_ymin <= ry_max + tolerance and
                ry_min - tolerance <= mb_ymax <= ry_max + tolerance
            )
            cell_height = ry_max - ry_min
            if inside and cell_height <= max_cell_height:
                area = (rx_max - rx_min) * (ry_max - ry_min)
                if area < smallest_area:
                    smallest_rect = rect
                    smallest_area = area
        if smallest_rect and smallest_rect not in filtered_rectangles:
            filtered_rectangles.append(smallest_rect)
    return filtered_rectangles

########################################
# [G] Association: 후보 셀 내의 멤버 박스 기준으로 TC/C 매칭
########################################
def associate_candidate_with_tc(filtered_rectangles, member_boxes, tc_c_entries, y_margin=900):
    associations = {}
    for idx, rect in enumerate(filtered_rectangles):
        rx_min = min(pt[0] for pt in rect)
        ry_min = min(pt[1] for pt in rect)
        rx_max = max(pt[0] for pt in rect)
        ry_max = max(pt[1] for pt in rect)
        
        matched_texts = set()
        for mb in member_boxes:
            bx_min, by_min, bx_max, by_max = mb["bbox"]
            if bx_min >= rx_min and bx_max <= rx_max and by_min >= ry_min and by_max <= ry_max:
                baseline = by_max
                for entry in tc_c_entries:
                    tc_y_min = entry["bbox"][1]
                    if baseline < tc_y_min <= baseline + y_margin:
                        matched_texts.add(entry["text"])
        associations[idx] = list(matched_texts)
        print(f"Candidate cell {idx+1}: matched TC/C texts => {associations[idx]}")
    return associations

########################################
# [H] 최종 결과 저장: 후보 셀 크롭 이미지 저장 (높이는 gap만큼; 파일명에 매칭된 TC/C 반영)
########################################
def save_images_with_tc_c(output_base_folder, json_filename, filtered_rectangles, associations, original_image, gap):
    output_folder = os.path.join(output_base_folder, os.path.splitext(json_filename)[0])
    os.makedirs(output_folder, exist_ok=True)
    for idx, rect in enumerate(filtered_rectangles):
        rx_min = int(min(pt[0] for pt in rect))
        ry_min = int(min(pt[1] for pt in rect))
        rx_max = int(max(pt[0] for pt in rect))
        # 높이는 후보 셀의 상단(y_min)부터 gap만큼 사용
        ry_max = ry_min + int(gap)
        
        cropped_image = original_image[ry_min:ry_max, rx_min:rx_max]
        
        matched = associations.get(idx, [])
        if matched:
            text_suffix = "_".join([t.strip() for t in matched])
            output_filename = f"filtered_rectangle_{idx+1}_{text_suffix}.png"
            print(f"Candidate {idx+1}: File will be saved as {output_filename}")
        else:
            output_filename = f"filtered_rectangle_{idx+1}.png"
        
        out_path = os.path.join(output_folder, output_filename)
        cv2.imwrite(out_path, cropped_image)
        print(f"{json_filename} - Saved final cell image: {out_path}")

########################################
# [I] Candidate Cell 이미지 저장 (오버레이 + 파일명 반영)
########################################
def save_candidate_cells_with_associations(output_base_folder, json_filename, filtered_rectangles, original_image, associations, gap):
    candidate_folder = os.path.join(output_base_folder, os.path.splitext(json_filename)[0], "CandidateCells")
    os.makedirs(candidate_folder, exist_ok=True)
    
    for idx, rect in enumerate(filtered_rectangles):
        rx_min = int(min(pt[0] for pt in rect))
        ry_min = int(min(pt[1] for pt in rect))
        rx_max = int(max(pt[0] for pt in rect))
        ry_max = ry_min + int(gap)
        
        cropped_image = original_image[ry_min:ry_max, rx_min:rx_max].copy()
        
        matched = associations.get(idx, [])
        if matched:
            text_str = ", ".join([t.strip() for t in matched])
            cv2.putText(cropped_image, text_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0,0,255), 2, cv2.LINE_AA)
            candidate_filename = f"candidate_cell_{idx+1}_{text_str}.png"
            print(f"Candidate {idx+1}: File will be saved as {candidate_filename}")
        else:
            candidate_filename = f"candidate_cell_{idx+1}.png"
        
        candidate_path = os.path.join(candidate_folder, candidate_filename)
        cv2.imwrite(candidate_path, cropped_image)
        print(f"Candidate cell {idx+1} saved: {candidate_path}")

########################################
# [J] TC/C 텍스트 오버레이 (전체 이미지에 녹색 테두리와 라벨)
########################################
def annotate_tc_c_entries(image, tc_c_entries):
    annotated = image.copy()
    for entry in tc_c_entries:
        x_min, y_min, x_max, y_max = map(int, entry["bbox"])
        cv2.rectangle(annotated, (x_min, y_min), (x_max, y_max), (0,255,0), 2)
        cv2.putText(annotated, entry["text"], (x_min, y_min-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv2.LINE_AA)
    return annotated

########################################
# [K] 모든 후보 셀 이미지를 한 폴더에 저장 (페이지 번호 포함)
########################################
def save_images_with_tc_c_all_in_one_folder(output_base_folder, page_idx, filtered_rectangles, associations, original_image, gap):
    all_filtered_folder = os.path.join(output_base_folder, "All_filtered_rectangles")
    os.makedirs(all_filtered_folder, exist_ok=True)
    
    for cell_idx, rect in enumerate(filtered_rectangles, start=1):
        rx_min = int(min(pt[0] for pt in rect))
        ry_min = int(min(pt[1] for pt in rect))
        rx_max = int(max(pt[0] for pt in rect))
        ry_max = ry_min + int(gap)
        
        cropped_image = original_image[ry_min:ry_max, rx_min:rx_max]
        
        matched = associations.get(cell_idx - 1, [])
        filename = f"Page_{page_idx}_Rectangle_{cell_idx}"
        if matched:
            text_suffix = "_".join([t.strip() for t in matched])
            filename += f"_{text_suffix}"
        filename += ".png"
        
        out_path = os.path.join(all_filtered_folder, filename)
        cv2.imwrite(out_path, cropped_image)
        print(f"[AllInOne] Saved: {out_path}")

########################################
# [L] Main 실행: 모든 JSON 파일 처리
########################################
def main():
    json_folder = r"D:\LLLast\SD\surya\raw_split"       # JSON 파일 폴더
    image_folder = r"D:\LLLast\SD\raw_split\table_extraction"         # 이미지 폴더
    output_base_folder = r"D:\LLLast\SD\MemberCode_extraction_All"
    os.makedirs(output_base_folder, exist_ok=True)
    
    # [NEW] 전역 x범위 수집
    from_code_global_min_x, from_code_global_max_x = gather_global_keyword_xrange(json_folder, keywords=["부호", "구분"])
    print(f"[Global Keyword XRange] min_x={from_code_global_min_x}, max_x={from_code_global_max_x}")
    
    global_start_y, gap = compute_global_reference_and_gap_advanced(
        json_folder, keywords=["부호", "구분"],
        tolerance_ratio=0.02, baseline_height=1000, same_line_threshold=30
    )
    if global_start_y is None or gap is None:
        print("Cannot compute global reference values.")
        return
    print(f"[Global Reference] start y: {global_start_y:.2f}, gap: {gap:.2f}")
    
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png','.jpg','.jpeg'))]
    if not image_files:
        print("No image files found.")
        return
    
    for page_idx, img_file in enumerate(image_files, start=1):
        base_name = os.path.splitext(img_file)[0]
        json_file = os.path.join(json_folder, base_name + ".json")
        if not os.path.exists(json_file):
            print(f"[{base_name}] JSON file not found.")
            continue
        
        image_path = os.path.join(image_folder, img_file)
        image = cv2.imread(image_path)
        if image is None:
            print(f"[{base_name}] Could not load image: {image_path}")
            continue
        
        print(f"\n=== Processing: {base_name} (Page {page_idx}) ===")
        
        # 1) 텍스트 박스 추출
        boxes = extract_all_text_boxes(json_file)
        
        # [NEW] 전역 X범위로 필터링 (부호/구분 박스와 동일 x 범위 제거)
        boxes = filter_boxes_by_global_xrange(boxes, from_code_global_min_x, from_code_global_max_x)
        
        # 2) 수직 점 필터링
        vertical_points = generate_vertical_points_from_reference(global_start_y, gap, image.shape[0])
        filtered_boxes = [box for box in boxes if any(abs(box["center"][1]-vp) <= 50 for vp in vertical_points)]
        
        # 3) 멤버 코드 박스 필터링
        member_boxes = filter_member_code_boxes(filtered_boxes)
        print(f"  => Member code boxes count: {len(member_boxes)}")
        
        member_json_out = os.path.join(output_base_folder, base_name + '_member_boxes.json')
        with open(member_json_out, 'w', encoding='utf-8') as f:
            json.dump(member_boxes, f, ensure_ascii=False, indent=4)
        
        # 4) 이미지 이진화 + 교차점 추출 (후보 셀)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
        vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
        intersections = cv2.bitwise_and(horizontal_lines, vertical_lines)
        contours, _ = cv2.findContours(intersections, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        intersection_points = [(x + w // 2, y + h // 2)
                               for x, y, w, h in [cv2.boundingRect(cnt) for cnt in contours]]
        
        valid_rects = find_valid_rectangles(intersection_points, horizontal_lines, vertical_lines)
        print(f"  => Candidate rectangles: {len(valid_rects)}")
        
        max_cell_height = 200
        filtered_rectangles = filter_rectangles_by_member_boxes(valid_rects, vertical_points, member_boxes, max_cell_height, tolerance=10)
        print(f"  => Filtered candidate cells: {len(filtered_rectangles)}")
        
        # 5) TC/C 텍스트 추출 + 매칭
        tc_c_entries = extract_tc_c_entries(json_file)
        candidate_associations = associate_candidate_with_tc(filtered_rectangles, member_boxes, tc_c_entries, y_margin=900)
        
        # 6) 디버깅: 전체 이미지에 후보 셀(빨간 사각형) 및 매칭 텍스트 오버레이
        annotated_candidate = image.copy()
        for idx, rect in enumerate(filtered_rectangles):
            rx_min = int(min(pt[0] for pt in rect))
            ry_min = int(min(pt[1] for pt in rect))
            rx_max = int(max(pt[0] for pt in rect))
            ry_max = int(max(pt[1] for pt in rect))
            cv2.rectangle(annotated_candidate, (rx_min, ry_min), (rx_max, ry_max), (0,0,255), 3)
            if idx in candidate_associations and candidate_associations[idx]:
                text_str = ", ".join(candidate_associations[idx])
                cv2.putText(annotated_candidate, text_str, (rx_min, ry_min-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2, cv2.LINE_AA)
        out_candidate = os.path.join(output_base_folder, base_name + "_candidate_cells.png")
        cv2.imwrite(out_candidate, annotated_candidate)
        print(f"  => Annotated candidate image saved: {out_candidate}")
        
        # 7) TC/C 텍스트 오버레이 (녹색) - 전체 이미지
        annotated_tc = annotate_tc_c_entries(image, tc_c_entries)
        out_tc = os.path.join(output_base_folder, base_name + "_tc_overlay.png")
        cv2.imwrite(out_tc, annotated_tc)
        print(f"  => TC/C overlay saved: {out_tc}")
        
        # 8) 후보 셀 크롭 이미지 저장 (높이는 gap만큼; 파일명에 매칭 TC/C 반영)
        save_candidate_cells_with_associations(output_base_folder, base_name + ".json",
                                               filtered_rectangles, image, candidate_associations, gap)
        
        # 9) 최종 결과: 각 후보 셀 크롭 이미지 저장 (높이는 gap만큼; 파일명에 매칭 TC/C 반영)
        save_images_with_tc_c(output_base_folder, base_name + ".json",
                              filtered_rectangles, candidate_associations, image, gap)
        
        # 10) (디버깅) 수직 점 및 멤버 박스 오버레이 이미지 저장
        annotated_members = annotate_tc_c_entries(image, tc_c_entries)
        cx = image.shape[1] // 2
        for vp in generate_vertical_points_from_reference(global_start_y, gap, image.shape[0]):
            cv2.circle(annotated_members, (cx, int(vp)), 5, (255,0,0), -1)
            cv2.putText(annotated_members, str(int(vp)), (cx+5, int(vp)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1, cv2.LINE_AA)
        out_members = os.path.join(output_base_folder, base_name + "_vertical_textboxes.png")
        cv2.imwrite(out_members, annotated_members)
        print(f"  => Member boxes + vertical lines overlay saved: {out_members}")
        
        # 11) 모든 페이지의 후보 셀 이미지를 한 폴더에 저장 (파일명에 페이지 번호 포함)
        save_images_with_tc_c_all_in_one_folder(output_base_folder, page_idx, filtered_rectangles, candidate_associations, image, gap)
    
    print("\nAll files processed.")

if __name__ == "__main__":
    main()

    
    
    
    
    
    
    
    
    
    
########################################################################################################################################
#원하는 영역 뽑아냈으니 다시 ocr 

import subprocess
from tqdm import tqdm
import re

# surya_ocr 명령어 및 인자 설정
command = "surya_ocr"
argument = r"D:\LLLast\SD\MemberCode_extraction_All\All_filtered_rectangles"

# surya_ocr 명령어 실행 (진행 정보가 stdout에 출력되어야 함)
process = subprocess.Popen([command, argument],
                           stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT,
                           text=True,
                           shell=True)

# 예를 들어, surya_ocr가 "Progress: 45%" 형식의 문자열을 출력한다고 가정하면,
# tqdm 진행률 표시를 위해 초기 total은 100(%)로 설정할 수 있습니다.
pbar = tqdm(total=100)

# 출력 라인을 실시간으로 읽으며 진행률 업데이트
while True:
    line = process.stdout.readline()
    if not line:
        break
    print(line, end="")  # 터미널에 출력 (원하는 경우)
    # "Progress: XX%"를 찾아서 진행률 업데이트
    match = re.search(r"Progress:\s*(\d+)%", line)
    if match:
        progress = int(match.group(1))
        pbar.n = progress
        pbar.refresh()

process.wait()
pbar.close()
#####################################################################################################
########################################################################################################
#다시 파일별로 ocr 나누기

import json
import os
import re

########################################
# [1] 상부근/하부근 정보 검사 함수 (스터럽 제외)
########################################
def has_rebar_info(data):
    """
    data(파이썬 객체 - dict, list, str 등)에 대해 재귀적으로 탐색하여,
    (숫자)-(UHD|SUHD|HD|D|SHD)(숫자) 패턴이 하나라도 있으면 True를 반환한다.
    예: "13-UHD25", "5-HD10", "12-SHD19"
    """
    rebar_pattern = re.compile(r'^\d+-?(?:UHD|SUHD|HD|D|SHD)\d+$', re.IGNORECASE)
    
    if isinstance(data, dict):
        for value in data.values():
            if has_rebar_info(value):
                return True
    elif isinstance(data, list):
        for item in data:
            if has_rebar_info(item):
                return True
    elif isinstance(data, str):
        normalized = re.sub(r"\s+", "", data)
        if rebar_pattern.match(normalized):
            return True
    return False

########################################
# [2] JSON 파일 분리 및 대응 이미지 삭제
########################################
def split_surya_ocr_result(combined_json_path, output_folder, image_folder):
    """
    통합 JSON 파일에서 각 항목을 분리하는데,
    각 항목 내에 (숫자)-(UHD|SUHD|HD|D|SHD)(숫자) 패턴의 정보가 없다면
    해당 항목은 별도의 JSON 파일로 저장하지 않고, 대응되는 이미지도 삭제한다.
    """
    with open(combined_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    os.makedirs(output_folder, exist_ok=True)

    # data가 dict 형태라고 가정 (각 key가 파일명 또는 식별자)
    for key, value in data.items():
        if not has_rebar_info(value):
            print(f"Skipping '{key}' because no rebar info was found.")
            # 대응되는 이미지 삭제 (예: key.png)
            for ext in [".png", ".jpg", ".jpeg"]:
                image_path = os.path.join(image_folder, f"{key}{ext}")
                if os.path.exists(image_path):
                    os.remove(image_path)
                    print(f"Deleted image: {image_path}")
            continue

        output_path = os.path.join(output_folder, f"{key}.json")
        with open(output_path, "w", encoding="utf-8") as out_f:
            json.dump(value, out_f, ensure_ascii=False, indent=4)
        print(f"Saved: {output_path}")








########################################
# 메인 실행
########################################
if __name__ == "__main__":
    combined_json_path = r"D:\LLLast\SD\MemberCode_extraction_All\All_filtered_rectangles\results.json"
    output_folder = r"D:\LLLast\SD\Filtered_information\text"
    image_folder = r"D:\LLLast\SD\MemberCode_extraction_All\All_filtered_rectangles"  # 대응 이미지가 있는 폴더
    split_surya_ocr_result(combined_json_path, output_folder, image_folder)


##################################################################################################
#이제 상부 하부 등 요소 추출 할거임.
# 이전 코드 이용해서 텍스트박스 계층 정의, 이미지 추출





import cv2
import json
import os
import numpy as np
import re
import csv

########################################
# 기본 유틸리티 함수
########################################
def get_center(bbox):
    """bbox: [x_min, y_min, x_max, y_max] → (center_x, center_y)"""
    return ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)

def is_valid_reinforce_text(text):
    """
    텍스트에서 공백 제거 후, 오직 숫자, 하이픈, 그리고 'U', 'H', 'D' (대소문자 무시)로만 구성되었는지 검사합니다.
    예: "13-UHD25", "5-HD10" → True; "X: 3-UHD15" → False.
    """
    s = re.sub(r"\s+", "", text)
    pattern = r"^\d+\-[UHDuhd]+\d+$"
    return bool(re.fullmatch(pattern, s))

########################################
# OCR 텍스트 박스 계층 구조 관련 함수
########################################
def classify_direction_group(group):
    """
    주어진 방향 그룹(텍스트 박스 리스트)을 아래 조건에 따라 분류합니다.
      1. 그룹 내에서 제일 아래의 텍스트 박스 → "stirrup"
      2. 그 바로 위에서 (스터럽 바로 위) 텍스트에 "UHD", "HD", "D"가 포함되고,
         텍스트가 오직 숫자, 하이픈, U/H/D로만 구성된 경우 → "lower"
      3. 그 위에서 동일 조건을 만족하는 텍스트 박스 → "upper"
      4. 나머지 텍스트 박스 중 세로(높이 > 너비)이며 숫자와 쉼표만 구성 → "height"
      5. 나머지 텍스트 박스 중 가로(너비 ≥ 높이)이며 숫자와 쉼표만 구성 → "width"
    해당 텍스트 박스가 없으면 None으로 설정합니다.
    """
    classification = {"stirrup": None, "lower": None, "upper": None, "height": None, "width": None}
    if not group:
        return classification

    # y_max 기준 오름차순(화면 아래로 갈수록 bbox[3]가 커짐)
    sorted_group = sorted(group, key=lambda tb: tb["bbox"][3])
    classification["stirrup"] = sorted_group[-1]
    remaining = sorted_group[:-1]

    reinforce_keywords = ["UHD", "HD", "D"]
    lower = None
    for tb in reversed(remaining):
        if any(k.lower() in tb["text"].lower() for k in reinforce_keywords) and is_valid_reinforce_text(tb["text"]):
            lower = tb
            break
    classification["lower"] = lower

    upper = None
    if lower is not None:
        lower_index = remaining.index(lower)
        for tb in remaining[:lower_index]:
            if any(k.lower() in tb["text"].lower() for k in reinforce_keywords) and is_valid_reinforce_text(tb["text"]):
                upper = tb
                break
    classification["upper"] = upper

    height_box = None
    width_box = None
    for tb in remaining:
        if tb is lower or tb is upper:
            continue
        if re.fullmatch(r"[0-9,]+", tb["text"].strip()):
            x, y, x2, y2 = tb["bbox"]
            box_w = x2 - x
            box_h = y2 - y
            if box_h > box_w and height_box is None:
                height_box = tb
            elif box_w >= box_h and width_box is None:
                width_box = tb
    classification["height"] = height_box
    classification["width"] = width_box

    return classification

########################################
# OCR JSON 파일에서 계층 구조 추출 함수
########################################
def extract_hierarchical_structure(json_file):
    """
    OCR 결과 JSON 파일에서 텍스트 박스를 읽어 계층적 구조를 생성합니다.
      - 가장 상단의 텍스트 박스를 MEMBER CODE로 지정합니다.
      - 나머지 텍스트 박스 중 "EXT", "INT", "CENTER", "ALL", "BOTH"를 포함하는 박스를 Direction 박스로 분류합니다.
      - 각 Direction 그룹에 대해 classify_direction_group을 적용합니다.
    반환:
       {"member_code": {...}, "directions": [ { "direction_box": {...} 또는 None, "assigned": [...], "classification": {...} }, ... ]}
    """
    # JSON 파일 로드
    try:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except:
        return None

    text_boxes = []
    # data가 list인지 dict인지에 따라 반복 처리
    if isinstance(data, list):
        iterable = data
    elif isinstance(data, dict):
        iterable = [data]
    else:
        iterable = []
    
    for item in iterable:
        if isinstance(item, dict) and "text_lines" in item:
            for line in item["text_lines"]:
                if "bbox" in line and "text" in line:
                    try:
                        bbox = list(map(int, line["bbox"]))
                    except:
                        continue
                    text = line["text"]
                    center = get_center(bbox)
                    text_boxes.append({"text": text, "bbox": bbox, "center": center})

    if not text_boxes:
        return None

    # y_min 기준 정렬(가장 위 → Member code로 가정)
    text_boxes = sorted(text_boxes, key=lambda tb: tb["bbox"][1])
    member_code = text_boxes[0]
    remaining = text_boxes[1:]
    
    keywords = ["EXT", "INT", "CENTER", "ALL", "BOTH"]
    direction_boxes = []
    others = []
    for tb in remaining:
        if any(k.lower() in tb["text"].lower() for k in keywords):
            direction_boxes.append(tb)
        else:
            others.append(tb)
    
    hierarchical_data = {"member_code": member_code, "directions": []}
    if direction_boxes:
        # 방향 박스가 있으면, 각 방향 박스와 가장 가까운 텍스트 박스들을 그룹화
        assignments = {i: [] for i in range(len(direction_boxes))}
        for ob in others:
            distances = [
                np.linalg.norm(np.array(ob["center"]) - np.array(db["center"]))
                for db in direction_boxes
            ]
            nearest_index = int(np.argmin(distances))
            assignments[nearest_index].append(ob)

        for i, dbox in enumerate(direction_boxes):
            group = [dbox] + assignments[i]
            classification = classify_direction_group(group)
            hierarchical_data["directions"].append({
                "direction_box": dbox,
                "assigned": assignments[i],
                "classification": classification
            })
    else:
        # 방향 박스가 없으면, 나머지 전부를 하나의 그룹으로 간주
        classification = classify_direction_group(others)
        hierarchical_data["directions"].append({
            "direction_box": None,
            "assigned": others,
            "classification": classification
        })
    return hierarchical_data

########################################
# 도면 영역 추출 함수
########################################
def extract_drawing_areas(image, max_width_ratio=0.8, max_height_ratio=0.8,
                           center_y_ratio=0.4, min_width=10, min_height=10):
    """
    이 예시는 윤곽선 기반으로 '사각형' 추정하는 로직. 필요에 따라 수정 가능
    """
    height, width = image.shape[:2]
    max_width = int(width * max_width_ratio)
    max_height = int(height * max_height_ratio)
    center_y_min = int(height * (0.5 - center_y_ratio / 2))
    center_y_max = int(height * (0.5 + center_y_ratio / 2))
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    areas = []
    for cnt in contours:
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if len(approx) in [4, 5] and cv2.isContourConvex(approx):
            x, y, w, h = cv2.boundingRect(approx)
            if (min_width <= w <= max_width) and (min_height <= h <= max_height):
                rect_center_y = y + h // 2
                if center_y_min <= rect_center_y <= center_y_max:
                    roi = thresh[y:y+h, x:x+w]
                    white_pixels_ratio = cv2.countNonZero(roi) / (w * h)
                    # 임의의 조건(흰 픽셀 비율 > 5%)
                    if white_pixels_ratio > 0.05:
                        areas.append((x, y, w, h))

    # 내부에 포함된 사각형 제거
    final_areas = []
    for rect1 in areas:
        x1, y1, w1, h1 = rect1
        contained = False
        for rect2 in areas:
            if rect1 == rect2:
                continue
            x2, y2, w2, h2 = rect2
            if x1 >= x2 and y1 >= y2 and (x1+w1) <= (x2+w2) and (y1+h1) <= (y2+h2):
                contained = True
                break
        if not contained:
            final_areas.append(rect1)
    return final_areas

########################################
# 도면 영역 할당 함수
########################################
def assign_drawing_areas_to_directions(drawing_areas, hierarchical_data):
    assignments = []
    direction_centers = []
    for d in hierarchical_data["directions"]:
        dbox = d["direction_box"]
        if dbox is not None:
            direction_centers.append(get_center(dbox["bbox"]))
        else:
            direction_centers.append(None)
    for (x, y, w, h) in drawing_areas:
        cx = x + w / 2
        cy = y + h / 2
        assigned_index = -1
        min_dist = float('inf')
        for idx, center in enumerate(direction_centers):
            if center is not None:
                dist = np.linalg.norm(np.array([cx, cy]) - np.array(center))
                if dist < min_dist:
                    min_dist = dist
                    assigned_index = idx
        assignments.append((x, y, w, h, assigned_index))
    return assignments

########################################
# 건너뛰기 조건 예시
########################################
def should_skip_due_to_no_size(hierarchical_data, image):
    """
    예시 로직: 도면 영역이 하나도 없고, 모든 Direction에 width/height가 없다면 스킵
    """
    drawing_areas = extract_drawing_areas(image, 0.8, 0.8, 0.4, 10, 10)
    if len(drawing_areas) == 0:
        for d in hierarchical_data.get("directions", []):
            classification = d.get("classification", {})
            if classification.get("width") is not None or classification.get("height") is not None:
                return False
        return True
    return False

########################################
# (옵션) 시각화 예시
########################################
def annotate_by_groups(json_file, image):
    """
    direction, stirrup, lower, upper, width, height 등을 색깔별로 표시
    """
    hierarchical_data = extract_hierarchical_structure(json_file)
    if not hierarchical_data:
        return None

    for d in hierarchical_data["directions"]:
        # 필요 시 bbox 그리기
        pass

    # 여기서는 간단히 image 원본 그대로 리턴
    return image

########################################
# [수정된] extract_prefix_from_filename
#   - 여러 개의 접두어를 한 번에 추출
########################################
def extract_prefix_from_filename(base_name):
    """
    예) "Page_11_Rectangle_1_C1, C2, C3"
       → ["C1", "C2", "C3"]
    """
    matches = re.findall(r"(TC\d+[A-Za-z]*|T\d+[A-Za-z]*|C\d+[A-Za-z]*)", base_name, re.IGNORECASE)
    return [match.strip() for match in matches]

########################################
# [수정된] Member code 처리 함수
#   - 여러 prefix를 한 줄로 합쳐 "C1, C2, C3"로 만들고,
#     member_text가 "B3F~PITF"이면 "C1, C2, C3 B3F~PITF" 형태로
#     최종 1개의 문자열만 반환
########################################
def process_member_code_with_prefix(member_text, prefix_list):
    # member_text가 쉼표(,)를 포함할 수 있으므로 분리
    splitted_codes = [x.strip() for x in member_text.split(',')]

    # prefix_list를 하나의 문자열로 합침
    if prefix_list:
        combined_prefix = ", ".join(prefix_list)  # ["C1","C2","C3"] → "C1, C2, C3"
    else:
        combined_prefix = ""

    merged_results = []
    for code in splitted_codes:
        if combined_prefix:
            merged_results.append(f"{combined_prefix} {code}")
        else:
            merged_results.append(code)

    # splitted_codes가 여러 개라도, 최종적으로 하나의 문자열로 합침
    # 예) ["C1, C2, C3 B3F~PITF", "C1, C2, C3 기타"] → "C1, C2, C3 B3F~PITF, C1, C2, C3 기타"
    single_line = ", ".join(merged_results)

    # CSV 작성 시 for문에 돌리기 위해 리스트에 담되, 요소는 1개만
    return [single_line]

########################################
# CSV 저장 및 전체 파일 처리 (텍스트 + 도면 연동)
########################################
def process_all_files_and_save_csv(json_folder, image_folder, output_folder, csv_output):
    rows = []
    header = ["Page", "Member code", "Direction", "Top_rebar", "Bot_rebar", "Stirrups", "Width", "Height"]
    rows.append(header)
    
    json_files = [f for f in os.listdir(json_folder) if f.lower().endswith('.json')]
    for json_file in json_files:
        base_name = os.path.splitext(json_file)[0]
        json_path = os.path.join(json_folder, json_file)
        hierarchical_data = extract_hierarchical_structure(json_path)
        if hierarchical_data is None:
            print(f"계층적 데이터 추출 실패: {json_path}")
            continue
        
        # 이미지 로드
        image_candidate = None
        for ext in ['.png', '.jpg', '.jpeg']:
            candidate = os.path.join(image_folder, base_name + ext)
            if os.path.exists(candidate):
                image_candidate = candidate
                break
        if image_candidate is None:
            print(f"이미지 파일을 찾을 수 없습니다: {base_name}")
            continue
        image = cv2.imread(image_candidate)
        if image is None:
            print(f"이미지 로드 실패: {base_name}")
            continue

        # skip 조건
        if should_skip_due_to_no_size(hierarchical_data, image):
            print(f"{json_file}은(는) 도면 영역 및 Width, Height 정보가 없어 건너뜁니다.")
            continue
        
        page_num = extract_page_number_from_filename(base_name)
        member_text = hierarchical_data["member_code"]["text"]
        prefix_list = extract_prefix_from_filename(base_name)
        # 여기서 한 줄만 나옴 (여러 prefix가 있어도 1개의 문자열)
        modified_member_codes = process_member_code_with_prefix(member_text, prefix_list)
        
        if hierarchical_data["directions"]:
            for direction in hierarchical_data["directions"]:
                if direction["direction_box"]:
                    direction_text = direction["direction_box"]["text"]
                else:
                    direction_text = ""
                classification = direction["classification"]
                top_rebar = classification["upper"]["text"] if classification["upper"] else ""
                bot_rebar = classification["lower"]["text"] if classification["lower"] else ""
                stirrups = classification["stirrup"]["text"] if classification["stirrup"] else ""
                width_val = classification["width"]["text"] if classification["width"] else ""
                height_val = classification["height"]["text"] if classification["height"] else ""
                
                # modified_member_codes → 항상 1개 요소만 있음
                for mmc in modified_member_codes:
                    row = [
                        page_num,
                        mmc,
                        direction_text,
                        top_rebar,
                        bot_rebar,
                        stirrups,
                        width_val,
                        height_val
                    ]
                    rows.append(row)
        else:
            # 방향 박스가 없는 경우
            for mmc in modified_member_codes:
                row = [page_num, mmc, "", "", "", "", "", ""]
                rows.append(row)
    
    # 페이지 번호 순 정렬 (옵션)
    header = rows[0]
    data_rows = rows[1:]
    def sort_key(row):
        try:
            return int(row[0])
        except:
            return float('inf')
    data_rows_sorted = sorted(data_rows, key=sort_key)
    
    with open(csv_output, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data_rows_sorted)
    print(f"CSV 파일이 저장되었습니다: {csv_output}")
    
########################################
# 파일명 및 CSV 관련 함수 (텍스트 관련)
########################################
def extract_page_number_from_filename(filename):
    match = re.search(r"page[_\-]?(\d+)", filename, re.IGNORECASE)
    return match.group(1) if match else ""
########################################
# (옵션) 모든 파일에 대해 시각화 이미지 저장
########################################
def process_all_files(json_folder, image_folder, output_folder):
    json_files = [f for f in os.listdir(json_folder) if f.lower().endswith('.json')]
    for json_file in json_files:
        base_name = os.path.splitext(json_file)[0]
        json_path = os.path.join(json_folder, json_file)
        image_path = None
        for ext in ['.png', '.jpg', '.jpeg']:
            candidate = os.path.join(image_folder, base_name + ext)
            if os.path.exists(candidate):
                image_path = candidate
                break
        if image_path is None:
            print(f"이미지 파일을 찾을 수 없습니다: {base_name}")
            continue

        hierarchical_data = extract_hierarchical_structure(json_path)
        if hierarchical_data is None:
            print(f"계층적 데이터 추출 실패: {json_path}")
            continue

        # 예시로 시각화만
        output_image = annotate_by_groups(json_path, cv2.imread(image_path))
        if output_image is not None:
            rec_suffix = "_rec"
            output_image_path = os.path.join(output_folder, base_name + rec_suffix + ".png")
            cv2.imwrite(output_image_path, output_image)
            print(f"{output_image_path}에 시각화 결과 저장됨.")
        else:
            print(f"시각화 실패: {image_path}")

########################################
# (옵션) 도면 영역만 별도로 추출하여 저장
########################################
def save_drawing_area_images(image_path, hierarchical_data, output_folder):
    image = cv2.imread(image_path)
    if image is None:
        print("이미지를 불러올 수 없습니다:", image_path)
        return

    drawing_areas = extract_drawing_areas(image, 0.8, 0.8, 0.4, 10, 10)
    drawing_areas_assigned = assign_drawing_areas_to_directions(drawing_areas, hierarchical_data)
    member_code = hierarchical_data["member_code"]["text"].strip() if hierarchical_data.get("member_code") else ""
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    for i, (x, y, w, h, assigned_index) in enumerate(drawing_areas_assigned):
        crop_img = image[y:y+h, x:x+w]
        direction_text = ""
        if assigned_index >= 0 and hierarchical_data["directions"][assigned_index]["direction_box"]:
            direction_text = hierarchical_data["directions"][assigned_index]["direction_box"]["text"].strip()
            if direction_text.lower() == "none":
                direction_text = ""

        if member_code and direction_text:
            out_filename = f"{base_name}_{member_code}_{direction_text}.png"
        elif member_code:
            out_filename = f"{base_name}_{member_code}.png"
        else:
            out_filename = f"{base_name}.png"
        
        out_path = os.path.join(output_folder, out_filename)
        success, encoded_image = cv2.imencode('.png', crop_img)
        if success:
            with open(out_path, 'wb') as f:
                f.write(encoded_image.tobytes())
            print(f"Drawing 영역 저장: {out_path}")
        else:
            print(f"이미지 인코딩 실패: {out_path}")

########################################
# 메인 실행 예시
########################################
if __name__ == "__main__":
    # 폴더 경로 예시(사용 환경에 맞게 수정)
    json_folder = r"D:\LLLast\SD\Filtered_information\text"
    image_folder = r"D:\LLLast\SD\MemberCode_extraction_All\All_filtered_rectangles"
    output_folder = r"D:\LLLast\SD\Eelement_hir"
    csv_output = r"D:\LLLast\SD\results_check.csv"
    drawing_output_folder = r"D:\LLLast\SD\rec"
    
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(drawing_output_folder, exist_ok=True)

    # 1) 모든 파일 처리 + CSV 생성
    process_all_files_and_save_csv(json_folder, image_folder, output_folder, csv_output)

    # 2) (옵션) 시각화 이미지 생성
    process_all_files(json_folder, image_folder, output_folder)

    # 3) (옵션) 도면 영역만 별도로 추출 저장
    json_files = [f for f in os.listdir(json_folder) if f.lower().endswith('.json')]
    for json_file in json_files:
        base_name = os.path.splitext(json_file)[0]
        json_path = os.path.join(json_folder, json_file)
        hierarchical_data = extract_hierarchical_structure(json_path)
        if hierarchical_data is None:
            continue
        image_path = None
        for ext in ['.png', '.jpg', '.jpeg']:
            candidate = os.path.join(image_folder, base_name + ext)
            if os.path.exists(candidate):
                image_path = candidate
                break
        if image_path:
            save_drawing_area_images(image_path, hierarchical_data, drawing_output_folder)


        
        
        
        
        
        
        
####################################################################################################################

#이제 철근 정보 뽑기


import cv2
import numpy as np
import os
import re
import json
import csv
import math

########################################
# [M] imread_unicode: Unicode 경로 지원
########################################
def imread_unicode(filename, flags=cv2.IMREAD_COLOR):
    try:
        with open(filename, 'rb') as f:
            file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, flags)
        if img is None:
            print(f"[DEBUG] cv2.imdecode returned None for file: {filename}")
        return img
    except Exception as e:
        print(f"[DEBUG] Exception reading {filename}: {e}")
        return None

########################################
# imwrite_unicode: 저장 단계에서 Unicode 문제 해결
########################################
def imwrite_unicode(filename, img):
    try:
        # 이미지를 PNG로 인코딩 (다른 포맷도 가능)
        result, encoded_img = cv2.imencode('.png', img)
        if result:
            with open(filename, 'wb') as f:
                f.write(encoded_img.tobytes())
            return True
        else:
            print(f"[DEBUG] cv2.imencode failed for file: {filename}")
            return False
    except Exception as e:
        print(f"[DEBUG] Exception writing {filename}: {e}")
        return False

########################################
# otsu_binarization: 이진화 및 저장 (imwrite_unicode 사용)
########################################
def otsu_binarization(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')
    
    processed_count = 0
    failed_files = []
    
    files = [f for f in os.listdir(input_folder) if f.lower().endswith(valid_extensions)]
    print(f"총 {len(files)}개의 파일을 찾았습니다.")
    
    for file_name in files:
        img_path = os.path.join(input_folder, file_name)
        img = imread_unicode(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"[DEBUG] 이미지를 읽지 못했습니다: {img_path}")
            failed_files.append(file_name)
            continue

        print(f"[DEBUG] {file_name} 읽음, shape: {img.shape}")
        
        ret, binary_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        unique_vals = np.unique(binary_img)
        print(f"[DEBUG] {file_name} 이진화 결과, 임계값: {ret}, 고유 픽셀 값: {unique_vals}")
        
        output_path = os.path.join(output_folder, file_name)
        if not imwrite_unicode(output_path, binary_img):
            print(f"[DEBUG] 이진화된 이미지를 저장하지 못했습니다: {output_path}")
            failed_files.append(file_name)
            continue
        
        print(f"이진화 완료: {output_path}")
        processed_count += 1

    print(f"\n총 {processed_count}개의 이미지를 성공적으로 처리했습니다.")
    if failed_files:
        print("다음 파일들은 최종 처리되지 않았습니다:")
        for f in failed_files:
            print(f"  - {f}")

if __name__ == "__main__":
    input_folder = r"D:\LLLast\SD\rec"       # 원본 이미지 폴더
    output_folder = r"D:\LLLast\SD\rec_binar"  # 결과 저장 폴더
    otsu_binarization(input_folder, output_folder)




#############################################################


###################################################################################################
#2-2) YOLOV5 적용

import pandas as pd
import os
import subprocess

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# YOLOv5 프로젝트 폴더로 작업 디렉토리 변경
os.chdir(r"C:\Users\S3PARC\yolov5")

# 학습 명령어 구성 (경로 구분자는 슬래시를 사용하거나 원시 문자열로 처리)
command = r'python detect.py --weights "D:\SD_final_summary\Img and Label for Trans learn\custom_rebar2\weights\best.pt" --source "D:\LLLast\SD\rec_binar" --img 640 --conf 0.5 --iou-thres 0.3 --save-txt --project "D:\LLLast\SD\rec_binar" --name exp'



# subprocess를 이용하여 명령어 실행 및 실시간 출력 캡처
process = subprocess.Popen(
    command,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    universal_newlines=True,
    encoding='utf-8',
    errors='replace'
)

# 실시간으로 출력 보기
while True:
    output = process.stdout.readline()
    if output == '' and process.poll() is not None:
        break
    if output:
        print(output.strip())

# 종료 코드 출력
return_code = process.poll()
print(f"프로세스 종료 코드: {return_code}")
###################################################################################################################################################
### 상 하부근 구분해서 정리


import os
import pandas as pd

# detect.py 결과 텍스트 파일들이 저장된 폴더 경로
results_dir = r"D:\LLLast\SD\rec_binar\exp\labels"

# 결과 데이터를 저장할 리스트
data = []

# 폴더 내의 모든 .txt 파일 처리
for txt_file in os.listdir(results_dir):
    if not txt_file.endswith('.txt'):
        continue

    file_path = os.path.join(results_dir, txt_file)
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 파일명에 특정 키워드(ALL, MID, BOTH, EXT., INT., CENTER, CEN)가 포함되면 RCBeam으로 인식 (대소문자 무시)
    is_beam = any(keyword in txt_file.lower() for keyword in ["all", "mid", "both", "ext.", "int.", "center", "cen"])
    
    # 초기 카운트 값
    upper_count = 0
    lower_count = 0

    if is_beam:
        # RCBeam: center_y 기준으로 상부와 하부를 나눔
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            try:
                center_y = float(parts[2])
            except ValueError:
                continue

            if center_y < 0.5:
                upper_count += 1
            else:
                lower_count += 1
    else:
        # COLUMN: 모든 객체를 하부로 처리
        total_count = 0
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            try:
                float(parts[2])
            except ValueError:
                continue
            total_count += 1
        lower_count = total_count

    # 0인 경우 빈 문자열로 처리
    upper_val = str(upper_count) if upper_count != 0 else ""
    lower_val = str(lower_count) if lower_count != 0 else ""

    data.append({
        "File": txt_file,
        "Top_Rebar_Img": upper_val,
        "Bot_Rebar_Img": lower_val
    })

# DataFrame 생성 후 CSV 저장 (UTF-8 BOM 포함)
df = pd.DataFrame(data)
output_csv = r"D:\LLLast\SD\rebar_counts.csv"
df.to_csv(output_csv, index=False, encoding="utf-8-sig")

print("CSV 파일이 저장되었습니다:", output_csv)

########################################################################


import pandas as pd
import re

# 확장자를 제거하는 헬퍼 함수 (선택사항)
def remove_extension(filename):
    return re.sub(r'\.txt$', '', filename, flags=re.IGNORECASE)

# 파일명에서 페이지, Member code, Direction 정보를 추출하는 함수
def extract_page_member_direction(filename):
    """
    두 가지 파일명 형식을 처리합니다.
    
    형식1 (Direction 있는 경우):
      예) "Page_3_Rectangle_2_-3~-1 B1A_CENTER.txt"
      → page: "3", member: "-3~-1 B1A", direction: "CENTER"
      
    형식2 (Direction 없는 경우):
      예) "Page_10_Rectangle_1_TC11_B3F~2F.txt"
      → page: "10", member: "TC11 B3F~2F", direction: ""
      
    known_directions: CENTER, EXT, INT, ALL, BOTH (대소문자 무시)
    """
    # 파일 확장자는 옵션으로 인식 (있어도 없다고 취급)
    pattern = re.compile(
        r'page[_\-]?(\d+)[_\-]rectangle[_\-]?\d+[_\-]?([^_]+)(?:[_\-](.+))?(?:\.txt)?$', 
        re.IGNORECASE
    )
    match = pattern.search(filename)
    if not match:
        return None, None, None
    page = match.group(1).strip()
    member1 = match.group(2).strip()
    group3 = match.group(3).strip() if match.group(3) else ""
    
    known_directions = {"CENTER", "EXT", "INT", "ALL", "BOTH", "EXT.", "INT."}
    # 만약 group3가 known_directions에 포함되면, group3는 Direction으로 판단
    if group3.upper() in known_directions:
        member = member1
        direction = group3
    else:
        # group3가 Direction이 아니라면, 두 member code를 합친다.
        member = f"{member1} {group3}" if group3 else member1
        direction = ""
    return page, member, direction

# rebar count 파일에서 키 생성: 공백 제거 후 "page_member(direction)" 형태로 생성
def create_rebar_key(filename):
    # 확장자 유무와 상관없이 처리하기 위해 remove_extension 사용 (옵션)
    filename_no_ext = remove_extension(filename)
    page, member, direction = extract_page_member_direction(filename_no_ext)
    if page is None or member is None:
        return None
    combined = f"{member} {direction}" if direction else member
    # 모든 공백 제거하여 비교에 영향을 주지 않도록 함
    combined_no_space = re.sub(r'\s+', '', combined)
    return f"{page}_{combined_no_space}"

# results 파일의 한 행에서 키 생성 (Page, Member code, Direction 열 사용)
def create_results_key(row):
    page = str(row['Page']).strip()
    member = str(row['Member code']).strip()
    direction = str(row['Direction']).strip() if 'Direction' in row and pd.notnull(row['Direction']) else ""
    combined = f"{member} {direction}" if direction else member
    combined_no_space = re.sub(r'\s+', '', combined)
    return f"{page}_{combined_no_space}"

# ==============================
# 데이터 병합 로직
# ==============================

# 1. rebar_counts 파일 읽기
df_rebar = pd.read_csv(r'D:\LLLast\SD\rebar_counts.csv')
# key 생성 (확장자가 있든 없든 동일하게 인식)
df_rebar['key'] = df_rebar['File'].apply(lambda x: create_rebar_key(x))
#print(df_rebar[['File', 'key']].head(10))  # 디버깅용

# 2. results 파일 읽기 (결과 파일에는 Page, Member code, Direction 열이 있다고 가정)
df_results = pd.read_csv(r'D:\LLLast\SD\results_check.csv')
df_results['key'] = df_results.apply(create_results_key, axis=1)
#print(df_results[['Page', 'Member code', 'Direction', 'key']].head(10))  # 디버깅용

# 3. 만약 df_results에 이미 "Top_Rebar_Img"와 "Bot_Rebar_Img" 열이 있다면 제거
for col in ["Top_Rebar_Img", "Bot_Rebar_Img"]:
    if col in df_results.columns:
        df_results.drop(columns=[col], inplace=True)

# 4. 두 DataFrame 병합 (df_results 기준, df_rebar에서 필요한 열만 가져옴)
df_merged = pd.merge(df_results, df_rebar[['key', 'Top_Rebar_Img', 'Bot_Rebar_Img']], on='key', how='left')

# 5. 병합 후 key 열 제거
df_merged.drop(columns=['key'], inplace=True)

# 6. 결과 CSV 저장 (UTF-8 BOM 포함)
output_csv_merged = r'D:\LLLast\SD\merged_results.csv'
df_merged.to_csv(output_csv_merged, index=False, encoding="utf-8-sig")

print("병합된 CSV 파일이 저장되었습니다:", output_csv_merged)




#################################################################################################################

#표준화

import pandas as pd
import re

def split_member_codes(df):
    """
    쉼표로 구분된 여러 Member code를 개별 행으로 분리하는 함수.
    
    단, 형식이 
      (Member code), (Member code), (Member code) (층수)
    인 경우, 예를 들어 "C1, C2, C3 B3F~PITF"라면,
    각 Member code에 공통으로 "B3F~PITF"를 붙여서
      "C1 B3F~PITF", "C2 B3F~PITF", "C3 B3F~PITF"로 분리합니다.
      
    그 외에는 쉼표 기준으로 그대로 분리합니다.
    """
    new_rows = []
    for _, row in df.iterrows():
        raw = str(row['Member code']).strip()
        tokens = [token.strip() for token in raw.split(',')]
        # 만약 여러 토큰이 있고, 첫 토큰은 단일 단어(즉, 층수 정보가 없는) 반면 마지막 토큰은
        # 공백으로 분리했을 때 2개 이상의 단어(즉, 부가적인 층수 정보가 붙은)가 있다면,
        # 마지막 토큰의 첫 번째 단어는 자체 코드로, 그 뒤 단어들은 공통 층수(또는 부가 정보)로 취급합니다.
        if len(tokens) > 1:
            first_parts = tokens[0].split()
            last_parts = tokens[-1].split()
            if len(first_parts) == 1 and len(last_parts) >= 2:
                # last_parts[0]는 마지막 Member code의 일부로 이미 들어있고,
                # last_parts[1:]는 모든 토큰에 공통으로 붙일 부가 정보
                common_info = " ".join(last_parts[1:])
                # 수정: 마지막 토큰을 그 자체의 첫 단어로 대체하고, 모든 토큰에 공통 정보를 붙임
                tokens[-1] = last_parts[0]
                tokens = [f"{token} {common_info}" for token in tokens]
        # 각 토큰별로 새 행 생성
        for code in tokens:
            new_row = row.copy()
            new_row['Member code'] = code
            new_rows.append(new_row)
    return pd.DataFrame(new_rows)

def convert_floor_str(floor_str: str):
    """
    'B3F' → -3
    '3F'  → 3
    변환 실패 시 None 반환
    """
    floor_str = floor_str.rstrip('F')  # 뒤에 붙은 F 제거
    if floor_str.startswith('B'):
        try:
            return -int(floor_str[1:])
        except:
            return None
    else:
        try:
            return int(floor_str)
        except:
            return None

def expand_member_code_range(df):
    """
    Member code에 '~' 또는 '～'가 포함된 경우 범위를 확장합니다.
    또한, '~'가 없는 경우에도 "prefix floor" 형태(예: "TC10A B3F")를 인식해
    floor를 숫자로 변환 가능하면 해당 값을 이용하여 1행을 생성합니다.
    """
    new_rows = []
    # 전각 틸드 → 반각 틸드 치환 함수
    def replace_fullwidth_tilde(s: str):
        return re.sub(r'[～]', '~', s)
    
    # 패턴1: "prefix floor1~floor2"
    pattern_tilde_1 = r"^\s*([^\s]+)\s+([B]?\w+F)\s*~\s*([B]?\w+F)\s*$"
    # 패턴2: "start_floor~end_floor prefix"
    pattern_tilde_2 = r"^\s*([+-]?\d+)\s*~\s*([+-]?\d+)\s+(\S+)\s*$"
    # 패턴3: 틸드가 없는 경우: "prefix floor"
    pattern_no_tilde = r"^\s*([^\s]+)\s+([B]?\w+F)\s*$"
    
    for _, row in df.iterrows():
        code = str(row['Member code']).strip()
        code = replace_fullwidth_tilde(code)
        
        if '~' in code:
            match1 = re.match(pattern_tilde_1, code, re.IGNORECASE)
            if match1:
                prefix = match1.group(1)
                start_str = match1.group(2)
                end_str   = match1.group(3)
                start_floor = convert_floor_str(start_str)
                end_floor   = convert_floor_str(end_str)
                if start_floor is not None and end_floor is not None:
                    if start_floor <= end_floor:
                        floor_range = range(start_floor, end_floor+1)
                    else:
                        floor_range = range(start_floor, end_floor-1, -1)
                    for fl in floor_range:
                        if fl == 0:
                            continue
                        new_row = row.copy()
                        new_row['Member code'] = f"{fl} {prefix}"
                        new_rows.append(new_row)
                else:
                    if start_floor is not None:
                        new_row = row.copy()
                        new_row['Member code'] = f"{start_floor} {prefix}"
                        new_rows.append(new_row)
                    else:
                        tmp_floor = start_str.rstrip("F")
                        new_row = row.copy()
                        new_row['Member code'] = f"{tmp_floor} {prefix}"
                        new_rows.append(new_row)
                    if end_floor is not None:
                        new_row = row.copy()
                        new_row['Member code'] = f"{end_floor} {prefix}"
                        new_rows.append(new_row)
                    else:
                        tmp_floor = end_str.rstrip("F")
                        new_row = row.copy()
                        new_row['Member code'] = f"{tmp_floor} {prefix}"
                        new_rows.append(new_row)
            else:
                match2 = re.match(pattern_tilde_2, code)
                if match2:
                    start_floor = int(match2.group(1))
                    end_floor   = int(match2.group(2))
                    prefix      = match2.group(3)
                    if start_floor <= end_floor:
                        floor_range = range(start_floor, end_floor+1)
                    else:
                        floor_range = range(start_floor, end_floor-1, -1)
                    for fl in floor_range:
                        if fl == 0:
                            continue
                        new_row = row.copy()
                        new_row['Member code'] = f"{fl} {prefix}"
                        new_rows.append(new_row)
                else:
                    new_rows.append(row)
        else:
            match_no_tilde = re.match(pattern_no_tilde, code, re.IGNORECASE)
            if match_no_tilde:
                prefix = match_no_tilde.group(1)
                floor_str = match_no_tilde.group(2)
                floor_val = convert_floor_str(floor_str)
                if floor_val is not None:
                    if floor_val == 0:
                        continue
                    new_row = row.copy()
                    new_row['Member code'] = f"{floor_val} {prefix}"
                    new_rows.append(new_row)
                else:
                    tmp_floor = floor_str.rstrip("F")
                    new_row = row.copy()
                    new_row['Member code'] = f"{tmp_floor} {prefix}"
                    new_rows.append(new_row)
            else:
                new_rows.append(row)
    
    return pd.DataFrame(new_rows)

# 예시: CSV 파일 읽기, 분리 및 확장 후 저장
df = pd.read_csv(r'D:\LLLast\SD\merged_results.csv', encoding='utf-8-sig')
df_split = split_member_codes(df)
df_expanded = expand_member_code_range(df_split)
output_csv = r'D:\LLLast\SD\merged_results_CONVERT1.csv'
df_expanded.to_csv(output_csv, index=False, encoding='utf-8-sig')

print("분리 및 확장 완료:", output_csv)


######################################

import pandas as pd

# 이미 1번, 2번 전처리를 마친 DataFrame (Member code 처리 결과)
# 예를 들어, 이 파일은 이전 단계에서 result.csv 로 저장된 파일이라고 가정합니다.
df_expanded = pd.read_csv(r'D:\LLLast\SD\merged_results_CONVERT1.csv', encoding='utf-8-sig')

# 3. direction 확장 함수: 지정된 매핑 규칙에 따라 새로운 행 생성
def expand_direction(df):
    mapping = {
        "BOTH": ["END-I", "END-J"],
        "ALL": ["END-I", "END-J", "MID"],
        "CENTER": ["MID"],
        "EXT.": ["END-J"],
        "INT.": ["END-I"]
    }
    new_rows = []
    for idx, row in df.iterrows():
        current_direction = row['Direction']
        if current_direction in mapping:
            for new_dir in mapping[current_direction]:
                new_row = row.copy()
                new_row['Direction'] = new_dir
                new_rows.append(new_row)
        else:
            # 매핑 규칙에 없는 경우는 그대로 유지
            new_rows.append(row)
    return pd.DataFrame(new_rows)

# direction 확장 적용
df_direction_expanded = expand_direction(df_expanded)

# 결과 CSV 파일 저장 (한글 깨짐 방지를 위해 인코딩 옵션 지정)
df_direction_expanded.to_csv(r'D:\LLLast\SD\merged_results_CONVERT2.csv', index=False, encoding='utf-8-sig')

print("Direction 전처리 완료. 결과는 result_with_direction.csv에 저장되었습니다.")





