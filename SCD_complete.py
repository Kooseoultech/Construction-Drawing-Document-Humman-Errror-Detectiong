# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 16:44:11 2025

@author: S3PARC
"""

#구조계산서 이미지 load 
######################################################################
import os
from pdf2image import convert_from_path

# PDF 파일이 들어 있는 폴더 경로
pdf_folder = r'D:\Doclaynet_yolo\SCD_data\raw'
output_folder = r'D:\LLLast\SCD'

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
#############################################################################
#
#
#############################################################################
#추출한 구조 계산서 이미지에 대한 Doclaynet-yolo 적용
#DocLayNet YOLO


from huggingface_hub import hf_hub_download
from doclayout_yolo import YOLOv10
from doclayout_yolo.nn.tasks import YOLOv10DetectionModel  # 허용할 전역 객체
import torch
import torch.serialization
import cv2

# Hugging Face에서 모델 다운로드
filepath = hf_hub_download(
    repo_id="juliozhao/DocLayout-YOLO-DocStructBench",
    filename="doclayout_yolo_docstructbench_imgsz1024.pt"
)

# 안전한 전역 컨텍스트를 사용하여 모델 로드
with torch.serialization.safe_globals([YOLOv10DetectionModel]):
    model = YOLOv10(filepath)

# 디바이스 설정: GPU 사용 가능하면 GPU, 아니면 CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"사용 중인 디바이스: {device}")



# 출력 폴더 생성


import os
import cv2
from doclayout_yolo import YOLOv10  # DocLayout-YOLO 라이브러리에서 모델 가져오기

# 모델 파일 경로 및 입력/출력 경로 설정
model_path = r"C:\Users\S3PARC\.cache\huggingface\hub\models--juliozhao--DocLayout-YOLO-DocStructBench\snapshots\8c3299a30b8ff29a1503c4431b035b93220f7b11\doclayout_yolo_docstructbench_imgsz1024.pt"
input_folder = r"D:\LLLast\SCD\raw_data"   # 이미지가 포함된 폴더 경로
output_folder_1st = r"D:\LLLast\SCD\DocLayNey_yolo"  # 1차 결과 저장 폴더
os.makedirs(output_folder_1st, exist_ok=True)  # 출력 폴더 생성

# Confidence 및 NMS IOU Threshold 설정
confidence_threshold = 0.2  # 신뢰도 기준
iou_threshold = 0.1 # NMS IOU Threshold

# 모델 로드
model = YOLOv10(model_path)

# 폴더 내 모든 이미지 파일 처리
image_files = [f for f in os.listdir(input_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
print(f"총 {len(image_files)}개의 이미지를 처리합니다.")

for image_file in image_files:
    input_file = os.path.join(input_folder, image_file)  # 각 이미지의 전체 경로
    print(f"처리 중: {image_file}")




    # 이미지 분석 (NMS IOU Threshold 추가)
    det_res = model.predict(
        input_file,
        imgsz=1024,
        conf=confidence_threshold,
        iou=iou_threshold,  # NMS IOU Threshold 설정
        device="cpu"
    )

    # 검출된 객체 정보
    boxes = det_res[0].boxes
    names = det_res[0].names

    # 원본 이미지 로드
    image = cv2.imread(input_file)

    # `figure`와 `plain text` 필터링 및 시각화
    for box in boxes:
        cls_id = int(box.cls.item())
        conf = box.conf.item()
        cls_name = names[cls_id]

        if cls_name in ["figure", "plain text"] and conf >= confidence_threshold:
            # 박스 좌표 가져오기
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            
            # 색상 및 라벨 설정
            if cls_name == "figure":
                color = (255, 0, 0)  # 파란색
            elif cls_name == "plain text":
                color = (0, 255, 0)  # 초록색
            
            # 박스 및 라벨 이미지에 표시
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            label = f"{cls_name} {conf:.2f}"
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 5.0, color, 2)

    # 시각화된 이미지 저장
    output_path = os.path.join(output_folder_1st, f"{os.path.splitext(image_file)[0]}_visualized.png")
    cv2.imwrite(output_path, image)

    print(f"시각화된 결과 저장 완료: {output_path}")

print("모든 이미지 처리가 완료되었습니다!")


##############################################################
#파일 이름 변경

import os

# 폴더 경로 설정
folder_path = r"D:\LLLast\SCD\raw_data"  # 파일들이 있는 폴더 경로

# 폴더 내 파일 목록 가져오기 (파일만 선택)
files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

# 파일명 변경: 모든 파일명을 "SCD_x" 형태로 변경 (x는 순번)
for i, file_name in enumerate(files, start=1):
    # 파일 확장자 분리
    _, ext = os.path.splitext(file_name)
    # 새로운 파일명 생성
    new_file_name = f"SCD_{i}{ext}"
    
    # 기존 파일 경로 및 새 파일 경로 설정
    old_file_path = os.path.join(folder_path, file_name)
    new_file_path = os.path.join(folder_path, new_file_name)
    
    # 파일 이름 변경
    os.rename(old_file_path, new_file_path)
    print(f"'{file_name}' -> '{new_file_name}'로 변경 완료")

print("모든 파일명이 성공적으로 변경되었습니다!")




#############################################################이미지로부터 PLAIN TEXT, FIGURE 분리


from huggingface_hub import hf_hub_download
from doclayout_yolo import YOLOv10
from doclayout_yolo.nn.tasks import YOLOv10DetectionModel
import torch
import torch.serialization

# 안전한 전역 객체 추가
torch.serialization.add_safe_globals([YOLOv10DetectionModel])

# 모델 파일 다운로드
filepath = hf_hub_download(
    repo_id="juliozhao/DocLayout-YOLO-DocStructBench",
    filename="doclayout_yolo_docstructbench_imgsz1024.pt"
)

# 모델 로드
model = YOLOv10(filepath)

print("모델이 성공적으로 로드되었습니다.")



# 입력 및 출력 경로 설정
input_folder = r"D:\LLLast\SCD\raw_data"  # 입력 이미지 폴더
visualized_folder = r"D:\LLLast\SCD\Visualized"  # 전체 박스 시각화 이미지 저장
plain_text_folder = r"D:\LLLast\SCD\Plain_Texts"  # plain text 저장 폴더
figure_folder = r"D:\LLLast\SCD\Figures"  # figure 저장 폴더
split_figure_folder = r"D:\LLLast\SCD\Split_Figures"  # figure 3등분 저장 폴더
os.makedirs(visualized_folder, exist_ok=True)
os.makedirs(plain_text_folder, exist_ok=True)
os.makedirs(figure_folder, exist_ok=True)
os.makedirs(split_figure_folder, exist_ok=True)

# 폴더 초기화 함수
def clear_folder(folder_path):
    """폴더 내 모든 파일 삭제"""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    else:
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            if os.path.isfile(file_path):
                os.remove(file_path)

# 폴더 초기화
clear_folder(visualized_folder)
clear_folder(plain_text_folder)
clear_folder(figure_folder)
clear_folder(split_figure_folder)

# 클래스별 Confidence 기준 설정
confidence_thresholds = {
    "figure": 0.3,  # figure의 Confidence Threshold
    "plain text": 0.5  # plain text의 Confidence Threshold
}

# 여백 설정 (픽셀 단위, 위/아래/좌/우)
top_padding = 100
bottom_padding = 30
left_padding = 10
right_padding = 40

# figure 크기 기준
min_figure_width = 1500  # figure의 최소 너비
min_figure_height = 1100  # figure의 최소 높이

# plain text 세로 크기 기준
min_plain_text_height = 400  # Plain Text 박스의 최소 높이

# 처리할 이미지 파일 가져오기
image_files = [f for f in os.listdir(input_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
print(f"총 {len(image_files)}개의 이미지를 처리합니다.")

for image_file in image_files:
    input_file = os.path.join(input_folder, image_file)
    print(f"처리 중: {image_file}")

    # 모델 예측
    det_res = model.predict(input_file, imgsz=1024, device="cpu")
    boxes = det_res[0].boxes
    names = det_res[0].names

    # 원본 이미지 로드
    image = cv2.imread(input_file)
    img_height, img_width = image.shape[:2]
    visualized_image = image.copy()

    # 검출된 박스 정리
    plain_text_boxes = []
    figure_boxes = []

    for box in boxes:
        cls_id = int(box.cls.item())
        conf = box.conf.item()
        cls_name = names[cls_id]
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        if cls_name == "plain text" and conf >= confidence_thresholds["plain text"]:
            box_height = y2 - y1  # 박스의 높이 계산
            if box_height >= min_plain_text_height:  # 세로 크기 기준 필터링
                plain_text_boxes.append((x1, y1, x2, y2))
        elif cls_name == "figure" and conf >= confidence_thresholds["figure"]:
            figure_boxes.append((x1, y1, x2, y2))

    # 겹치는 figure 박스 처리
    filtered_figure_boxes = []
    for i, box in enumerate(figure_boxes):
        keep = True
        for j, other_box in enumerate(figure_boxes):
            if i != j:
                # 겹치는 경우 큰 박스만 남김
                bx1, by1, bx2, by2 = other_box
                if (
                    box[0] >= bx1 and box[1] >= by1 and
                    box[2] <= bx2 and box[3] <= by2
                ):
                    keep = False
                    break
        if keep:
            filtered_figure_boxes.append(box)

    # 겹치지 않는 경우 작은 박스만 남기기
    if len(filtered_figure_boxes) > 1:
        filtered_figure_boxes.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        filtered_figure_boxes = [filtered_figure_boxes[0]]

    # 최종 figure 박스
    largest_figure = filtered_figure_boxes[0] if filtered_figure_boxes else None

    # plain text 저장
    for idx, (x1, y1, x2, y2) in enumerate(plain_text_boxes):
        cropped_plain_text = image[y1:y2, x1:x2]
        plain_text_path = os.path.join(plain_text_folder, f"{os.path.splitext(image_file)[0]}_plain{idx}.png")
        cv2.imwrite(plain_text_path, cropped_plain_text)

    # figure 처리 및 저장
    if largest_figure:
        x1, y1, x2, y2 = largest_figure
        x1_padded = max(x1 - left_padding, 0)
        y1_padded = max(y1 - top_padding, 0)
        x2_padded = min(x2 + right_padding, img_width)
        y2_padded = min(y2 + bottom_padding, img_height)

        cropped_figure = image[y1_padded:y2_padded, x1_padded:x2_padded]
        figure_path = os.path.join(figure_folder, f"{os.path.splitext(image_file)[0]}_figure.png")
        cv2.imwrite(figure_path, cropped_figure)

        # 크기 기준 확인
        figure_width = x2_padded - x1_padded
        figure_height = y2_padded - y1_padded

        if figure_width >= min_figure_width and figure_height >= min_figure_height:
            # 3등분 수행
            split_width = figure_width // 3
            for part in range(3):
                part_x1 = split_width * part
                part_x2 = split_width * (part + 1) if part < 2 else figure_width
                split_image = cropped_figure[:, part_x1:part_x2]

                split_path = os.path.join(split_figure_folder, f"{os.path.splitext(image_file)[0]}_figure_part{part + 1}.png")
                cv2.imwrite(split_path, split_image)
        else:
            # 크기가 기준 미달 → 그대로 저장
            split_path = os.path.join(split_figure_folder, f"{os.path.splitext(image_file)[0]}_figure.png")
            cv2.imwrite(split_path, cropped_figure)

    # 박스 시각화
    for (x1, y1, x2, y2) in plain_text_boxes:
        cv2.rectangle(visualized_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(visualized_image, "plain text", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

  
    if largest_figure:
        x1, y1, x2, y2 = largest_figure
        cv2.rectangle(visualized_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(visualized_image, "figure", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    visualized_path = os.path.join(visualized_folder, f"{os.path.splitext(image_file)[0]}_visualized.png")
    cv2.imwrite(visualized_path, visualized_image)

print("모든 이미지 처리가 완료되었습니다!")

###############################################

#############################################이제 PLAIN TEXT로부터 MEMBER코드 추출 실시- SUYA 쓴 결과 활용할거임. 추출된 SUYA 파일별로 분리 + 여기서 이제 로직 적용해서 뽑아낼 예정
#Surya-OCR

import os
import subprocess
import shutil

# GPU 비활성화: CUDA_VISIBLE_DEVICES를 빈 문자열로 설정하여 PyTorch가 GPU를 인식하지 않도록 함.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# 입력 및 출력 폴더 경로 설정
input_folder = r"D:\LLLast\SCD\Plain_Texts"
output_folder = r"D:\LLLast\SCD\Surya"

# Surya-OCR 실행 (출력 폴더 옵션이 없으므로 기본 위치에 저장됨)
command = ["surya_ocr", input_folder]

# 명령어 실행
result = subprocess.run(command, capture_output=True, text=True)

# 실행 결과 출력
print("stdout:")
print(result.stdout)
if result.stderr:
    print("stderr:")
    print(result.stderr)

# Surya-OCR 기본 출력 폴더 확인 (출력된 파일이 어디에 저장되는지 확인)
default_output_folder = os.path.join(input_folder, "output")  # Surya-OCR의 기본 출력 폴더(추정)

# 기본 출력 폴더에서 원하는 출력 폴더로 파일 이동
if os.path.exists(default_output_folder):
    for file in os.listdir(default_output_folder):
        shutil.move(os.path.join(default_output_folder, file), os.path.join(output_folder, file))

    print(f"결과 파일을 {output_folder}로 이동 완료.")
else:
    print("Surya-OCR의 기본 출력 폴더를 찾을 수 없습니다. Surya-OCR이 생성한 결과 위치를 확인하세요.")



###############################################
#1) 파일별 txt 파일 분할



import os
import json

# 입력 JSON 파일 및 출력 폴더 설정
input_file = r"D:\LLLast\SCD\Plain_Texts\results.json"  # JSON 파일 경로
output_folder = r"D:\LLLast\SCD\split"  # 분리된 JSON 파일을 저장할 폴더
os.makedirs(output_folder, exist_ok=True)  # 출력 폴더 생성

# JSON 데이터 로드
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# 이미지 이름별로 데이터 분리 및 저장
for image_name, items in data.items():
    # 출력 파일 경로 설정
    output_file = os.path.join(output_folder, f"{image_name}.json")
    
    # 현재 데이터 저장
    with open(output_file, "w", encoding="utf-8") as out_f:
        json.dump({image_name: items}, out_f, indent=4, ensure_ascii=False)
    
    print(f"파일 저장 완료: {output_file}")

print(f"모든 데이터를 '{output_folder}' 폴더에 저장했습니다.")
##########

import os
import json

# 입력 폴더 및 출력 폴더 설정
input_folder = r"D:\LLLast\SCD\split"  # JSON 파일이 있는 폴더
output_folder = r"D:\LLLast\SCD\filter_no_member_out"  # 필터링된 JSON 저장 폴더
os.makedirs(output_folder, exist_ok=True)  # 출력 폴더 생성

# 입력 폴더 내 모든 JSON 파일 처리
files = [f for f in os.listdir(input_folder) if f.endswith('.json')]

for file_name in files:
    input_file = os.path.join(input_folder, file_name)
    output_file = os.path.join(output_folder, file_name)

    # JSON 데이터 로드
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # text와 bbox 필드만 추출
    filtered_data = []
    contains_member = False  # "Member" 키워드 포함 여부 확인

    for key, values in data.items():
        for item in values:
            if "text_lines" in item:
                for line in item["text_lines"]:
                    if "text" in line and "bbox" in line:
                        text_value = line["text"]
                        bbox_value = line["bbox"]
                        filtered_data.append({
                            "text": text_value,
                            "bbox": bbox_value
                        })
                        
                        # "Member" 키워드가 포함되어 있는지 확인
                        if "Member" in text_value:
                            contains_member = True

    # "Member"가 포함된 경우만 저장
    if contains_member:
        with open(output_file, "w", encoding="utf-8") as out_f:
            json.dump(filtered_data, out_f, indent=4, ensure_ascii=False)
        print(f"{file_name}에서 'Member' 포함된 text와 bbox만 추출하여 저장 완료: {output_file}")
    else:
        print(f"{file_name}에는 'Member'가 포함되지 않아 제외됨.")

print(f"모든 작업이 완료되었습니다! 결과는 '{output_folder}' 폴더에 저장되었습니다.")

##########
#MEMBER 추출 + rebar pattenr + section property

import os
import json
import re

# 입력 폴더 및 출력 폴더 설정
input_folder = r"D:\LLLast\SCD\filter_no_member_out"  # 필터링된 JSON 파일 폴더
output_folder = r"D:\LLLast\SCD\results"  # 결과 저장 폴더
os.makedirs(output_folder, exist_ok=True)

# y축 오차 허용 범위 (픽셀 단위)
y_tolerance = 20

# 입력 폴더 내 모든 JSON 파일 처리
files = [f for f in os.listdir(input_folder) if f.endswith('.json')]

for file_name in files:
    input_file = os.path.join(input_folder, file_name)
    output_file = os.path.join(output_folder, file_name)

    # JSON 데이터 로드
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 기본 앵커: "Member" 박스 (앞뒤 공백 제거 후 정확히 "Member"인 경우)
    member_anchors = [item for item in data if item["text"].strip() == "Member"]

    # "Rebar Pattern" 박스 추출 (공백과 대소문자 무시)
    rebar_anchors = [item for item in data if re.sub(r"\s+", "", item["text"].strip()).lower() == "rebarpattern"]

    # "Section Property" 박스 추출 (공백과 대소문자 무시)
    section_anchors = [item for item in data if re.sub(r"\s+", "", item["text"].strip()).lower() == "sectionproperty"]

    # "Member" 박스가 없으면 처리하지 않음
    if not member_anchors:
        print(f"{file_name}에 'Member'가 없어 파일을 생성하지 않습니다.")
        continue

    # 앵커 박스 결정: 기본은 Member, 있으면 Rebar Pattern, 있으면 Section Property도 추가
    anchor_boxes = member_anchors.copy()
    if rebar_anchors:
        anchor_boxes += rebar_anchors
    if section_anchors:
        anchor_boxes += section_anchors

    # 앵커 박스와 y축이 유사한 박스 추출 (중복 제거)
    aligned_boxes = []
    seen = set()  # 중복 추가 방지를 위한 id 저장

    for anchor in anchor_boxes:
        anchor_y1 = anchor["bbox"][1]
        anchor_y2 = anchor["bbox"][3]
        for item in data:
            if item is anchor:
                continue
            item_y1 = item["bbox"][1]
            item_y2 = item["bbox"][3]
            # 앵커 박스의 시작 혹은 끝 y 좌표와 item의 y 좌표 차이가 허용 범위 이내면 추출
            if abs(anchor_y1 - item_y1) <= y_tolerance or abs(anchor_y2 - item_y2) <= y_tolerance:
                if id(item) not in seen:
                    aligned_boxes.append(item)
                    seen.add(id(item))

    # 결과 저장
    with open(output_file, "w", encoding="utf-8") as out_f:
        json.dump(aligned_boxes, out_f, indent=4, ensure_ascii=False)

    print(f"{file_name}에서 추출 완료: {output_file}")

print(f"모든 작업이 완료되었습니다! 결과는 '{output_folder}' 폴더에 저장되었습니다.")

#################


import os
import json

# 입력 폴더 및 출력 폴더 설정
input_folder = r"D:\LLLast\SCD\results"  # 기존 JSON 파일이 있는 폴더
output_folder = r"D:\LLLast\SCD\Surya\MEMBER_EXTRACTION"  # 결과 텍스트 파일을 저장할 폴더
os.makedirs(output_folder, exist_ok=True)

# 입력 폴더 내 모든 JSON 파일 처리
files = [f for f in os.listdir(input_folder) if f.endswith('.json')]

for file_name in files:
    input_file = os.path.join(input_folder, file_name)
    output_file = os.path.join(output_folder, f"{os.path.splitext(file_name)[0]}.txt")

    # JSON 데이터 로드
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # "text" 필드만 추출 (순서대로 저장되어 있다고 가정)
    texts = [item["text"].strip() for item in data if "text" in item]

    # 순서대로 추출:
    # 첫 번째 텍스트: Member
    # 두 번째 텍스트: Rebar Pattern (없으면 빈 문자열)
    # 세 번째 텍스트: Section Property (없으면 빈 문자열)
    member_text = texts[0] if len(texts) >= 1 else ""
    rebar_pattern_text = texts[1] if len(texts) >= 2 else ""
    section_property_text = texts[2] if len(texts) >= 3 else ""

    # 텍스트 파일에 라벨을 붙여 저장
    with open(output_file, "w", encoding="utf-8") as out_f:
        out_f.write(f"Member: {member_text}\n")
        out_f.write(f"Rebar Pattern: {rebar_pattern_text}\n")
        out_f.write(f"Section Property: {section_property_text}\n")
    
    print(f"{file_name}에서 텍스트 추출 완료: {output_file}")

print(f"모든 작업이 완료되었습니다! 결과는 '{output_folder}' 폴더에 저장되었습니다.")


##############################################FIGURE에서 뽑아낸 정보 정리  FIGURE 대상으로 한번 더 OCR 돌릴거임

# SURYA=== 결과파일 수동으로 경로 변경해야함. or 결과치 나오는 폴더로 경로 지정 해야함.

import os
import subprocess
import shutil
from tqdm import tqdm
import time

# GPU 비활성화: CUDA_VISIBLE_DEVICES를 빈 문자열로 설정하여 PyTorch가 GPU를 인식하지 않도록 함.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# 입력 및 출력 폴더 경로 설정
input_folder = r"D:\LLLast\SCD\Figures"
output_folder = r"D:\LLLast\SCD"

# Surya-OCR 실행 (출력 폴더 옵션이 없으므로 기본 위치에 저장됨)
command = ["surya_ocr", input_folder]

print("Surya-OCR 실행 중... (이 작업은 시간이 걸릴 수 있습니다.)")

# tqdm으로 진행 상태 표시
with tqdm(total=1, desc="Surya-OCR 실행", unit="step") as pbar:
    result = subprocess.run(command, capture_output=True, text=True)
    pbar.update(1)  # 실행 완료되면 100% 표시

# 실행 결과 출력
print("\nOCR 실행 완료!")
print("stdout:")
print(result.stdout)
if result.stderr:
    print("stderr:")
    print(result.stderr)

# Surya-OCR 기본 출력 폴더 확인 (출력된 파일이 어디에 저장되는지 확인)
default_output_folder = os.path.join(input_folder, "output")  # Surya-OCR의 기본 출력 폴더(추정)

# 결과 파일을 원하는 출력 폴더로 이동
if os.path.exists(default_output_folder):
    files = os.listdir(default_output_folder)

    if not files:
        print("⚠️ Surya-OCR의 기본 출력 폴더에 파일이 없습니다. OCR 결과를 확인하세요.")
    else:
        print(f"총 {len(files)}개의 OCR 결과 파일을 이동 중...")

        # tqdm으로 파일 이동 진행률 표시
        for file in tqdm(files, desc="파일 이동", unit="file"):
            src_path = os.path.join(default_output_folder, file)
            dest_path = os.path.join(output_folder, file)
            shutil.move(src_path, dest_path)
            time.sleep(0.1)  # UI 표시를 위해 짧은 대기 추가 (실제 파일 처리에는 필요 없음)

        print(f"\n✅ 결과 파일 {output_folder}로 이동 완료!")

else:
    print("❌ Surya-OCR의 기본 출력 폴더를 찾을 수 없습니다. Surya-OCR이 생성한 결과 위치를 확인하세요.")


######################################################################################



# #1) 파일명 바꾸기

# import os

# # 폴더 경로 설정
# folder_path = r"D:\LLLast\SCD\Figures"  # 파일들이 있는 폴더 경로

# # 폴더 내 파일 목록 가져오기 (파일만 선택)
# files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

# # 파일명 변경: 모든 파일명을 "SCD_Figure_x" 형태로 변경 (x는 순번)
# for i, file_name in enumerate(files, start=1):
#     # 파일 확장자 분리
#     _, ext = os.path.splitext(file_name)
#     # 새로운 파일명 생성
#     new_file_name = f"SCD_Figure_{i}{ext}"
    
#     # 기존 파일 경로 및 새 파일 경로 설정
#     old_file_path = os.path.join(folder_path, file_name)
#     new_file_path = os.path.join(folder_path, new_file_name)
    
#     # 파일 이름 변경
#     os.rename(old_file_path, new_file_path)
#     print(f"'{file_name}' -> '{new_file_name}'로 변경 완료")

# print("모든 파일명이 성공적으로 변경되었습니다!")


#2)파일별로 분리

import os
import json

# 입력 JSON 파일 및 출력 폴더 설정
input_file = r"D:\LLLast\SCD\results.json"  # JSON 파일 경로
output_folder = r"D:\LLLast\SCD\Surya\Figure\split"  # 분리된 JSON 파일을 저장할 폴더
os.makedirs(output_folder, exist_ok=True)  # 출력 폴더 생성

# JSON 데이터 로드
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# 이미지 이름별로 데이터 분리 및 저장
for image_name, items in data.items():
    # 출력 파일 경로 설정
    output_file = os.path.join(output_folder, f"{image_name}.json")
    
    # 현재 데이터 저장
    with open(output_file, "w", encoding="utf-8") as out_f:
        json.dump({image_name: items}, out_f, indent=4, ensure_ascii=False)
    
    print(f"파일 저장 완료: {output_file}")

print(f"모든 데이터를 '{output_folder}' 폴더에 저장했습니다.")

 #3) text랑 bbox만 남기고 나머지 제거
import os
import json
import math

# 입력 및 출력 폴더 설정
input_folder = r"D:\LLLast\SCD\Surya\Figure\split"  # 입력 JSON 파일 폴더
output_folder = r"D:\LLLast\SCD\Surya\Figure\filtered_txts"  # 결과 TXT 파일 저장 폴더
os.makedirs(output_folder, exist_ok=True)

# 박스 병합을 위한 허용 오차
x_tolerance = 200  # X축 중심 간 거리
y_tolerance = 30  # Y축 중심 간 거리

def merge_boxes(boxes, x_tolerance, y_tolerance):
    """
    중심점을 기준으로 텍스트 박스를 병합.
    """
    if not boxes:
        return []

    merged_boxes = []
    used = [False] * len(boxes)  # 병합된 박스 추적

    def calculate_center(bbox):
        """
        BBox의 중심 좌표를 계산.
        """
        x_min, y_min, x_max, y_max = bbox
        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2
        return cx, cy

    for i, current_box in enumerate(boxes):
        if used[i]:
            continue

        current_cx, current_cy = calculate_center(current_box["bbox"])
        combined_text = current_box["text"]
        combined_bbox = current_box["bbox"]

        for j, next_box in enumerate(boxes[i + 1:], start=i + 1):
            if used[j]:
                continue

            next_cx, next_cy = calculate_center(next_box["bbox"])
            x_distance = abs(current_cx - next_cx)
            y_distance = abs(current_cy - next_cy)

            if x_distance <= x_tolerance and y_distance <= y_tolerance:
                # 병합 수행
                combined_text += f" {next_box['text']}"
                combined_bbox = [
                    min(combined_bbox[0], next_box["bbox"][0]),
                    min(combined_bbox[1], next_box["bbox"][1]),
                    max(combined_bbox[2], next_box["bbox"][2]),
                    max(combined_bbox[3], next_box["bbox"][3]),
                ]
                used[j] = True

        merged_boxes.append({"text": combined_text, "bbox": combined_bbox})
        used[i] = True

    return merged_boxes

# 입력 폴더 내 모든 JSON 파일 처리
files = [f for f in os.listdir(input_folder) if f.endswith('.json')]

for file_name in files:
    input_file = os.path.join(input_folder, file_name)
    output_file = os.path.join(output_folder, f"{os.path.splitext(file_name)[0]}.txt")

    # JSON 데이터 로드
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 병합 대상 추출 및 병합
    all_boxes = []
    for key, values in data.items():
        for item in values:
            if "text_lines" in item:
                for line in item["text_lines"]:
                    if "text" in line and "bbox" in line:
                        all_boxes.append({"text": line["text"], "bbox": line["bbox"]})

    # X축 기준으로 박스 정렬
    all_boxes.sort(key=lambda x: (x["bbox"][1], x["bbox"][0]))  # Y축 우선, X축 정렬

    # 박스 병합
    merged_boxes = merge_boxes(all_boxes, x_tolerance, y_tolerance)

    # 결과를 TXT 파일로 저장
    lines = []
    for box in merged_boxes:
        lines.append(f"Text: {box['text']}\n")
        lines.append(f"BBox: {box['bbox']}\n")

    with open(output_file, "w", encoding="utf-8") as out_f:
        out_f.writelines(lines)

    print(f"{file_name} TXT 파일 저장 완료: {output_file}")

print(f"모든 파일 TXT 저장 작업이 완료되었습니다! 결과는 '{output_folder}' 폴더에 저장되었습니다.")
##############################################################################################


import os
import cv2
import numpy as np

# 공통 함수 정의
def clean_coordinates(coord_str):
    """
    좌표 문자열을 리스트로 변환.
    """
    coord_str = coord_str.strip("[] \n")
    try:
        return list(map(float, coord_str.split(",")))
    except ValueError:
        return None

def calculate_area(coords):
    """
    좌표 영역의 면적 계산.
    """
    x_min, y_min, x_max, y_max = coords
    return (x_max - x_min) * (y_max - y_min)

def is_valid_number(text):
    """
    텍스트가 숫자와 소수점만 포함하고 있는지 확인.
    """
    try:
        value = float(text)
        return True
    except ValueError:
        return False

def clean_text_value(text):
    """
    텍스트에서 숫자와 소수점만 남기고 나머지는 제거.
    """
    cleaned_text = ''.join([char for char in text if char.isdigit() or char == '.'])
    if cleaned_text:  # 텍스트가 비어 있지 않다면
        value = float(cleaned_text)
        if value < 0.1 or value > 5:  # 조건에 맞지 않으면 1로 설정
            return "1"
    return cleaned_text

def assign_height_and_width(filtered_boxes, image_width, image_height):
    """
    RC기둥에서 WIDTH와 HEIGHT를 결정.
    """
    width, height = 1, 1  # 기본값 설정

    # WIDTH: y좌표가 가장 작은 텍스트 박스 (이미지 끝 95% 범위 내 제외)
    bottom_boxes = [box for box in filtered_boxes if box["coords"][3] <= 0.95 * image_height]
    if bottom_boxes:
        bottommost_box = max(bottom_boxes, key=lambda box: box["coords"][3])
        width = clean_text_value(bottommost_box["text"])
        filtered_boxes.remove(bottommost_box)  # 사용된 텍스트 박스 제거

    # HEIGHT: x좌표가 가장 작은 텍스트 박스
    leftmost_boxes = [box for box in filtered_boxes]
    if leftmost_boxes:
        leftmost_box = min(leftmost_boxes, key=lambda box: box["coords"][0])
        height = clean_text_value(leftmost_box["text"])
        filtered_boxes.remove(leftmost_box)  # 사용된 텍스트 박스 제거

    return height, width

def filter_valid_boxes(boxes, min_area=800, image_width=None):
    """
    유효한 텍스트 박스를 필터링 (너무 작은 텍스트 박스 제거).
    """
    valid_boxes = []
    for box in boxes:
        text = box["text"]
        area = calculate_area(box["coords"])
        x_min, _, x_max, _ = box["coords"]

        if image_width and x_max > 0.95 * image_width:  # 이미지 끝 95% 초과 박스 제외
            continue

        if is_valid_number(text) and area >= min_area:  # 최소 면적 조건 추가
            valid_boxes.append(box)
    return valid_boxes

def adjust_text_order(text):
    """
    TOP/BOT 키워드를 기준으로 텍스트 순서를 조정.
    """
    parts = text.split()
    if "TOP" in parts or "BOT" in parts:
        keywords = [p for p in parts if "TOP" in p or "BOT" in p]
        remaining = [p for p in parts if p not in keywords]
        return " ".join(keywords + remaining)
    return text

def process_rc_beam(txt_path, output_folder, base_name):
    headers, boxes = [], []

    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if line.startswith("Text:"):
            text = line.split("Text:", 1)[1].strip()
        elif line.startswith("BBox:"):
            coords = clean_coordinates(line.split("BBox:", 1)[1])
            if coords:
                # 헤더 조건 수정
                if text.startswith("[") and text.endswith("]"):
                    headers.append({"text": text.strip("[]"), "coords": coords})
                else:
                    boxes.append({"text": text, "coords": coords})

    if not headers:
        print(f"{txt_path}: 헤더가 없습니다. 건너뜁니다.")
        return

    grouped_data = {header['text']: [] for header in headers}

    for box in boxes:
        distances = [(abs(header['coords'][0] - (box['coords'][0] + box['coords'][2]) / 2), header['text']) for header in headers]
        if distances:
            _, closest_header = min(distances)
            grouped_data[closest_header].append(box)

    os.makedirs(output_folder, exist_ok=True)
    for header, group in grouped_data.items():
        results = {"HEIGHT": None, "WIDTH": None, "TOP_REBAR": None, "BOT_REBAR": None, "STIRRUPS": None}

        # HEIGHT: 그룹화된 텍스트 박스 중 가장 좌측
        if group:
            leftmost_box = min(group, key=lambda box: box["coords"][0])
            results["HEIGHT"] = clean_text_value(leftmost_box["text"])

        # WIDTH: "TOP" 포함 텍스트의 이전 텍스트
        for i, box in enumerate(group):
            if "TOP" in box["text"]:
                if i > 0:
                    results["WIDTH"] = clean_text_value(group[i - 1]["text"])
                break

        # TOP_REBAR, BOT_REBAR, STIRRUPS 추출
        for box in group:
            text = adjust_text_order(box["text"])
            if "TOP" in text:
                results["TOP_REBAR"] = text.split("TOP", 1)[-1].strip()
            if "BOT" in text:
                results["BOT_REBAR"] = text.split("BOT", 1)[-1].strip()
            if "STIRRUPS" in text:
                results["STIRRUPS"] = text.split("STIRRUPS", 1)[-1].strip()

        output_file = os.path.join(output_folder, f"{base_name}_{header}.txt")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"=== {header} ===\n")
            for key, value in results.items():
                f.write(f"{key}: {value if value else 'N/A'}\n")
        print(f"RC보 결과 저장 완료: {output_file}")

def process_rc_column(txt_path, output_folder, base_name, image_width, image_height):
    boxes = []
    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if line.startswith("Text:"):
            text = line.split("Text:", 1)[1].strip()
        elif line.startswith("BBox:"):
            coords = clean_coordinates(line.split("BBox:", 1)[1])
            if coords:
                boxes.append({"text": text, "coords": coords})

    if not boxes:
        print(f"{txt_path}: 유효한 텍스트 박스가 없습니다.")
        return

    filtered_boxes = filter_valid_boxes(boxes, min_area=600, image_width=image_width)
    height, width = assign_height_and_width(filtered_boxes, image_width, image_height)

    output_file = os.path.join(output_folder, f"{base_name}_RC_COLUMN.txt")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("CODE: RC_COLUMN\n")
        f.write(f"HEIGHT: {height}\n")
        f.write(f"WIDTH: {width}\n")
    print(f"RC기둥 결과 저장 완료: {output_file}")

def process_all_files(txt_folder, img_folder, output_folder):
    for file_name in os.listdir(txt_folder):
        if not file_name.lower().endswith('.txt'):
            continue

        txt_path = os.path.join(txt_folder, file_name)
        base_name = os.path.splitext(file_name)[0]
        # 이미지 파일은 별도의 이미지 폴더에서 찾음.
        img_path = os.path.join(img_folder, f"{base_name}.png")

        if not os.path.exists(img_path):
            print(f"{img_path} 이미지를 찾을 수 없습니다.")
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"{img_path} 이미지를 읽을 수 없습니다.")
            continue

        image_height, image_width, _ = img.shape

        if image_width >= 1500:  # RC보 처리
            process_rc_beam(txt_path, output_folder, base_name)
        else:  # RC기둥 처리
            process_rc_column(txt_path, output_folder, base_name, image_width, image_height)


# 실행
txt_folder = r'D:\LLLast\SCD\Surya\Figure\filtered_txts'  # txt 파일 폴더
img_folder = r'D:\LLLast\SCD\Figures'         # 이미지 폴더 (예시)
output_folder = r'D:\LLLast\SCD\Surya\Figure\Element'       # 출력 폴더
os.makedirs(output_folder, exist_ok=True)

process_all_files(txt_folder, img_folder, output_folder)


##############################################################################################


import os
import csv
import re

def extract_member_data(member_folder):
    """
    Member 폴더에서 페이지별 Member 데이터를 추출.
    각 파일은 최대 세 줄을 포함할 수 있다.
      - 첫 번째 줄: "Member:" 뒤의 값.
      - 두 번째 줄 (옵션): "Rebar Pattern:" 뒤의 값.
      - 세 번째 줄 (옵션): "Section Property:" 뒤의 값.
    """
    member_data = {}
    for file_name in os.listdir(member_folder):
        # 파일명이 SCD_x로 시작하고 .txt로 끝나는 경우만 처리
        if file_name.startswith("SCD_") and file_name.endswith(".txt"):
            try:
                # 정규식을 사용하여 "SCD_(\d+)" 패턴에서 x 추출
                m = re.search(r"SCD_(\d+)", file_name)
                if not m:
                    continue
                page_number = int(m.group(1))
                with open(os.path.join(member_folder, file_name), 'r', encoding='utf-8') as f:
                    lines = [line.strip() for line in f.readlines() if line.strip()]
                
                member_value = ""
                rebar_pattern_value = ""
                section_property_value = ""
                if len(lines) >= 1:
                    member_value = lines[0].replace("Member:", "").strip()
                if len(lines) >= 2:
                    rebar_pattern_value = lines[1].replace("Rebar Pattern:", "").strip()
                if len(lines) >= 3:
                    section_property_value = lines[2].replace("Section Property:", "").strip()
                
                member_data[page_number] = {
                    "member": member_value,
                    "rebar_pattern": rebar_pattern_value,
                    "section_property": section_property_value
                }
            except ValueError:
                print(f"잘못된 파일 형식 건너뜀: {file_name}")
    return member_data

def extract_scd_data(scd_folder):
    """
    SCD 폴더에서 페이지별 SCD 데이터를 추출.
    """
    scd_data = []
    for file_name in os.listdir(scd_folder):
        # 파일명이 SCD_x로 시작하고 .txt로 끝나는 경우만 처리
        if file_name.startswith("SCD_") and file_name.endswith(".txt"):
            try:
                m = re.search(r"SCD_(\d+)", file_name)
                if not m:
                    continue
                page_number = int(m.group(1))
                # 파일명에서 마지막 부분을 추출하여 Direction 값으로 사용
                raw_direction = file_name.split("_")[-1].split(".")[0]
                direction = raw_direction if raw_direction in ["END-I", "END-J", "MID"] else ""
                data = {
                    "Page": page_number,
                    "Direction": direction,
                    "Top_rebar": None,
                    "bot_rebar": None,
                    "stirrups": None,
                    "width": None,
                    "height": None
                }
                
                with open(os.path.join(scd_folder, file_name), 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    for line in lines:
                        if line.startswith("TOP_REBAR:"):
                            data["Top_rebar"] = line.split(":", 1)[1].strip()
                        elif line.startswith("BOT_REBAR:"):
                            data["bot_rebar"] = line.split(":", 1)[1].strip()
                        elif line.startswith("STIRRUPS:"):
                            data["stirrups"] = line.split(":", 1)[1].strip()
                        elif line.startswith("WIDTH:"):
                            data["width"] = line.split(":", 1)[1].strip()
                        elif line.startswith("HEIGHT:"):
                            data["height"] = line.split(":", 1)[1].strip()
                scd_data.append(data)
            except ValueError:
                print(f"잘못된 파일 형식 건너뜀: {file_name}")
    return scd_data

def determine_type(entry):
    """
    데이터의 Type을 결정.
    """
    if entry["Top_rebar"] or entry["bot_rebar"] or entry["stirrups"]:
        return "RCBeam"
    return "RCColumn"

def is_valid_member(member_str):
    """
    member_str에 알파벳과 숫자가 모두 포함되어 있으면 유효한 값으로 간주.
    (단순 숫자만 있는 경우는 False)
    """
    if not member_str:
        return False
    return bool(re.search(r"[A-Za-z]", member_str)) and bool(re.search(r"\d", member_str))

def generate_csv(member_folder, scd_folder, output_csv):
    """
    Member 폴더와 SCD 폴더 데이터를 결합하여 CSV 파일 생성.
    기본적으로 Section Property 값을 사용하고,
    만약 Member 값이 존재하며 유효(숫자와 알파벳의 조합)하다면,
    "Section Property, Member" 형식으로 최종 Member code를 생성.
    Rebar Pattern 값이 특정 패턴 (숫자)-(숫자)-(SUHD|UHD|SHD|HD|D|SD)(숫자) 에 부합하면
    Bot_rebar 로 사용, 아니면 SCD 데이터에서 추출된 bot_rebar 사용.
    """
    pattern = re.compile(r"^(\d+)-(\d+)-(?:SUHD|UHD|SHD|HD|D|SD)\d+$")
    member_data = extract_member_data(member_folder)
    scd_data = extract_scd_data(scd_folder)
    rows = []
    
    for entry in scd_data:
        page = entry["Page"]
        if page in member_data:
            mdata = member_data[page]
            sp_value = mdata["section_property"]
            member_value = mdata["member"]
            if sp_value:
                if member_value and is_valid_member(member_value):
                    final_member_code = sp_value + ", " + member_value
                else:
                    final_member_code = sp_value
            else:
                if member_value and is_valid_member(member_value):
                    final_member_code = member_value
                else:
                    final_member_code = "N/A"
            
            rebar_pattern = mdata["rebar_pattern"].replace(":", "").replace(" ", "") if mdata["rebar_pattern"] else ""
        else:
            final_member_code = "N/A"
            rebar_pattern = ""
        
        entry["Type"] = determine_type(entry)
        
        if rebar_pattern:
            if pattern.match(rebar_pattern):
                final_bot_rebar = rebar_pattern
            else:
                final_bot_rebar = entry["bot_rebar"] if entry["bot_rebar"] else ""
        else:
            final_bot_rebar = entry["bot_rebar"] if entry["bot_rebar"] else ""
        
        row = {
            "Page": page,
            "Member code": final_member_code,
            "Direction": entry["Direction"] if entry["Direction"] in ["END-I", "END-J", "MID"] else "",
            "Top_rebar": entry["Top_rebar"] if entry["Top_rebar"] else "",
            "Bot_rebar": final_bot_rebar,
            "Stirrups": entry["stirrups"] if entry["stirrups"] else "",
            "Width": entry["width"] if entry["width"] else "",
            "Height": entry["height"] if entry["height"] else "",
            "Top_Rebar_Img": "",
            "Bot_Rebar_Img": ""
        }
        rows.append(row)
    
    rows = sorted(rows, key=lambda x: x["Page"])
    
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = [
            "Page", "Member code", "Direction",
            "Top_rebar", "Bot_rebar", "Stirrups",
            "Width", "Height", "Top_Rebar_Img", "Bot_Rebar_Img"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
        print(f"CSV 파일 저장 완료: {output_csv}")

# 경로 설정 (환경에 맞게 수정)
member_folder = r'D:\LLLast\SCD\Surya\MEMBER_EXTRACTION'
scd_folder = r'D:\LLLast\SCD\Surya\Figure\Element'
output_csv = r'D:\LLLast\SCD\final_results.csv'

# 실행
generate_csv(member_folder, scd_folder, output_csv)


################################################################################################################

#1) 이미지에서 사각형으로 된 도면 이미지 추출

import cv2
import os

def process_large_images(input_image_path, output_folder, min_width=50, min_height=50):
    """
    이미지 크기가 1500 이상인 경우, 이미지를 3등분하고 각 조각에 해당하는 사각형 추출 및 저장.
    """
    # 이미지 읽기
    img_color = cv2.imread(input_image_path, cv2.IMREAD_COLOR)
    if img_color is None:
        print(f"이미지를 불러올 수 없습니다: {input_image_path}")
        return

    # 이미지 크기 확인
    height, width, _ = img_color.shape
    one_third_width = width // 3  # 이미지 가로를 3등분

    # 그레이스케일 변환
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    # 이진화 처리
    _, img_binary = cv2.threshold(img_gray, 120, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # 컨투어 검출
    contours, _ = cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 사각형 검출 및 조각별 저장
    regions = ["END-I", "MID", "END-J"]
    region_boundaries = [(0, one_third_width), (one_third_width, 2 * one_third_width), (2 * one_third_width, width)]

    count = 0
    for cnt in contours:
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # 사각형 조건: 꼭짓점이 4개 + 닫힌 윤곽선 + Convex 확인
        if len(approx) == 4 and cv2.isContourConvex(approx):
            x, y, w, h = cv2.boundingRect(approx)

            # 크기 조건 필터링
            if w >= min_width and h >= min_height:
                # 사각형의 중심점 계산
                center_x = x + w // 2

                # 어느 조각에 속하는지 확인
                for i, (start, end) in enumerate(region_boundaries):
                    if start <= center_x < end:
                        region = regions[i]
                        cropped = img_color[y:y + h, x:x + w]
                        output_path = os.path.join(
                            output_folder,
                            f"{os.path.basename(input_image_path).split('.')[0]}_{region}.png"
                        )
                        cv2.imwrite(output_path, cropped)
                        count += 1

    print(f"{count}개의 사각형이 추출되었습니다: {input_image_path}")


def process_small_images(input_image_path, output_folder):
    """
    이미지 크기가 1500 미만인 경우, 가장 큰 사각형만 추출하여 저장.
    """
    # 이미지 읽기
    image = cv2.imread(input_image_path)
    if image is None:
        print(f"이미지를 불러올 수 없습니다: {input_image_path}")
        return

    # 그레이스케일 변환
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 블러링 (노이즈 제거)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 이진화 (Threshold)
    _, binary = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # 컨투어 찾기
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 컨투어를 감싸는 사각형 중 가장 큰 사각형 찾기
    max_area = 0
    largest_rectangle = None
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if area > max_area:
            max_area = area
            largest_rectangle = (x, y, w, h)

    if largest_rectangle:
        x, y, w, h = largest_rectangle
        cropped = image[y:y + h, x:x + w]
        output_path = os.path.join(output_folder, f"{os.path.basename(input_image_path).split('.')[0]}.png")
        cv2.imwrite(output_path, cropped)
        print(f"가장 큰 사각형 추출 완료: {output_path}")


# 입력 및 출력 폴더 설정
input_folder = r"D:\LLLast\SCD\Figures"  # 입력 이미지 폴더
output_folder = r"D:\LLLast\SCD\Circle\rectangles"  # 추출된 사각형 저장 폴더
os.makedirs(output_folder, exist_ok=True)

# 폴더 내 모든 이미지 처리
for file_name in os.listdir(input_folder):
    if file_name.endswith(".png"):
        input_path = os.path.join(input_folder, file_name)

        # 이미지 크기 확인
        image = cv2.imread(input_path)
        if image is None:
            print(f"이미지를 불러올 수 없습니다: {input_path}")
            continue

        height, width, _ = image.shape

        if width >= 1500:  # 가로 길이가 1500 이상인 이미지
            process_large_images(input_path, output_folder)
        else:  # 가로 길이가 1500 미만인 이미지
            process_small_images(input_path, output_folder)
            
            
##############################################################################################
#2) 이제 도면 내 철근 인식(동그라미) ################################################################# yolov5 
#2-1) 이진화

import os
import cv2

def otsu_binarization(input_folder, output_folder):
    """
    input_folder 내 모든 이미지 파일을 읽어 Otsu 이진화 수행 후,
    output_folder에 저장합니다.
    """
    # 출력 폴더가 존재하지 않으면 생성
    os.makedirs(output_folder, exist_ok=True)

    # 이미지 파일 확장자 목록 (필요 시 추가)
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')

    for file_name in os.listdir(input_folder):
        # 확장자를 보고 이미지인지 판별
        if file_name.lower().endswith(valid_extensions):
            img_path = os.path.join(input_folder, file_name)

            # 이미지를 그레이스케일로 읽기
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"이미지를 읽을 수 없습니다: {img_path}")
                continue

            # Otsu 이진화
            # threshold=0, maxValue=255, type=cv2.THRESH_BINARY | cv2.THRESH_OTSU
            _, binary_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

            # 결과 저장 경로 설정
            output_path = os.path.join(output_folder, file_name)
            
            # 이진화된 이미지 저장
            cv2.imwrite(output_path, binary_img)
            print(f"이진화 완료: {output_path}")

# 예시 실행
input_folder = r"D:\LLLast\SCD\Circle\rectangles"  # 이진화할 이미지 폴더
output_folder = r"D:\LLLast\SCD\Circle\rectangles_binarized"  # 결과 저장 폴더

otsu_binarization(input_folder, output_folder)

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
command = r'python detect.py --weights "D:\SD_final_summary\Img and Label for Trans learn\custom_rebar2\weights\best.pt" --source "D:\LLLast\SCD\Circle\rectangles_binarized" --img 640 --conf 0.5 --iou-thres 0.3 --save-txt --project "D:\LLLast\SCD\Circle\yolov5_Rebar_recog" --name exp'



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
results_dir = r"D:\LLLast\SCD\Circle\yolov5_Rebar_recog\exp\labels"

# 결과 데이터를 저장할 리스트
data = []

# 폴더 내의 모든 .txt 파일 처리
for txt_file in os.listdir(results_dir):
    if not txt_file.endswith('.txt'):
        continue

    file_path = os.path.join(results_dir, txt_file)
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 파일명에 특정 키워드(END-I, END-J, MID)가 포함되면 RCBeam으로 인식
    is_beam = any(keyword in txt_file for keyword in ["END-I", "END-J", "MID"])
    
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
output_csv = r"D:\LLLast\SCD\Circle\rebar_counts3.csv"
df.to_csv(output_csv, index=False, encoding="utf-8-sig")

print("CSV 파일이 저장되었습니다:", output_csv)
#######################################################################################################


# import pandas as pd
# import re

# # rebar_counts3 파일과 transformed_results 파일 읽기
# df_rebar = pd.read_csv(r'D:\LLLast\SCD\Circle\rebar_counts3.csv')
# df_trans = pd.read_csv(r'D:\LLLast\SCD\final_results.csv')



# def extract_page_and_type(filename):
#     """
#     파일명에서 page 번호와 rebar 타입(END-I, END-J, MID)이 있으면 추출.
#     없으면 rebar 타입은 빈 문자열 반환.
#     """
#     # page 번호 추출: "page_83" 등
#     page_match = re.search(r'page_(\d+)', filename)
#     page = page_match.group(1) if page_match else None
    
#     # rebar 타입 추출: "END-I", "END-J", "MID"
#     rebar_match = re.search(r'(END-I|END-J|MID)', filename)
#     rebar_type = rebar_match.group(1) if rebar_match else ""
    
#     return page, rebar_type

# # df_rebar에 page, rebar_type 열 추가 후 key 생성
# df_rebar[['page', 'rebar_type']] = df_rebar['File'].apply(lambda x: pd.Series(extract_page_and_type(x)))
# df_rebar['key'] = df_rebar.apply(lambda row: f"{row['page']}_{row['rebar_type']}" if row['rebar_type'] else str(row['page']), axis=1)

# # transformed_results 파일에서는 Page 열을 문자열로 변환 후, Direction 값을 이용해 key 생성
# if 'Direction' in df_trans.columns:
#     df_trans['key'] = df_trans.apply(
#         lambda row: f"{str(row['Page'])}_{row['Direction']}" 
#         if row['Direction'] in ["END-I", "END-J", "MID"] else str(row['Page']), axis=1
#     )
# else:
#     df_trans['key'] = df_trans['Page'].astype(str)

# # 만약 df_trans에 이미 "Top_Rebar_Img"와 "Bot_Rebar_Img" 열이 있다면 제거 (중복 방지)
# for col in ["Top_Rebar_Img", "Bot_Rebar_Img"]:
#     if col in df_trans.columns:
#         df_trans.drop(columns=[col], inplace=True)

# # 두 DataFrame 병합 (df_trans 기준, df_rebar에서 필요한 열만 가져옴)
# df_merged = pd.merge(df_trans, df_rebar[['key', 'Top_Rebar_Img', 'Bot_Rebar_Img']], on='key', how='left')

# # 최종 결과에서 key 열 제거
# df_merged.drop(columns=['key'], inplace=True)

# # 결과 CSV 저장
# output_csv_merged = r'D:\last_check\SCD\merged_results.csv'
# df_merged.to_csv(output_csv_merged, index=False, encoding="utf-8-sig")

# print("병합된 CSV 파일이 저장되었습니다:", output_csv_merged)

import os
import csv
import re
import pandas as pd

def extract_page_and_type(filename):
    """
    파일명에서 페이지 번호와, 파일명이 figure인 경우 방향(END-I, END-J, MID)이 있으면 추출.
    예)
      "SCD_1_plain0.txt" → ("1", "")
      "SCD_1_figure_END-I.txt" → ("1", "END-I")
      "SCD_1_figure.txt" → ("1", "")
    """
    # "SCD_x" 패턴에서 x 추출
    m = re.search(r"SCD_(\d+)", filename)
    page = m.group(1) if m else None
    # figure 파일에서 방향 추출 (예: _figure_END-I 또는 _figureEND-I)
    m2 = re.search(r"_figure(?:_)?(END-I|END-J|MID)", filename)
    rebar_type = m2.group(1) if m2 else ""
    return page, rebar_type

# df_rebar (rebar_counts3 파일) 읽기
df_rebar = pd.read_csv(r'D:\LLLast\SCD\Circle\rebar_counts3.csv')
# final_results 파일 읽기
df_trans = pd.read_csv(r'D:\LLLast\SCD\final_results.csv')

# df_rebar: File 열에서 페이지와 rebar 타입 추출 및 키 생성
df_rebar[['page', 'rebar_type']] = df_rebar['File'].apply(lambda x: pd.Series(extract_page_and_type(x)))
df_rebar['key'] = df_rebar.apply(lambda row: f"{row['page']}_{row['rebar_type']}" if row['rebar_type'] else str(row['page']), axis=1)

# df_trans: Page 열은 숫자, Direction 열이 있으면 해당 값 사용해 키 생성
if 'Direction' in df_trans.columns:
    df_trans['key'] = df_trans.apply(lambda row: f"{str(row['Page'])}_{row['Direction']}" if row['Direction'] in ["END-I", "END-J", "MID"] else str(row['Page']), axis=1)
else:
    df_trans['key'] = df_trans['Page'].astype(str)

# 만약 df_trans에 이미 "Top_Rebar_Img"와 "Bot_Rebar_Img" 열이 있다면 제거
for col in ["Top_Rebar_Img", "Bot_Rebar_Img"]:
    if col in df_trans.columns:
        df_trans.drop(columns=[col], inplace=True)

# 두 DataFrame 병합 (df_trans 기준, df_rebar에서 필요한 열만 가져옴)
df_merged = pd.merge(df_trans, df_rebar[['key', 'Top_Rebar_Img', 'Bot_Rebar_Img']], on='key', how='left')

# 최종 결과에서 키 열 제거
df_merged.drop(columns=['key'], inplace=True)

# 결과 CSV 저장
output_csv_merged = r'D:\LLLast\SCD\merged_results.csv'
df_merged.to_csv(output_csv_merged, index=False, encoding="utf-8-sig")

print("병합된 CSV 파일이 저장되었습니다:", output_csv_merged)

###############################################################################
#표준화 + 중복 추출 제거

import os
import re
import pandas as pd

def remove_colon_suffix(s: str) -> str:
    """콜론(:)과 그 뒤의 숫자를 제거한다."""
    return re.sub(r":\d+", "", s).strip()

def transform_rc_column_format(member_str):
    """
    Member code 형식을 변환하고, 결과를 리스트로 반환.
    
    처리 순서:
      1. 쉼표(,)가 있으면 분리 후 각각 처리.
      2. tilde(~) 패턴 처리:
         - 좌변과 우변 모두 숫자인 경우: 예) "1~2TC7" → ["1 TC7", "2 TC7"]
         - 좌변은 숫자이고 우변은 숫자가 아닌 경우:
             만약 우변이 "PIT"로 시작하면, 예) "1~PITC6" → ["1 C6", "PIT C6"],
             그렇지 않으면 단순히 [left, right]를 반환.
      3. 그 외에는 "( Base : " 기준 처리 및 콜론(:) 제거.
    
    반환 예시:
      "A101 ( Base : F3 )"   -> ["3 A101"]
      "A102 ( Base : B1 )"   -> ["-1 A102"]
      "[ 3 G13:5"           -> ["3 G13"]
      "1~2TC7"              -> ["1 TC7", "2 TC7"]
      "-3~-1TC5"            -> ["-3 TC5", "-2 TC5", "-1 TC5", "1 TC5"]
      "1~PITC6"             -> ["1 C6", "PIT C6"]
      "1~PITTC10"           -> ["1 TC10", "PIT TC10"]
    """
    # 1. 쉼표 처리
    if "," in member_str:
        parts = [p.strip() for p in member_str.split(",")]
        result = []
        for part in parts:
            result.extend(transform_rc_column_format(part))
        return result

    # 2. tilde(~) 패턴 처리
    if "~" in member_str:
        # 먼저, 좌변과 우변 모두 숫자인 경우
        m = re.match(r"^([+-]?\d+)\s*~\s*([+-]?\d+)(.*)$", member_str)
        if m:
            start_floor = int(m.group(1))
            end_floor = int(m.group(2))
            remainder = m.group(3).strip()
            # 이미 remainder에 콜론 제거 적용
            remainder = remove_colon_suffix(remainder)
            if start_floor <= end_floor:
                floors = list(range(start_floor, end_floor + 1))
            else:
                floors = list(range(start_floor, end_floor - 1, -1))
            # 0층은 제거
            floors = [fl for fl in floors if fl != 0]
            return [f"{fl} {remainder}".strip() for fl in floors]
        else:
            # 좌변은 숫자인데 우변이 숫자가 아닌 경우: 예) "1~PITC6" 또는 "1~PITTC10"
            m2 = re.match(r"^([+-]?\d+)\s*~\s*(.+)$", member_str)
            if m2:
                left_val = m2.group(1).strip()
                right_val = m2.group(2).strip()
                # 만약 우변이 "PIT"로 시작하면 특수 처리
                if right_val.upper().startswith("PIT"):
                    remainder = right_val[3:].strip()  # "PIT" 제거
                    remainder = remove_colon_suffix(remainder)
                    return [f"{left_val} {remainder}".strip(), f"PIT {remainder}".strip()]
                else:
                    # 그 외는 단순 분리하여 반환 (콜론 제거 적용)
                    left_val = remove_colon_suffix(left_val)
                    right_val = remove_colon_suffix(right_val)
                    return [left_val, right_val]

    # 3. 기본 처리: "( Base : " 기준 처리
    if "( Base :" in member_str:
        try:
            base_info = member_str.split("( Base : ")[-1].split(" )")[0].strip()
            code = member_str.split(" ( Base : ")[0].strip()
        except IndexError:
            code = member_str.strip()
            base_info = ""
        # 코드와 base_info 모두에서 ":" 뒤 숫자 제거
        code = remove_colon_suffix(code)
        base_info = remove_colon_suffix(base_info)
        if base_info:
            if "F" in base_info:
                floor = base_info.replace("F", "").strip()
            elif "B" in base_info:
                floor = "-" + base_info.replace("B", "").strip()
            else:
                floor = ""
            return [f"{floor} {code}".strip()]
        else:
            return [code]
    else:
        # 기본적으로 콜론 제거 적용
        return [remove_colon_suffix(member_str)]

# CSV 파일 로드
file_path = r"D:\LLLast\SCD\merged_results.csv"  # 입력 파일 경로
data = pd.read_csv(file_path)

# "Member code" 열에 변환 함수 적용 (각 행 결과는 리스트)
data["Member code"] = data["Member code"].apply(transform_rc_column_format)

# 행 확장: 여러 값이 있는 경우 각 값을 별도의 행으로 생성 (다른 열은 그대로 유지)
expanded_rows = []
for _, row in data.iterrows():
    codes_list = row["Member code"]  # 이미 리스트임
    for code in codes_list:
        new_row = row.copy()
        new_row["Member code"] = code
        expanded_rows.append(new_row)
expanded_data = pd.DataFrame(expanded_rows)

# 중복 행 제거 (모든 열 기준)
expanded_data.drop_duplicates(keep='first', inplace=True)

# 결과 저장
output_path = r"D:\LLLast\SCD\transformed_results_last.csv"  # 결과 파일 저장 경로
expanded_data.to_csv(output_path, index=False, encoding="utf-8-sig")

print(f"변환, 행 확장 및 중복 제거 완료! 결과 파일이 '{output_path}'에 저장되었습니다.")
















