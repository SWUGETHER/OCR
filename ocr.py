import easyocr
import cv2
from sklearn.cluster import DBSCAN
import numpy as np
import math
import pandas as pd

# 텍스트 일치 여부 검사
def checkLabel(text, expect):
  text = text.replace(" ", "")
  detected = []
  i = 0

  for c_e in expect:
    for c in text[i:]:
      if c_e == c:
        detected.append(c)
        i += 1
        break

  if len(detected) >= len(expect) - 1:
    return True

  return False

# Bbox 간 거리 계산
def distance(b1, b2):
  # Bbox의 시작점, 종료점
  b1_start = (float(b1[0]), float(b1[1]))
  b1_end = (float(b1[2]), float(b1[3]))
  b2_start = (float(b2[0]), float(b2[1]))
  b2_end = (float(b2[2]), float(b2[3]))

  # 중심점 계산
  center1 = np.array([(b1_start[0] + b1_end[0]) / 2, (b1_start[1] + b1_end[1]) / 2])
  center2 = np.array([(b2_start[0] + b2_end[0]) / 2, (b2_start[1] + b2_end[1]) / 2])

  b1_center = (b1_start[0] + b1_end[0] // 2, b1_start[1] + b1_end[1] // 2)
  b2_center = (b2_start[0] + b2_end[0] // 2, b2_start[1] + b2_end[1] // 2)

  # 거리 계산
  d = math.sqrt((b1_center[0] - b2_center[0]) ** 2 + (b1_center[1] - b2_center[1]) ** 2)

  # 두 직사각형의 중심점 간의 유클리디안 거리 계산
  distance = np.linalg.norm(center1 - center2)

  return distance

# Bbox 거리 행렬 계산
def bbox_d_matrix(bboxes):
  num_bboxes = len(bboxes)
  d_matrix = np.zeros((num_bboxes, num_bboxes)) # 초기화

  # 모든 bbox 쌍에 대한 거리 계산 (비대각 요소만)
  for i in range(num_bboxes):
    for j in range(i+1, num_bboxes):
      d_matrix[i][j] = distance(bboxes[i], bboxes[j])
      d_matrix[j][i] = d_matrix[i][j] # 거리 행렬은 대칭적

  return d_matrix


# DBSCAN (밀도 기반 클러스터링)
def bboxCluster_DBSCAN(ocr_result, target_i, eps_per):
  # bbox
  bboxes = np.array([(detection[0][0][0], detection[0][0][1], detection[0][2][0], detection[0][2][1]) for detection in ocr_result])

  # target bbox
  target = bboxes[target_i]

  # 거리 행렬을 이용하여 eps 탐색
  # 1. 거리 행렬
  d_matrix = bbox_d_matrix(bboxes)

  # 2. 행렬 평탄화 및 정렬
  sorted_d = np.sort(d_matrix.flatten())

  # 3. 거리 행렬 중 하위 20% 해당하는 거리를 eps로 설정
  percentile_i = int(np.percentile(range(len(sorted_d)), eps_per))
  eps = sorted_d[percentile_i]

  # DBSCAN clustering
  dbscan = DBSCAN(eps=eps, min_samples=1, metric='precomputed')
  cluster_labels = dbscan.fit_predict(d_matrix)

  # 특정 bbox와 같은 클러스터에 속한 가까운 bbox들 찾기
  target_cluster_label = cluster_labels[target_i]
  nearby_bboxes = []
  nearby_bboxes_i = []

  for i, bbox in enumerate(bboxes):
    if cluster_labels[i] == target_cluster_label and i != target_i:
      nearby_bboxes_i.append(i)
      nearby_bboxes.append((ocr_result[i][1]))
    elif cluster_labels[i] == target_cluster_label and i == target_i:
      nearby_bboxes_i.append(i)
      nearby_bboxes.append((ocr_result[i][1]))

  return nearby_bboxes

def extract_data(result):
  name_found, comp_found = False, False
  name, comp = "", ""
  
  for i in range(len(result)):
    text = result[i][1]

    if checkLabel(text, "전성분"):
        comp_bboxes = bboxCluster_DBSCAN(result, i+1, 20)
        comp_found = True
    elif checkLabel(text, "명칭"):
        name_bboxes = bboxCluster_DBSCAN(result, i+1, 10)
        name_found = True
    elif checkLabel(text, "제품명"):
        name_bboxes = bboxCluster_DBSCAN(result, i+1, 10)
        name_found = True

    if comp_found and name_found:
        name = ' '.join(str(t) for t in name_bboxes)
        comp = ' '.join(str(t) for t in comp_bboxes)
        break

  return name, comp

def read_image(image_path):
    # EasyOCR 객체 생성
    reader = easyocr.Reader(['ko', 'en'], gpu=True)

    # 이미지 파일 OpenCV로 불러오기
    image = cv2.imread(image_path)

    # 이미지 파일 OR URL로부터 텍스트 추출
    result = reader.readtext(image)

    # 추출 결과 출력
    df = pd.DataFrame(data = result, columns=['Bbox', 'Text', 'Confidence'])

    name, comp = extract_data(result)
    
    return name, comp