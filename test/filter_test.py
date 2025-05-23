# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    filter_test.py                                     :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: kyumin1227 <kyumin12271227@gmail.com>      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/11/24 14:26:25 by kyumin1227        #+#    #+#              #
#    Updated: 2024/12/14 18:04:25 by kyumin1227       ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import cv2
import numpy as np
import time
import os

# 저장 디렉토리 설정
save_dir = "ppt"
os.makedirs(save_dir, exist_ok=True)

def filter_white_yellow(image):
    # 이미지를 HSV 색 공간으로 변환
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 흰색 범위 설정
    lower_white = np.array([0, 0, 200])   # 낮은 경계 (H, S, V)
    upper_white = np.array([180, 25, 255])  # 높은 경계
    white_mask = cv2.inRange(hsv, lower_white, upper_white)

    # 노란색 범위 설정 (더 짙고 어두운 노란색 포함)
    lower_yellow = np.array([10, 30, 30])  # 낮은 경계 (H, S, V)
    upper_yellow = np.array([40, 255, 255])  # 높은 경계 (H, S, V)
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # 검은색 범위 설정
    lower_black = np.array([0, 0, 50])
    upper_black = np.array([180, 255, 50])  # 낮은 밝기 범위
    black_mask = cv2.inRange(hsv, lower_black, upper_black)

    # 흰색과 노란색을 합친 마스크
    combined_mask = cv2.bitwise_or(white_mask, yellow_mask)
    combined_mask = cv2.bitwise_or(combined_mask, black_mask)

    # 원본 이미지에서 흰색과 노란색만 강조
    result = cv2.bitwise_and(image, image, mask=combined_mask)

    # 노란색 영역만 강조
    yellow_highlight = cv2.bitwise_and(image, image, mask=yellow_mask)

    return result, white_mask, yellow_mask, combined_mask, yellow_highlight

# 실시간 영상 처리
cap = cv2.VideoCapture(0)  # 0번 카메라 연결

# 해상도 설정
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 버퍼 크기를 1로 설정
cap.set(cv2.CAP_PROP_FPS, 10)  # FPS 조정

# FPS 제한 설정
fps_limit = 10  # 초당 프레임 10개
frame_delay = 1 / fps_limit  # 각 프레임 사이의 시간 간격

if not cap.isOpened():
    print("Error: Unable to access the camera.")
    exit()

print("Press 'q' to quit.")
print("Press 's' to save all frames.")
last_time = time.time()
save_index = 0

while True:
    current_time = time.time()
    elapsed_time = current_time - last_time

    # 프레임 속도 제한
    if elapsed_time < frame_delay:
        continue

    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    # 이미지 크기
    h, w = frame.shape[:2]

    # 아래에서 3분의 2 영역 선택
    y1 = h // 3  # 높이의 1/3 지점
    y2 = h       # 전체 높이
    roi = frame[y1:y2, :]  # ROI 설정 (너비 전체 사용)

    # 60 x 60 사이즈로 리사이징
    resized = cv2.resize(roi, (60, 60))

    # 흰색과 노란색 필터 적용
    filtered_frame, white_mask, yellow_mask, combined_mask, yellow_highlight = filter_white_yellow(resized)

    # 결과 표시
    cv2.imshow("Original Frame", frame)
    cv2.imshow("Filtered Frame (Bottom 2/3)", filtered_frame)
    cv2.imshow("Mask (Bottom 2/3)", combined_mask)
    cv2.imshow("Yellow Highlight", yellow_highlight)

    key = cv2.waitKey(1) & 0xFF

    # 's' 키를 누르면 모든 이미지를 저장
    if key == ord('s'):
        cv2.imwrite(os.path.join(save_dir, f"original_{save_index:04d}.jpg"), frame)
        cv2.imwrite(os.path.join(save_dir, f"roi_{save_index:04d}.jpg"), roi)
        cv2.imwrite(os.path.join(save_dir, f"resized_{save_index:04d}.jpg"), resized)
        cv2.imwrite(os.path.join(save_dir, f"filtered_frame_{save_index:04d}.jpg"), filtered_frame)
        cv2.imwrite(os.path.join(save_dir, f"white_mask_{save_index:04d}.jpg"), white_mask)
        cv2.imwrite(os.path.join(save_dir, f"yellow_mask_{save_index:04d}.jpg"), yellow_mask)
        cv2.imwrite(os.path.join(save_dir, f"combined_mask_{save_index:04d}.jpg"), combined_mask)
        cv2.imwrite(os.path.join(save_dir, f"yellow_highlight_{save_index:04d}.jpg"), yellow_highlight)
        print(f"Saved all frames with index {save_index}")
        save_index += 1

    # 'q' 키를 누르면 종료
    if key == ord('q'):
        break

    # 마지막 프레임 처리 시간을 갱신
    last_time = current_time

# 리소스 해제
cap.release()
cv2.destroyAllWindows()
