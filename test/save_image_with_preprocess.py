# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    save_image_with_preprocess.py                      :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: kyumin1227 <kyumin12271227@gmail.com>      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/11/15 18:39:19 by kyumin1227        #+#    #+#              #
#    Updated: 2024/12/16 02:39:56 by kyumin1227       ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import cv2
import numpy as np
import os

# 저장 경로 설정
output_dir = 'filtered_images'
os.makedirs(output_dir, exist_ok=True)

def apply_filter_and_save(image, count, target_size=(128, 128)):
    # 1. 크기 조정
    resized = cv2.resize(image, target_size)

    # 2. 블러 적용
    blurred = cv2.GaussianBlur(resized, (5, 5), 0)

    # 3. 흰색 필터 적용
    lower_white = np.array([0, 0, 200])  # 흰색 최소값
    upper_white = np.array([180, 55, 255])  # 흰색 최대값
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_white, upper_white)

    # 4. 정규화 (0-1 범위로 스케일링)
    normalized = mask / 255.0

    # 5. 저장 경로 설정
    output_path = os.path.join(output_dir, f'image_{count:04d}.jpg')

    # 6. 이미지 저장
    cv2.imwrite(output_path, (normalized * 255).astype(np.uint8))  # 원래 픽셀 값으로 복원
    print(f"Saved: {output_path}")

    return normalized  # 처리된 이미지를 반환

def capture_and_process_images(video_source=0):
    cap = cv2.VideoCapture(video_source)
    count = 0

    print("Press 's' to save the frame, 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read from camera.")
            break

        # 필터와 블러 적용
        filtered_frame = apply_filter_and_save(frame, count)

        # 필터 적용 결과 보여주기
        cv2.imshow("Filtered Frame", (filtered_frame * 255).astype(np.uint8))

        # 키 입력 대기
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):  # 's' 키를 누르면 이미지 저장
            apply_filter_and_save(frame, count)
            count += 1
        elif key == ord('q'):  # 'q' 키를 누르면 종료
            break

    cap.release()
    cv2.destroyAllWindows()

# 함수 호출
capture_and_process_images(0)  # 카메라 입력
