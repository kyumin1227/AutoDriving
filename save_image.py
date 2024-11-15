# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    save_image.py                                      :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: kyumin1227 <kyumin12271227@gmail.com>      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/11/15 18:39:15 by kyumin1227        #+#    #+#              #
#    Updated: 2024/11/15 18:40:52 by kyumin1227       ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import cv2
import os

# 저장 경로 설정
output_dir = 'resized_images'
os.makedirs(output_dir, exist_ok=True)

def resize_and_save(image, count, target_size=(128, 128)):
    # 1. 크기 조정
    resized = cv2.resize(image, target_size)

    # 2. 저장 경로 설정
    output_path = os.path.join(output_dir, f'image_{count:04d}.jpg')

    # 3. 이미지 저장
    cv2.imwrite(output_path, resized)
    print(f"Saved: {output_path}")

    return resized

def capture_and_process_images(video_source=0):
    cap = cv2.VideoCapture(video_source)
    count = 0

    print("Press 's' to save the frame, 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read from camera.")
            break

        # 128x128으로 크기 조정
        resized_frame = cv2.resize(frame, (128, 128))

        # 조정된 프레임을 보여주기
        cv2.imshow("Resized Frame", resized_frame)

        # 키 입력 대기
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):  # 's' 키를 누르면 이미지 저장
            resize_and_save(frame, count)
            count += 1
        elif key == ord('q'):  # 'q' 키를 누르면 종료
            break

    cap.release()
    cv2.destroyAllWindows()

# 함수 호출
capture_and_process_images(0)  # 카메라 입력
