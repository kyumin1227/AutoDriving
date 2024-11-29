# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    motor_and_image.py                                 :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: kyumin1227 <kyumin12271227@gmail.com>      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/11/21 22:14:46 by kyumin1227        #+#    #+#              #
#    Updated: 2024/11/29 15:48:37 by kyumin1227       ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

# 서보 모터와 DC 모터 테스트 코드
# サーボモーターとDCモーターのテストコード
 
import Jetson.GPIO as GPIO
import time
import tkinter as tk
import cv2
import numpy as np
import os
import threading
import queue

# PWM 및 모터 제어 핀 설정
SERVO_PIN = 32  # 서보 모터 PWM 핀
DC_PIN = 33     # DC 모터 PWM 핀

# GPIO 핀 설정 (Jetson Nano의 GPIO 핀 번호)
IN1_PIN = 13  # 모터 1, IN1
IN2_PIN = 15  # 모터 1, IN2
IN3_PIN = 16  # 모터 2, IN3
IN4_PIN = 18  # 모터 2, IN4

DEFAULT_ANGLE = 100 # 중립 각도
DEFAULT_SPEED = 30  # 초기 속도
ANGLE_STEP = 15 # 각도 증가/감소 단위
SPEED_STEP = 30  # 속도 증가/감소 단위

# GPIO 설정
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD)
GPIO.setup(SERVO_PIN, GPIO.OUT)
GPIO.setup(DC_PIN, GPIO.OUT)
GPIO.setup(IN1_PIN, GPIO.OUT)
GPIO.setup(IN2_PIN, GPIO.OUT)
GPIO.setup(IN3_PIN, GPIO.OUT)
GPIO.setup(IN4_PIN, GPIO.OUT)

# 서보 모터 50Hz, PWM 초기화
pwm_servo = GPIO.PWM(SERVO_PIN, 50)
pwm_servo.start(7.5)  # 초기 중립 각도 (100도)

# DC 모터 PWM 100Hz 설정 (속도 제어를 위한 주파수)
pwm_dc = GPIO.PWM(DC_PIN, 100)
pwm_dc.start(0)  # 초기 속도 0%

# 현재 속도 변수
current_speed = DEFAULT_SPEED

# 현재 각도 변수
current_angle = DEFAULT_ANGLE

# 라벨링 데이터를 저장할 디렉토리 설정
base_dir = 'captured_data'
os.makedirs(base_dir, exist_ok=True)

# 카메라 설정
cap = cv2.VideoCapture(0)  # 0번 카메라
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 버퍼 크기를 1로 설정
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)  # 해상도 너비
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)  # 해상도 높이
cap.set(cv2.CAP_PROP_FPS, 30)  # FPS 조정

# 작업 큐 생성
task_queue = queue.Queue()

# 서보 모터 각도 조정 함수
def set_servo_angle(angle):
    if angle > DEFAULT_ANGLE + ANGLE_STEP * 2:
        print("Max angle reached")
        return
    if angle < DEFAULT_ANGLE - ANGLE_STEP * 2:
        print("Min angle reached")
        return
    global current_angle
    print(f"각도 조절 {angle}")
    duty = 3.0 + (angle / 180.0) * 9.5  # 각도에 맞는 듀티 사이클 계산
    pwm_servo.ChangeDutyCycle(duty)
    current_angle = angle
    time.sleep(0.1)  # 서보 모터가 움직이는 시간 대기

# DC 모터 전진 설정 (전진 시 방향 설정)
def move_forward():
    print(f"Moving forward with speed: {current_speed} angle: {current_angle}")
    GPIO.output(IN1_PIN, GPIO.HIGH)  # 모터 1 전진
    GPIO.output(IN2_PIN, GPIO.LOW)
    GPIO.output(IN3_PIN, GPIO.HIGH)  # 모터 2 전진
    GPIO.output(IN4_PIN, GPIO.LOW)
    pwm_dc.ChangeDutyCycle(current_speed)  # 속도 제어

# DC 모터 후진 설정 (후진 시 방향 설정)
def move_backward():
    print(f"Moving backward with speed: {current_speed} angle: {current_angle}")
    GPIO.output(IN1_PIN, GPIO.LOW)   # 모터 1 후진
    GPIO.output(IN2_PIN, GPIO.HIGH)
    GPIO.output(IN3_PIN, GPIO.LOW)   # 모터 2 후진
    GPIO.output(IN4_PIN, GPIO.HIGH)
    pwm_dc.ChangeDutyCycle(current_speed)  # 속도 제어

# DC 모터 정지 함수 (모터 정지)
def stop_motors():
    print("Stopping motors")
    GPIO.output(IN1_PIN, GPIO.LOW)
    GPIO.output(IN2_PIN, GPIO.LOW)
    GPIO.output(IN3_PIN, GPIO.LOW)
    GPIO.output(IN4_PIN, GPIO.LOW)
    pwm_dc.ChangeDutyCycle(0)  # 속도를 0으로 설정 (정지)

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

    # 흰색과 노란색을 합친 마스크
    combined_mask = cv2.bitwise_or(white_mask, yellow_mask)

    # 원본 이미지에서 흰색과 노란색만 강조
    result = cv2.bitwise_and(image, image, mask=combined_mask)

    return result

# 이미지 저장 작업 쓰레드
def capture_and_save_image_task():
    while True:
        try:
            # 큐에서 작업 가져오기
            task = task_queue.get()
            if task is None:  # None이면 종료 신호
                break

            speed, angle, count = task

            cap.grab()
            ret, frame = cap.read()
            if ret:
                
                # 이미지 크기
                h, w = frame.shape[:2]

                # 아래에서 3분의 2 영역 선택
                y1 = h // 3  # 높이의 1/3 지점
                y2 = h       # 전체 높이
                roi = frame[y1:y2, :]  # ROI 설정 (너비 전체 사용)

                # 60 x 60 사이즈로 리사이징
                resized = cv2.resize(roi, (60, 60))

                mask = filter_white_yellow(resized)

                # 디렉토리 구조: speed_{속도}/angle_{각도}/
                speed_dir = os.path.join(base_dir, f"speed_{speed}")
                angle_dir = os.path.join(speed_dir, f"angle_{angle}")
                os.makedirs(angle_dir, exist_ok=True)

                # 이미지 파일 저장
                filename = os.path.join(angle_dir, f"{count:04d}.jpg")
                cv2.imwrite(filename, mask)
                print(f"Captured: {filename}")

        except Exception as e:
            print(f"Error saving image: {e}")

# 쓰레드 시작
image_thread = threading.Thread(target=capture_and_save_image_task)
image_thread.start()

# 캡처 요청 함수
capture_count = 0
def request_capture():
    global capture_count
    
    # 작업 큐에 요청 추가
    task_queue.put((current_speed, current_angle, capture_count))
    capture_count += 1

# 속도 증가 함수 (F 키)
def increase_speed():
    global current_speed
    if current_speed + SPEED_STEP <= 100:
        current_speed += SPEED_STEP
        print("Increased speed to:", current_speed)
        # 현재 전진 또는 후진 상태의 경우 속도 업데이트
        if key_pressed['w'] or key_pressed['s']:
            pwm_dc.ChangeDutyCycle(current_speed)
    else:
        print("Max speed reached")

# 속도 감소 함수 (G 키)
def decrease_speed():
    global current_speed
    if current_speed - SPEED_STEP >= SPEED_STEP:
        current_speed -= SPEED_STEP
        print("Decreased speed to:", current_speed)
        # 현재 전진 또는 후진 상태의 경우 속도 업데이트
        if key_pressed['w'] or key_pressed['s']:
            pwm_dc.ChangeDutyCycle(current_speed)
    else:
        print("Min speed reached")

# 키 상태 관리 (W, S 키)
key_pressed = {
    'w': False,
    's': False
}

def on_key_press(event):
    if event.keysym == 'a':
        set_servo_angle(current_angle - ANGLE_STEP)
    elif event.keysym == 'd':
        set_servo_angle(current_angle + ANGLE_STEP)
    if event.keysym == 'w':
        key_pressed['w'] = True
    elif event.keysym == 's':
        key_pressed['s'] = True
    if event.keysym == 'f':
        increase_speed()  # F 키로 속도 증가
    elif event.keysym == 'g':
        decrease_speed()  # G 키로 속도 감소
    if event.keysym == "Escape":
        root.quit()

def on_key_release(event):
    if event.keysym == 'w':
        key_pressed['w'] = False
    if event.keysym == 's':
        key_pressed['s'] = False

def check_keys(): 
    if key_pressed['w']:
        move_forward() 
        request_capture()  # 이미지 저장 요청
    elif key_pressed['s']:
        move_backward()
        request_capture()  # 이미지 저장 요청
    else:
        stop_motors()

    # 100ms마다 키 상태 체크
    root.after(100, check_keys)

# tkinter GUI 설정
root = tk.Tk()

# 키보드 이벤트 바인딩
root.bind('<KeyPress>', on_key_press)
root.bind('<KeyRelease>', on_key_release)

# 키 상태 체크
check_keys()

# Tkinter 메인 루프
root.mainloop()

# 프로그램 종료 시 GPIO 정리
set_servo_angle(DEFAULT_ANGLE)
pwm_dc.stop()
GPIO.cleanup()
cap.release()
cv2.destroyAllWindows()

# 쓰레드 종료 신호 보내기
task_queue.put(None)
image_thread.join()