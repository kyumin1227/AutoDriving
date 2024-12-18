# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    motor_test.py                                      :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: kyumin1227 <kyumin12271227@gmail.com>      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/11/15 12:14:46 by kyumin1227        #+#    #+#              #
#    Updated: 2024/12/16 02:39:55 by kyumin1227       ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

# 서보 모터와 DC 모터 테스트 코드
# サーボモーターとDCモーターのテストコード
 
import Jetson.GPIO as GPIO
import time
import tkinter as tk

# PWM 및 모터 제어 핀 설정
SERVO_PIN = 32  # 서보 모터 PWM 핀
DC_PIN = 33     # DC 모터 PWM 핀

# GPIO 핀 설정 (Jetson Nano의 GPIO 핀 번호)
IN1_PIN = 13  # 모터 1, IN1
IN2_PIN = 15  # 모터 1, IN2
IN3_PIN = 16  # 모터 2, IN3
IN4_PIN = 18  # 모터 2, IN4

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
current_speed = 70  # 초기 속도 (70%)
speed_step = 30  # 속도 증가/감소 단위

# 현재 각도 변수
current_angle = 100 # 초기 각도 (100)
angle_step = 15 # 각도 증가/감소 단위

# 서보 모터 각도 조정 함수
def set_servo_angle(angle):
    if angle > 130:
        print("Max angle reached")
        return
    if angle < 70:
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

# 속도 증가 함수 (F 키)
def increase_speed():
    global current_speed
    if current_speed + speed_step <= 100:
        current_speed += speed_step
        print("Increased speed to:", current_speed)
        # 현재 전진 또는 후진 상태의 경우 속도 업데이트
        if key_pressed['w'] or key_pressed['s']:
            pwm_dc.ChangeDutyCycle(current_speed)
    else:
        print("Max speed reached")

# 속도 감소 함수 (G 키)
def decrease_speed():
    global current_speed
    if current_speed - speed_step >= 40:
        current_speed -= speed_step
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
        set_servo_angle(current_angle - angle_step)
    elif event.keysym == 'd':
        set_servo_angle(current_angle + angle_step)
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
    elif key_pressed['s']:
        move_backward()
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
set_servo_angle(100)
pwm_dc.stop()
GPIO.cleanup()
