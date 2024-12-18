# 자율 주행 모터 쓰레드

import Jetson.GPIO as GPIO
import time
import queue

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

DEFAULT_ANGLE = 102
ANGLE_STEP = 15

# 서보 모터 50Hz, PWM 초기화
pwm_servo = GPIO.PWM(SERVO_PIN, 50)
pwm_servo.start(7.5)  # 초기 중립 각도 (100도)

# DC 모터 PWM 100Hz 설정 (속도 제어를 위한 주파수)
pwm_dc = GPIO.PWM(DC_PIN, 100)
pwm_dc.start(0)  # 초기 속도 0%

# 현재 속도 변수
current_speed = 45  # 초기 속도 (45%)
speed_step = 1  # 속도 증가/감소 단위

# 현재 각도 변수
current_angle = DEFAULT_ANGLE # 초기 각도 (100)

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
    # pwm_dc.ChangeDutyCycle(current_speed)  # 속도 제어

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
        pwm_dc.ChangeDutyCycle(current_speed)
    else:
        print("Max speed reached")

# 속도 감소 함수 (G 키)
def decrease_speed():
    global current_speed
    if current_speed - speed_step >= 0:
        current_speed -= speed_step
        print("Decreased speed to:", current_speed)
        # 현재 전진 또는 후진 상태의 경우 속도 업데이트
        pwm_dc.ChangeDutyCycle(current_speed)
    else:
        print("Min speed reached")

def set_speed(speed):
    pwm_dc.ChangeDutyCycle(speed)
    # print("속도 변경: ", current_speed)

def handle_motor(data_queue):
    angle_mapping = {0: DEFAULT_ANGLE - ANGLE_STEP * 2, 1: DEFAULT_ANGLE - ANGLE_STEP, 2: DEFAULT_ANGLE, 3: DEFAULT_ANGLE + ANGLE_STEP, 4: DEFAULT_ANGLE + ANGLE_STEP * 2, 5: None}

    prev_angle = None

    while True:
        try:
            # 큐에서 최신 값만 가져오기
            while not data_queue.empty():
                angle_key = data_queue.get_nowait()  # 큐에서 값을 가져오되, 비우기
                
            angle_key = data_queue.get(timeout = 0.5)

            print("받은 값", angle_key)

            # 이전과 같으면 패스
            if angle_key == prev_angle:
                continue
            
            if angle_key in angle_mapping:
                prev_angle = angle_key
                if angle_key == 5:
                    set_servo_angle(DEFAULT_ANGLE)
                    set_speed(100)
                else:
                    angle = angle_mapping[angle_key]
                    set_servo_angle(angle)
                    set_speed(current_speed)

                move_forward()

            elif angle_key == "up":
                increase_speed()

            elif angle_key == "down":
                decrease_speed()

            # 프로그램 종료
            else:
                set_servo_angle(100)
                pwm_dc.stop()
                GPIO.cleanup()
        
        except queue.Empty:
            print("queue가 비어있음")
            stop_motors()
