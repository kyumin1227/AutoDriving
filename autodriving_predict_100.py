import cv2
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import threading
import autodriving_motor_100 as motor
import queue
import os
import wandb

# CNN 모델 정의 (학습할 때 사용한 모델과 동일)
class SimpleCNN(torch.nn.Module):
    def __init__(self, num_classes=6):
        super(SimpleCNN, self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(32 * 15 * 15, 128),  # (32채널, 15x15 크기)
            torch.nn.ReLU(),
            torch.nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# 모델 로드
run = wandb.init()
artifact = run.use_artifact('kyumin1227-yeungjin-college/Autodriving/simple_cnn_speed_30_and_100_model:v0', type='model')
artifact_dir = artifact.download()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SimpleCNN(num_classes=6).to(device)
model.load_state_dict(torch.load(artifact_dir, map_location=device))
model.eval()  # 평가 모드로 전환

# 예측 값을 넘길 queue
data_queue = queue.Queue()
# 촬영 값을 넘길 queue
task_queue = queue.Queue()

recent_files = []

# 데이터 변환 설정 (학습 시 사용한 변환과 동일)
transform = transforms.Compose([
    transforms.Resize((60, 60)),  # 학습 시와 동일한 크기로 리사이즈
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 정규화
])

# 저장 디렉토리 설정
save_dir = "predict_error"
os.makedirs(save_dir, exist_ok=True)

# 파일 저장 인덱스
save_index = 0

# 실시간 예측 함수
def predict_frame(frame):
    # OpenCV 이미지를 PIL 이미지로 변환
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # 데이터 변환 및 모델 입력
    input_tensor = transform(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        predicted_class = torch.argmax(outputs, dim=1).item()
    return predicted_class

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

    return result, combined_mask

motor_thread = threading.Thread(target=motor.handle_motor, kwargs={"data_queue": data_queue})
motor_thread.start()

# OpenCV로 실시간 영상 처리
cap = cv2.VideoCapture(0)  # 0번 카메라
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
cap.set(cv2.CAP_PROP_FPS, 10)

if not cap.isOpened():
    print("Error: Unable to access the camera.")
    exit()

angle_list = (72, 87, 102, 117, 132, 102100)
base_dir = "predict_capture"
SPEED = 30

# 이미지 저장 작업 쓰레드
def capture_and_save_image_task():
    while True:
        try:
            # 큐에서 작업 가져오기
            task = task_queue.get()
            if task is None:  # None이면 종료 신호
                break

            mask_image, angle, count = task

            angle = angle_list[angle]

            # 디렉토리 구조: speed_{속도}/angle_{각도}/
            speed_dir = os.path.join(base_dir, f"speed_{SPEED}")
            angle_dir = os.path.join(speed_dir, f"angle_{angle}")
            os.makedirs(angle_dir, exist_ok=True)

            # 이미지 파일 저장
            filename = os.path.join(angle_dir, f"{count:04d}.jpg")
            cv2.imwrite(filename, mask_image)
            print(f"Captured: {filename}")

            # 최근 파일 경로 업데이트
            recent_files.append(filename)
            if len(recent_files) > 10:
                recent_files.pop(0)  # 최근 10개만 유지

        except Exception as e:
            print(f"Error saving image: {e}")

# 최근 파일 이동 함수
def move_recent_files_to_error():
    global recent_files
    if recent_files:
        print("Moving last 10 files to predict_error folder...")
        for file_path in reversed(recent_files):
            try:
                # 이동 대상 경로 생성
                base_name = os.path.basename(file_path)
                error_path = os.path.join(save_dir, base_name)
                
                # 파일 이동
                os.rename(file_path, error_path)
                print(f"Moved: {file_path} -> {error_path}")
            except FileNotFoundError:
                print(f"File not found: {file_path}")
        recent_files.clear()
    else:
        print("No files to move.")

# 쓰레드 시작
image_thread = threading.Thread(target=capture_and_save_image_task)
image_thread.start()

# 저장할 사진 번호
capture_count = 0

pause = True

print("Press 'q' to quit.")
while True:
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

    filtered_frame, mask = filter_white_yellow(resized)

    # 모델로 예측
    predicted_angle = predict_frame(filtered_frame)

    print("예측 값", predicted_angle, "값 전송", not pause)

    # pause 상태가 아니면 값 전송
    if not pause:
        data_queue.put(predicted_angle)

        # 저장할 값 queue에 전송
        task_queue.put((filtered_frame, predicted_angle, capture_count))
        capture_count += 1

    # 예측 결과 표시
    cv2.putText(
        roi,
        f"Predicted Angle: {predicted_angle}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Live Prediction", filtered_frame)
    cv2.imshow("Original", roi)

    key = cv2.waitKey(1) & 0xFF

    # space 누르면 정지 상태 변경
    if key == 32: # space
        pause = not pause
    # 위쪽 화살표
    elif key == 82:
        print("위쪽 화살표")
        data_queue.put("up")
    # 아래쪽 화살표
    elif key == 84:
        print("아래쪽 화살표")
        data_queue.put("down")
    # s 키를 눌러 filtered_frame 저장
    elif key == ord('s'):
        save_path = os.path.join(save_dir, f"filtered_frame_{save_index:04d}.jpg")
        cv2.imwrite(save_path, filtered_frame)
        print(f"Saved filtered frame to {save_path}")
        save_index += 1
    elif key == 8:
        move_recent_files_to_error()  # Backspace 키로 최근 파일 이동
    # 'q' 키를 누르면 종료
    elif key == ord('q'):
        data_queue.put("stop")
        break


cap.release()
cv2.destroyAllWindows()
motor_thread.join()
image_thread.join()
