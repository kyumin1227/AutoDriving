import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image

# 데이터셋 정의
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): 데이터가 저장된 루트 디렉토리 경로 (speed_100)
            transform (callable, optional): 데이터 변환 적용
        """
        self.root_dir = root_dir
        self.transform = transform
        self.data = []
        self.labels = []
        
         # 라벨 매핑 정의 (70 -> 0, 85 -> 1, ...)
        label_mapping = {70: 0, 85: 1, 100: 2, 115: 3, 130: 4}

        # 각 angle 폴더를 탐색
        for angle_folder in sorted(os.listdir(root_dir)):
            angle_path = os.path.join(root_dir, angle_folder)
            if not os.path.isdir(angle_path):
                continue

            # angle_xx 폴더의 이름에서 라벨 추출 (예: angle_70 -> 70)
            label = int(angle_folder.split('_')[1])
            
             # 라벨을 0~4로 매핑
            if label in label_mapping:
                mapped_label = label_mapping[label]
            else:
                continue  # 매핑되지 않은 라벨은 무시

            for image_name in os.listdir(angle_path):
                image_path = os.path.join(angle_path, image_name)
                
                # 이미지 파일만 선택 (확장자가 .jpg, .jpeg, .png 등)
                if not image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                
                self.data.append(image_path)
                self.labels.append(mapped_label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.data[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert("RGB")  # 컬러 이미지로 변환
        if self.transform:
            image = self.transform(image)
        return image, label

# 하이퍼파라미터
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = 50

# 데이터 변환 및 로더
transform = transforms.Compose([
    transforms.Resize((60, 60)),  # 이미지 크기 조정
    transforms.ToTensor(),  # 텐서로 변환
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 정규화 (R, G, B 채널 각각)
])

root_dir = r"C:\autodriving\captured_data_2\speed_100"
dataset = CustomDataset(root_dir=root_dir, transform=transform)
train_size = int(0.7 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# CNN 모델 정의
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),  # 3채널 입력
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 15 * 15, 128),  # (32채널, 15x15 크기)
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# 모델, 손실 함수, 옵티마이저 초기화
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN(num_classes=5).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 학습 및 평가 루프
def train_and_evaluate():
    for epoch in range(EPOCHS):
        # Training
        model.train()
        train_loss = 0.0
        correct = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                val_correct += (outputs.argmax(1) == labels).sum().item()

        # 결과 출력
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        print(f"Train Loss: {train_loss / len(train_loader):.4f}, Train Acc: {correct / len(train_dataset):.4f}")
        print(f"Val Loss: {val_loss / len(val_loader):.4f}, Val Acc: {val_correct / len(val_dataset):.4f}")

# 학습 실행
train_and_evaluate()

# 모델 저장
torch.save(model.state_dict(), "simple_cnn_speed_100_rgb.pth")
print("Model saved to simple_cnn_speed_100_rgb.pth")