import os
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import wandb

# WandB 프로젝트 초기화
run = wandb.init(project="Autodriving", name="simple_cnn_speed_30_and_100_training")

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
        label_mapping = {72: 0, 87: 1, 102: 2, 117: 3, 132: 4, 102_100: 5}

        for angle_folder in sorted(os.listdir(root_dir)):
            angle_path = os.path.join(root_dir, angle_folder)
            if not os.path.isdir(angle_path):
                continue

            label = int(angle_folder.split('_')[1])
            
             # 라벨을 0~4로 매핑
            if label in label_mapping:
                mapped_label = label_mapping[label]
            else:
                continue

            for image_name in os.listdir(angle_path):
                image_path = os.path.join(angle_path, image_name)
                
                if not image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                
                self.data.append(image_path)
                self.labels.append(mapped_label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.data[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# 하이퍼파라미터
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
EPOCHS = 30

# 데이터 변환 및 로더
transform = transforms.Compose([
    transforms.Resize((60, 60)),  # 이미지 크기 조정
    transforms.ToTensor(),  # 텐서로 변환
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 정규화 (R, G, B 채널 각각)
])

# WandB artifact로 데이터셋 가져오기
artifact = run.use_artifact('kyumin1227-yeungjin-college/Autodriving/speed30_and_100_60x60:v1', type='dataset')
dataset_dir = artifact.download()

root_dir = os.path.join(dataset_dir, "speed_30")
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
model = SimpleCNN(num_classes=6).to(device)
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

            outputs = model(images)
            loss = criterion(outputs, labels)

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

        # 결과 출력 및 로깅
        train_loss_avg = train_loss / len(train_loader)
        train_acc = correct / len(train_dataset)
        val_loss_avg = val_loss / len(val_loader)
        val_acc = val_correct / len(val_dataset)

        print(f"Epoch {epoch + 1}/{EPOCHS}")
        print(f"Train Loss: {train_loss_avg:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss_avg:.4f}, Val Acc: {val_acc:.4f}")

        # WandB에 로깅
        run.log({
            "epoch": epoch + 1,
            "train_loss": train_loss_avg,
            "train_accuracy": train_acc,
            "val_loss": val_loss_avg,
            "val_accuracy": val_acc
        })

# 학습 실행
train_and_evaluate()

# 모델 저장 및 업로드
model_path = "simple_cnn_speed_30_and_100_model.pth"
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")

# WandB artifact로 모델 업로드
artifact = wandb.Artifact("simple_cnn_speed_30_and_100_model", type="model")
artifact.add_file(model_path)
run.log_artifact(artifact)

# WandB 종료
run.finish()
