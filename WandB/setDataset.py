import wandb

# WandB 프로젝트 초기화
wandb.init(project="Autodriving")

# Artifact 생성
artifact = wandb.Artifact(name="dataset_speed40_60x60", type="dataset", description="전처리 및 인식 실패 데이터 추가")

# 데이터셋 경로 지정
artifact.add_dir("/Users/kyumin/AI/자율주행 자동차/captured_data_2 2")

# Artifact 업로드
wandb.log_artifact(artifact)
