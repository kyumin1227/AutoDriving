import wandb

# WandB 프로젝트 초기화
wandb.init(project="Autodriving")

# Artifact 생성
artifact = wandb.Artifact(name="speed30_and_100_60x60", type="dataset", description="데이터 전처리 (102도 기준 15도)")

# 데이터셋 경로 지정
artifact.add_dir("/Users/kyumin/AI/자율주행 자동차/captured_data")

# Artifact 업로드
wandb.log_artifact(artifact)
