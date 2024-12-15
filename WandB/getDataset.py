import wandb
run = wandb.init()
artifact = run.use_artifact('kyumin1227-yeungjin-college/Autodriving/speed30_and_100_60x60:v2', type='dataset')
artifact_dir = artifact.download()