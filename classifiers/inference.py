# this file runs inference / generates predictions using little owl models

import torch
import yaml
import os
from pathlib import Path

from kaytoo_small import BirdSoundModel, TrainingParameters, DefaultAudio, FilePaths, real_life_evaluate_with_roc, test_inference

audio_cfg = DefaultAudio()
# Load use case
with open('classifiers/use_case.yaml') as f:
    use_case = yaml.safe_load(f)
train_cfg = TrainingParameters(options=use_case)
paths = FilePaths(use_case)

model_path = 'classifiers/evaluation_results/Exp_2443/Results/binary_classifier-epoch=05-rl_auc=0.898.ckpt'
bird_model = BirdSoundModel(train_cfg, audio_cfg, paths, in_channels=3)
model_state_dict = torch.load(model_path)
bird_model.load_state_dict(model_state_dict)

# here we run inference on all the little owl sites
data_path = '/Volumes/Rectangle/little_owl/data-backup/Little Owl AudioMoth'
save_dir = 'classifiers/inference_results/Exp_2443_e5'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
# get directories in data_path
dirs = [f for f in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, f))]
for dir_path in dirs:
    print(f'Processing {dir_path}')
    save_to = os.path.join(save_dir, dir_path)
    dir_path = Path(dir_path)
    save_to = Path(save_to)
    test_inference(bird_model, audio_cfg, os.path.join(data_path, dir_path), save_to=save_to)
    print(f'Finished {dir_path}')

print('fone')