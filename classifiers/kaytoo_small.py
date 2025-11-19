# This Python 3 environment comes with many helpful analytics libraries installed
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

#General Python
import gc
import os
from pathlib import Path
import shutil # Add this import
# from tqdm.notebook import tqdm
from tqdm import tqdm
import ast
from ast import literal_eval
from functools import reduce
import yaml
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message='A new version')
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve

#Math & Plotting
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import plotly.express as px
import csv
import datetime

#Machine Learning 
import albumentations as A
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn import metrics as skm
import cv2

#Torch and PyTorch specific
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint,  EarlyStopping
from torch.utils.data import  DataLoader, Dataset, WeightedRandomSampler
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torchaudio.functional import compute_deltas

# MPS config
if torch.backends.mps.is_available():
    device = torch.device("mps")
    # Initialize MPS device
    _ = torch.zeros(1).to(device)
    # Set default tensor type to float32
    torch.set_default_dtype(torch.float32)
    # configure memory
    torch.mps.set_per_process_memory_fraction(0.8)

#Audio
import librosa
import torchaudio
import colorednoise as cn


class FilePaths:
    def __init__(self, options=None):
        self.PROJECT_DIR = Path(options['project_root'])
        self.DATA_DIR = self.PROJECT_DIR / 'augmented_dataset' 
        self.LABELS_PATH = str(self.DATA_DIR / 'labels.csv')
        self.TRAIN_AUDIO_DIR = str(self.DATA_DIR / 'sound_files')
        # self.BACKGROUND_NOISE_FLDR =  str(self.DATA_DIR / 'background_noise')
        
        self.EVAL_DIR = self.PROJECT_DIR / 'L_O_eval'

        _experiment = options['experiment']
        if options.get('experiment_dir', None):
            experiments_path = options['experiment_dir']
        else:
            experiments_path = 'Experiments'
        print(f'Experiments path is {experiments_path}')
        
        self.temp_dir = str(self.PROJECT_DIR / f'{experiments_path}/Exp_{_experiment}' / 'Temp')
        self.chkpt_dir = self.temp_dir  + '/checkpoints'
        self.out_dir = self.PROJECT_DIR / f'{experiments_path}/Exp_{_experiment}' / 'Results'
        self.model_deploy = self.PROJECT_DIR / f'{experiments_path}/Exp_{_experiment}/Exp_{_experiment}_Deploy'
        self.last_weights_path = str(Path(self.chkpt_dir) / 'last.ckpt')
        self.bird_names_map = str(self.DATA_DIR  / 'Bird_Names/bird_map.csv')
        self.bird_map_for_model = self.model_deploy / f'exp_{_experiment}_bird_map.csv'
        self.model_config = self.model_deploy / f'exp_{_experiment}_config.yaml'
        # self.background_noise_paths = [path for path in Path(self.BACKGROUND_NOISE_FLDR).rglob('*') if path.suffix in {'.ogg', '.flac', '.wav', '.mp3', '.WAV'}]

class TrainingParameters:
    def __init__(self, options=None):
        self.TRAIN = options['run_training']
        self.EPOCHS = options['epochs'] 
        self.YEAR = 25
        self.EXPERIMENT = options['experiment']
        self.NUM_WORKERS = options['num_cores']
        self.BATCH_SIZE = 32 # 12,  16, 32, 64 for sizes 512, 348, 32-larger network, 256
        self.TEST_BATCH_SIZE = 16
        self.PATIENCE = 8
        self.KEEP_LAST= 4
        self.MIN_DELTA = 0
        self.SEED = 2025
        self.MODEL = 'tf_efficientnet_b0.ns_jft_in1k' #, #'eca_nfnet_l0' #'tf_efficientnet_b0.ns_jft_in1k' #'convnext_tiny.in12k_ft_in1k' #'convnext_tiny.fb_in22k', 'eca_nfnet_l0' #  # 'tf_efficientnetv2_s.in21k_ft_in1k'
        self.WEIGHTED_SAMPLING = True
        self.WEIGHT_DECAY = 1e-5
        self.WARMUP_EPOCHS = 2
        self.INITIAL_LR = 1e-4 
        self.LR = 1e-3
        self.MIN_LR = 1e-5
        self.LR_CYCLE_LENGTH = 12
        self.LR_DECAY = 0.2
        self.EPOCHS_TO_UNFREEZE_BACKBONE = 8
        self.DEVICE = torch.device('mps')
        self.GPU = 'mps'# if torch.cuda.is_available() else 'cpu' #for Pytorch Lightning
        self.PRECISION = '16-mixed' #if self.GPU == 'gpu' else 32
        self.LOSS_FUNCTION_NAME =  'BCEWithLogitsLoss'#'BCEFocal2WayLoss'#'BCEFocal2WayLoss' #'BCEWithLogitsLoss', 'BCEFocalLoss',
        self.USE_MIXUP = False
        self.MIXUP_ALPHA = .25 #Tried .4 and performance droped slightly from .64 to .63 (so inconclusive)
        self.LOW_ALPHA = 0.2 #For Focal Loss, for the most common classes, we downweigt the 'easy' prediction of 'false'
        self.MID_ALPHA = 0.3 
        self.HIGH_ALPHA = 0.4 #For the rare classes, we want the decision to have more impact on the loss compared to the common ones.
        self.FIRST_AUGMENTATION_UPDATE = 5
        self.SECOND_AUGMENTATION_UPDATE = 10
        
        #Alpha does two things. 
        # 1.  For alpha < 0.5 rewards the hard prediction (True) more than the easy one (False) 
        # 2.  We want the magnitude of (1) to be greater for the rare labels so the training gradients aren't dominated by performance on common labels.                        


class NzBirdData:
    N_FOLDS = 10
    USE_SECONDARY = False
    RARE_THRESHOLD = 10 # Classes with less samples than this will not be allowed in validation dataset, and will be up-sampled to this value
    SPATIAL_LIMITS = None #Filter the dataset by lat and long. For example: {'WEST':0, 'EAST':10, 'NORTH': -20, 'SOUTH':-30}
    MAX_PER_CLASS = 30000   #Cap the maximum number of samples allowed in any particular class to prevent extreme imbalance
    MAX_PER_CLASS_VAL = 30000 #(Disabled) 300  #Cap the max for the val classes so that the val score isn't too dominated by the common classes
    EXCLUDED_CLASSES = []
    LOW_ALPHA_CLASSES = ['morepo2'  'nezbel1' 'gryger1' 'silver3' 'tomtit1' 'eurbla' 'tui1' 'nezkak1'],
    HIGH_ALPHA_CLASSES = ['spocra2', 'easros1', 'spocra1', 'redjun1', 'takahe3', 'codpet1', 'chukar', 'caster1', 'parpet1', 
                          'charob1', 'okbkiw1', 'motpet', 'gretea1', 'bluduc1', 'saddle2', 'blbgul1', 'kokako3', 'dobplo1', 
                          'rinphe1', 'chiger2', 'aussho1', 'welswa1', 'litowl1', 'whfter1', 'larus', 'compea', 'litpen1', 
                          'mallar3', 'baicra4', 'houspa', 'blkswa', 'coopet', 'swahar1', 'calqua', 'blfter1', 'piesti1'], 
    #LABEL_SMOOTHING = 0.1
    SECONDARY_WEIGHTS_TRAIN = 0.7
    SECONDARY_WEIGHTS_VAL = 0.4

    DO_WINDOWING_BEFORE_DATASET_CREATION = False

class DefaultAudio:
    IMAGE_SHAPE = (1,1)  #5 second chunks position in final image: height x width
    DURATION = 10  # Duration the loaded sound file will be randomly cropped or padded to for training.
    OVERLAP = 0.5  # The proportion of the duration that the chunks will overlap
    SR = 48000
    IMAGE_WIDTH = 386 #384 #512 # 256 #The spectrogram will get cropped/padded to this square regardless of any audio considerations
    CHUNK_WIDTH = IMAGE_WIDTH if IMAGE_SHAPE[1] == 1 else IMAGE_WIDTH // 2  #Number of frames wide for each sub-image
    N_MELS = IMAGE_WIDTH // 2 if IMAGE_SHAPE[0] == 2 else IMAGE_WIDTH #Height of the chunk spectrograms
    N_FFT = 2048 #3072 #2048 *2 #3072 or 2048 #N_fft/2 + 1 bins will get made prior to downsampling to the value of N_MELS
    FMIN = 20
    FMAX = 14000 
    HOP_LENGTH = 1243 #826 #620 #310, 620, 826, 1243, for chunks widths of 128, 192, 256, 516 respectively
    PCEN = False
    USE_DELTAS = True


def save_model_config(paths, audio_cfg, train_cfg):
    model_config = {
        'basename': train_cfg.MODEL,
        'window_duration': 10,  # Add window parameters
        'window_overlap': 0.5,
        'n_mels': audio_cfg.IMAGE_WIDTH,
        'n_fft': audio_cfg.N_FFT,
        'use_deltas': audio_cfg.USE_DELTAS,
        'hop_length': audio_cfg.HOP_LENGTH,
        'aggregation': 'none'  # Remove multi-segment aggregation
    }
    
    with paths.model_config.open("w") as f:
        yaml.dump(model_config, f, default_flow_style=False)

class Stop: #bold red
    S = '\033[1m' + '\033[91m'
    E = '\033[0m'
    
class Go: #bold green
    S = '\033[1m' + '\033[32m'
    E = '\033[0m'
    
class Blue: #for general info
    S = '\033[1m' + '\033[94m'
    E = '\033[0m'


def get_pseudos(path):
    '''
    To train where the dataset has been pre-classified by other models
    Returns a dict of list of lists.  Each sub-list is the prediction values, 
    the position of the sub-list corresponts to the time-position
    in the sample, with each list representing a chunk of 5 seconds
    '''
    pseudo_df = pd.read_csv(path)

    if 'latitude' in pseudo_df.columns and 'longitude' in pseudo_df.columns:
        pseudo_df.drop(columns=['latitude', 'longitude'], inplace=True)

    #drop any rows where all the values are 0
    cols_after_4th = pseudo_df.columns[4:]
    mask = (pseudo_df[cols_after_4th] == 0).all(axis=1)
    pseudo_df = pseudo_df[~mask]

    print(pseudo_df.iloc[:,:6].head())

    grouped = pseudo_df.groupby('filename')
    birdlist = pseudo_df.iloc[:,4:].columns.tolist()
    print(f'There are {len(birdlist)} birds in the value columns')
    pseudo_dict = {}

    for filename, group in grouped:
        group_sorted = group.sort_values(by='time')
        values = group_sorted[birdlist].values.tolist()
        pseudo_dict[filename] = values

    return pseudo_dict


def load_sf(wav_path):
    y, _ = torchaudio.load(wav_path)
    if y.shape[0] == 2:
        y = torch.mean(y, dim=0, keepdim=True)
    y = y.squeeze().numpy() 
    if not np.isfinite(y).all():
        y[np.isnan(y)] = np.mean(y)
    return y
    

def balance_primary_label(df, label_column='primary_label', max_count=200):
    value_counts = df[label_column].value_counts()
    balanced_df = pd.DataFrame(columns=df.columns)
    for value, count in value_counts.items():
        value_df = df[df[label_column] == value]
        if count > max_count:
            value_df = value_df.sample(n=max_count, random_state=1)
        balanced_df = pd.concat([balanced_df, value_df], axis=0)
    balanced_df = balanced_df.sample(frac=1, random_state=1).reset_index(drop=True)
    
    return balanced_df

    
def compute_pcen(y, audio_cfg):
    if not np.isfinite(y).all():
        y[np.isnan(y)] = np.zeros_like(y)
        y[np.isinf(y)] = np.max(y)
    
    melspec = librosa.feature.melspectrogram(y=y, 
                                             sr=audio_cfg.SR, 
                                             n_mels=audio_cfg.N_MELS, 
                                             n_fft= audio_cfg.N_FFT, 
                                             fmin=audio_cfg.FMIN, 
                                             fmax=audio_cfg.FMAX
                                            )
    pcen = librosa.pcen(melspec, 
                        sr=audio_cfg.SR, 
                        gain=0.98, 
                        bias=2, 
                        power=0.5, 
                        time_constant=0.4, 
                        eps=0.000001
                       )
    return pcen.astype(np.float32)


def compute_melspec(y, audio_cfg):
    if not np.isfinite(y).all():
        y[np.isnan(y)] = np.zeros_like(y)
        y[np.isinf(y)] = np.max(y)
    
    melspec = librosa.feature.melspectrogram(y=y, 
                                             sr=audio_cfg.SR, 
                                             n_mels=audio_cfg.N_MELS, 
                                             n_fft=audio_cfg.N_FFT, 
                                             hop_length = audio_cfg.HOP_LENGTH, 
                                             fmin=audio_cfg.FMIN, 
                                             fmax=audio_cfg.FMAX
                                            ) 
    return librosa.power_to_db(melspec)


def mono_to_color(X, eps=1e-6, use_deltas=False):
    _min, _max = X.min(), X.max()
    if (_max - _min) > eps:
        X = (X - _min) / (_max - _min) #scales to a range of [0,1]
        X = X.astype(np.float32)
    else:
        X = np.zeros_like(X, dtype=np.float32)

    if use_deltas:
        T = torch.tensor(X, dtype=torch.float32)
        delta = compute_deltas(T)
        delta_2 = compute_deltas(delta)
        delta, delta_2 = delta.numpy(), delta_2.numpy()
        X = np.stack([X, delta, delta_2], axis=-1)
    else:
        X = np.stack([X, X, X], axis=-1) #puts the chanels last, like a normal image
    
    return X


def crop_or_pad(y, length,  train='train', path=None, background_paths=None):
    initial_length = len(y)
    max_vol = np.abs(y).max()
    if max_vol == 0:
        print('Warning, there was training sample of all zeros before padding')
        if path is not None:
            print(f'The filepath of this sample was {path}')
    if initial_length == 0:
        print('Warning, there was a sample of initial length zero before padding')
    if 3 * initial_length < length:
        random_values = np.random.random(initial_length)
        y = np.concatenate([y,random_values,y])
    elif 2 * initial_length < length:
        random_values = np.random.random(initial_length//2)
        y = np.concatenate([y,random_values,y])
    if len(y) < length:
        y = np.concatenate([y, y]) 
    
    def Normalize(array):
        max_vol = np.abs(array).max()
        if max_vol == 0:
            length = len(array)
            array = np.random.random(length)
            print('Warning, there was a final training sample of all zeros, replacing with random noise')
            return array  # or return array filled with zeros, if appropriate
        return array * 1 / max_vol

    if len(y) < length:
        difference = length - len(y)
        fill=np.zeros(difference)
        y = np.concatenate([y, fill])
    else:
        if train != 'train':
            start = 0
        else:
            start = 0
            start = np.random.randint(len(y) - length)
        y = y[start: start + length]
    y = Normalize(y)
    return y


def random_crop(arr, length):
    '''For cropping backgrounds from a larger clip to a chosen length'''
    if len(arr) > length:
        start = np.random.randint(len(arr) - length)
        arr = arr[start: start + length]
    return arr


def padded_cmap(solution, submission, padding_factor=5):
    solution = solution.fillna(0).replace([np.inf, -np.inf], 0)
    submission = submission.fillna(0).replace([np.inf, -np.inf], 0)
    new_rows = []
    for i in range(padding_factor):
        new_rows.append([1 for i in range(len(solution.columns))])
    new_rows = pd.DataFrame(new_rows)
    new_rows.columns = solution.columns
    padded_solution = pd.concat([solution, new_rows]).reset_index(drop=True).copy()
    padded_submission = pd.concat([submission, new_rows]).reset_index(drop=True).copy()
    score = skm.average_precision_score(
        padded_solution.values,
        padded_submission.values,
        average='macro')    
    return score


def padded_cmap_by_class(solution, submission, padding_factor=5):
    solution = solution.fillna(0).replace([np.inf, -np.inf], 0)
    submission = submission.fillna(0).replace([np.inf, -np.inf], 0)
    new_rows = []
    for i in range(padding_factor):
        new_rows.append([1 for i in range(len(solution.columns))])
    new_rows = pd.DataFrame(new_rows)
    new_rows.columns = solution.columns
    padded_solution = pd.concat([solution, new_rows]).reset_index(drop=True).copy()
    padded_submission = pd.concat([submission, new_rows]).reset_index(drop=True).copy()
    
    column_headers = list(solution.columns)
    scores = {}
    
    for column in column_headers:
        score = skm.average_precision_score(
            padded_solution[[column]].values,
            padded_submission[[column]].values,
            average='macro')    
        scores[column] = score
    return scores


def map_score(solution, submission):
    solution = solution.fillna(0).replace([pd.np.inf, -pd.np.inf], 0)
    submission = submission.fillna(0).replace([pd.np.inf, -pd.np.inf], 0)
    score = skm.average_precision_score(
        solution.values,
        submission.values,
        average='micro')  
    return score


def plot_by_class(df_target, df_pred):
    cmap5_by_class = padded_cmap_by_class(df_target, df_pred, padding_factor=5)
    col_sums = [(col, df_target[col].sum()) for col in df_target.columns]
    names_by_frequency = sorted(col_sums, key=lambda x: x[1], reverse=True)
    names = [name for name, _ in names_by_frequency]
    counts = [count for _, count in names_by_frequency]
    scores = [cmap5_by_class[name] for name in names]
    df = pd.DataFrame({'names': names, 'counts': counts, 'scores': scores})
    df["scores"] = pd.to_numeric(df["scores"])
    df["counts"] = pd.to_numeric(df["counts"])
    fig = px.bar(df, x='scores', y='names', color='counts', orientation='h', hover_data=['counts', 'scores'], range_x=[0, 1])
    fig.update_layout(height=1200)
    fig.show()
    return names, scores, counts


def save_naming_scheme(train_df, val_df, naming_csv_path):
    '''Saves out basic statistics about the binary classification dataset'''
    
    # Calculate counts for training set
    train_pos = (train_df['label'] == 1).sum()
    train_neg = (train_df['label'] == 0).sum()
    train_total = len(train_df)
    
    # Calculate counts for validation set
    val_pos = (val_df['label'] == 1).sum()
    val_neg = (val_df['label'] == 0).sum()
    val_total = len(val_df)
    
    # Create summary DataFrame
    summary_df = pd.DataFrame({
        'Set': ['Training', 'Training', 'Validation', 'Validation'],
        'Class': ['Positive', 'Negative', 'Positive', 'Negative'],
        'Samples': [train_pos, train_neg, val_pos, val_neg],
        'Percentage': [
            train_pos/train_total * 100,
            train_neg/train_total * 100,
            val_pos/val_total * 100,
            val_neg/val_total * 100
        ]
    })
    
    # Add totals
    total_row = pd.DataFrame({
        'Set': ['Total', 'Total'],
        'Class': ['Training', 'Validation'],
        'Samples': [train_total, val_total],
        'Percentage': [100, 100]
    })
    
    summary_df = pd.concat([summary_df, total_row])
    
    print("Dataset Summary:")
    print(f"Training set: {train_total} total samples")
    print(f"  Positive: {train_pos} ({train_pos/train_total*100:.1f}%)")
    print(f"  Negative: {train_neg} ({train_neg/train_total*100:.1f}%)")
    print(f"Validation set: {val_total} total samples")
    print(f"  Positive: {val_pos} ({val_pos/val_total*100:.1f}%)")
    print(f"  Negative: {val_neg} ({val_neg/val_total*100:.1f}%)")
    
    # Save to CSV
    summary_df.to_csv(naming_csv_path, index=False)
    return


############################################# Data Augmentation # ######################################
########################################################################################################

class Compose:
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, y: np.ndarray, sr):
        for trns in self.transforms:
            y = trns(y, sr)
        return y
    

class AudioTransform:
    def __init__(self, always_apply=False, p=0.5):
        self.always_apply = always_apply
        self.p = p

    def __call__(self, y: np.ndarray, sr):
        if self.always_apply:
            return self.apply(y, sr=sr)
        else:
            if np.random.rand() < self.p:
                return self.apply(y, sr=sr)
            else:
                return y

    def apply(self, y: np.ndarray, **params):
        raise NotImplementedError
        
        
class OneOf(Compose):
    def __init__(self, transforms, p=0.5):
        super().__init__(transforms)
        self.p = p
        transforms_ps = [t.p for t in transforms]
        s = sum(transforms_ps)
        self.transforms_ps = [t / s for t in transforms_ps]

    def __call__(self, y: np.ndarray, sr):
        data = y
        if self.transforms_ps and (np.random.random() < self.p):
            random_state = np.random.RandomState(np.random.randint(0, 2 ** 16 - 1))
            t = random_state.choice(self.transforms, p=self.transforms_ps)
            data = t(y, sr)
        return data
    
    
class Normalize(AudioTransform):
    def __init__(self, always_apply=False, p=1):
        super().__init__(always_apply, p)

    def apply(self, y: np.ndarray, **params):
        max_vol = np.abs(y).max()
        if max_vol < 1e-10:
            return y
        y_vol = y * (1.0 / (max_vol + 1e-10))
        return np.asfortranarray(y_vol)
    
    
class RandomNoiseInjection(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, max_noise_level=1):
        super().__init__(always_apply, p)

        self.noise_level = (0.0, max_noise_level)

    def apply(self, y: np.ndarray, **params):
        noise_level = np.random.uniform(*self.noise_level)
        noise = np.random.randn(len(y))
        augmented = (y + noise * noise_level).astype(y.dtype)
        return augmented
    
    
class GaussianNoise(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, min_snr=5, max_snr=20):
        super().__init__(always_apply, p)

        self.min_snr = min_snr
        self.max_snr = max_snr

    def apply(self, y: np.ndarray, **params):
        snr = np.random.uniform(self.min_snr, self.max_snr)
        a_signal = np.sqrt(y ** 2).max()
        a_noise = a_signal / (10 ** (snr / 20))

        white_noise = np.random.randn(len(y))
        a_white = np.sqrt(white_noise ** 2).max()
        augmented = (y + white_noise * 1 / a_white * a_noise).astype(y.dtype)
        return augmented
    
#https://github.com/felixpatzelt/colorednoise
class PinkNoise(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, min_snr=5, max_snr=20):
        super().__init__(always_apply, p)
        self.min_snr = min_snr
        self.max_snr = max_snr

    def apply(self, y: np.ndarray, **params):
        snr = np.random.uniform(self.min_snr, self.max_snr)
        a_signal = np.sqrt(y ** 2).max()
        a_noise = a_signal / (10 ** (snr / 20))

        pink_noise = cn.powerlaw_psd_gaussian(1, len(y))
        a_pink = np.sqrt(pink_noise ** 2).max()
        augmented = (y + pink_noise * 1 / a_pink * a_noise).astype(y.dtype)
        return augmented
    
    
class BrownNoise(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, min_snr=5, max_snr=20):
        super().__init__(always_apply, p)
        self.min_snr = min_snr
        self.max_snr = max_snr

    def apply(self, y: np.ndarray, **params):
        snr = np.random.uniform(self.min_snr, self.max_snr)
        a_signal = np.sqrt(y ** 2).max()
        a_noise = a_signal / (10 ** (snr / 20))

        brown_noise = cn.powerlaw_psd_gaussian(2, len(y))
        a_brown = np.sqrt(brown_noise ** 2).max()
        augmented = (y + brown_noise * 1 / a_brown * a_noise).astype(y.dtype)
        return augmented
    

#https://www.kaggle.com/code/hidehisaarai1213/rfcx-audio-data-augmentation-japanese-english
#https://medium.com/@makcedward/data-augmentation-for-audio-76912b01fdf6
class AddBackround(AudioTransform):
    def __init__(self, 
                 duration,
                 sr,
                 background_noise_paths,
                 always_apply=True, 
                 p=0.6, 
                 min_snr=1, 
                 max_snr=3,
                 ):
        super().__init__(always_apply, p)

        self.min_snr = min_snr
        self.max_snr = max_snr
        self.back_pths = background_noise_paths
        self.background = load_sf(random.choice(self.back_pths))
        self.d_len = duration * sr

    def apply(self, y: np.ndarray, **params):
        snr = np.random.uniform(self.min_snr, self.max_snr)
        if random.random() < 0.2:
            self.background = load_sf(random.choice(self.back_pths))
        
        cropped_background = random_crop(self.background, self.d_len)

        a_signal = np.sqrt(y ** 2).max()
        a_noise = a_signal / (10 ** (snr / 20))  
        l_signal = len(y)

        a_background = np.sqrt(cropped_background ** 2).max()
        l_background = len(cropped_background)

        if l_signal > l_background:
            ratio = l_signal//l_background
            cropped_background = np.tile(cropped_background, ratio+1 )
            cropped_background = cropped_background[0:l_signal]

        if l_signal < l_background:    
            cropped_background = cropped_background[0:l_signal]

        augmented = (y + cropped_background * 1 / a_background * a_noise).astype(y.dtype)
        return augmented  
    
    
def spec_augment(spec: np.ndarray, 
                 num_mask=3, 
                 freq_masking_max_percentage=0.1,
                 time_masking_max_percentage=0.1, 
                 p=0.5):
    if random.uniform(0, 1) > p:
        return spec

    # frequency masking
    num_freq_masks = random.randint(1, num_mask)
    for i in range(num_freq_masks):
        freq_percentage = random.uniform(0, freq_masking_max_percentage)
        freq_mask_size = int(freq_percentage * spec.shape[0])
        freq_mask_pos = random.randint(0, spec.shape[0] - freq_mask_size)
        spec[freq_mask_pos:freq_mask_pos+freq_mask_size, :] = 0

    # time masking
    num_time_masks = random.randint(1, num_mask)
    for i in range(num_time_masks):
        time_percentage = random.uniform(0, time_masking_max_percentage)
        time_mask_size = int(time_percentage * spec.shape[1])
        time_mask_pos = random.randint(0, spec.shape[1] - time_mask_size)
        spec[:, time_mask_pos:time_mask_pos+time_mask_size] = 0

    return spec


class AbluTransforms():
    MEAN = (0.485, 0.456, 0.406) # RGB
    STD = (0.229, 0.224, 0.225) # RGB
    
    def __init__(self, audio, model_input_size=(224, 224)):
        # self.image_width = audio.IMAGE_WIDTH
        self.target_height = model_input_size[0]
        self.target_width = model_input_size[1]

        self.train = A.Compose([
                        A.CoarseDropout(p=0.4), #max_holes=4?
                        A.Resize(height=self.target_height, width=self.target_width),
                        # A.PadIfNeeded(min_height=self.image_width, min_width=self.image_width),
                        # A.CenterCrop(width=self.image_width, height=self.image_width), 
                        # A.Normalize(self.MEAN, self.STD, max_pixel_value=1.0),    #, always_apply=True?
                        ])
        
        self.valid = A.Compose([
                        A.Resize(height=self.target_height, width=self.target_width),
                        # A.PadIfNeeded(min_height=self.image_width, min_width=self.image_width),
                        # A.CenterCrop(width=self.image_width, height=self.image_width),  
                        # A.Normalize(self.MEAN, self.STD, max_pixel_value=1.0), #,always_apply=True?
                        ])


class PrepareImage():
    mean = .5
    std = .22
    def __init__(self, height, width):
        self.height = width
        self.width = width
        self.prep = A.Compose([
            A.PadIfNeeded(min_height=self.height, min_width=self.width),
            A.CenterCrop(width=self.width, height=self.height), 
            # A.Normalize(mean=self.mean, std=self.std, max_pixel_value=1.0), #, always_apply=True?
        ])


def mixup_data(x, y, alpha, device):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def fold_image(arr, shape): 
    '''chop the image in half along the temporal dimension and stack to a square image
    Goal is to allow more pixels and segments in the temporal domain than frequency'''
    length = arr.shape[1]
    num_vertical = shape[0]
    cols = length//num_vertical
    remainder = length % num_vertical
    if num_vertical == 2:
        half0 = arr[:, :cols + remainder]   #added the .T v55
        half1 = arr[:, cols:]  #added the .T v53
        arr =  np.vstack((half0, half1))  #changed to h-stack v55
    elif num_vertical == 4:
        half0 = arr[:, :cols + remainder]
        half1 = arr[:, cols:]
        half2 = arr[:, cols:]
        half3 = arr[:, cols:]
        arr =  np.vstack((half0, half1, half2, half3))  #changed to h-stack v55
    return arr


############################################# Dataset Definition  ######################################
########################################################################################################

class WaveformDataset(Dataset):
    def __init__(self, 
                 df, #This is the default dataframe with only human-labelled data
                 audio,
                 paths,
                 epoch=0,
                 train=True, 
                 augmentation_updates = [6,11]
                ): 
        self.epoch=epoch
        self.sr = audio.SR
        self.train = train
        self.df = df

        self.window_samples = int(audio.DURATION * audio.SR)

        self.duration = audio.DURATION
        # self.d_len = self.duration * self.sr
        self.image_transform = AbluTransforms(audio).train if train else AbluTransforms(audio).valid
        # self.back_pths = paths.background_noise_paths
        self.height = audio.N_MELS
        self.width = audio.CHUNK_WIDTH
        # self.image_shape = audio.IMAGE_SHAPE
        # self.num_chunks = self.image_shape[0] * self.image_shape[1]
        # self.chunk_lenth = self.d_len // self.num_chunks
        self.prep_image = PrepareImage(height=self.height, width = self.width)
        # self.pseudo_dict = pseudo_dict
        self.pcen = audio.PCEN
        self.use_deltas = audio.USE_DELTAS
        self.audio_cfg = audio
        self.first_aug_reset = augmentation_updates[0]
        self.second_aug_reset = augmentation_updates[1]

        if self.train:
            self.wave_transforms = Compose(
                [
                    OneOf(
                        [
                            RandomNoiseInjection(p=1, max_noise_level=0.04),
                            GaussianNoise(p=1, min_snr=1, max_snr=5),
                            PinkNoise(p=1, min_snr=1, max_snr=5),
                            BrownNoise(p=1, min_snr=1, max_snr=5),
                        ],
                        p=.25,
                    ),
                    # AddBackround(audio.DURATION, self.sr, self.back_pths, p=.25, min_snr=1.5, max_snr=3),
                    Normalize(p=1),
                ]
            )
        else:
            self.wave_transforms = Compose([Normalize(p=1)])
        
    def __len__(self):
        return len(self.df)
    
    def set_epoch(self, epoch):
        self.epoch = epoch

    def reset_wave_augmentation(self, epoch):
        if self.train and self.first_aug_reset <= epoch < self.second_aug_reset:    #self.first_aug_reset  was 5 & 10
            print(f'Using medium waveform augmentation on epoch {epoch}')
            self.wave_transforms = Compose(
                [
                    OneOf(
                        [
                            RandomNoiseInjection(p=1, max_noise_level=0.04),
                            GaussianNoise(p=1, min_snr=1, max_snr=5),
                            PinkNoise(p=1, min_snr=1, max_snr=5),
                            BrownNoise(p=1, min_snr=1, max_snr=5),
                        ],
                        p=.15,
                    ),
                    # AddBackround(self.duration, self.sr, self.back_pths, p=.15, min_snr=1.5, max_snr=3),  #Tried various SNR and p combinations.  Adding background noise just doesn't seem to do much.
                    Normalize(p=1),
                ]
            )
        elif self.train and epoch >= self.second_aug_reset:
            print(f'Using minimal waveform augmentation on epoch {epoch}')
            self.wave_transforms = Compose(
                [
                    OneOf(
                        [
                            RandomNoiseInjection(p=1, max_noise_level=0.04),
                            GaussianNoise(p=1, min_snr=1.5, max_snr=5),
                            PinkNoise(p=1, min_snr=1.5, max_snr=5),
                            BrownNoise(p=1, min_snr=1.5, max_snr=5),
                        ],
                        p=.1,
                    ),
                    Normalize(p=1),
                ]
            )

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio_path = row['filepath']
        
        # Load audio and extract window
        try:
            audio = load_sf(audio_path)
            start_sample = row.get('start_sample', 0)
            window = audio[start_sample:start_sample + self.window_samples]
        except:
            # Create random noise if file can't be loaded
            print(f'Error loading {audio_path}, creating random noise')
            window = np.random.randn(self.window_samples)
        
        # Apply audio augmentation
        window = self.wave_transforms(window, sr=self.sr)
        
        # Convert to spectrogram
        if self.pcen:
            image = compute_pcen(window, self.audio_cfg)
        else:
            image = compute_melspec(window, self.audio_cfg)
        
        # Normalize spectrogram
        image = self.prep_image.prep(image=image)['image']
        # Convert to color and apply image transforms
        image = mono_to_color(image, use_deltas=self.use_deltas)
        image = self.image_transform(image=image)['image']
        image = image.transpose(2, 0, 1).astype(np.float32)
        
        target = torch.tensor(row['label']).float()
        
        return image, target


############################################# Loss Functions ######################################
##################################################################################################

# https://www.kaggle.com/c/rfcx-species-audio-detection/discussion/213075
# https://www.kaggle.com/code/thedrcat/focal-multilabel-loss-in-pytorch-explained
class BCEFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, loss_alphas=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_factor = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        
        loss = alpha_factor * modulating_factor * bce_loss
        return loss.mean()


class BCEFocal2WayLoss(nn.Module):
    def __init__(self, weights=[1, 1], loss_alphas=None):
        super().__init__()

        self.focal = BCEFocalLoss(loss_alphas=loss_alphas)
        self.weights = weights

    def forward(self, input, target):
        input_ = input["logit"]
        target = target.float()
        loss = self.focal(input_, target)

        #my simplified version, using the segment logits directly instead of the interpolated function from original code
        segmentwise_logit, _ = input['segmentwise_logit'].max(dim=1) #also tried mean, but it didn't work for some reason
        aux_loss = self.focal(segmentwise_logit, target)   

        return self.weights[1] * loss + self.weights[1] * aux_loss


class LossFunctions():
    '''A wrapper class, that incudes various loss function types and takes a dictionary
    as an input with the various outputs from the model'''
    def __init__(self, loss_fn_nm, loss_alphas=None):
        loss_dict = {
                'BCEFocal2WayLoss': BCEFocal2WayLoss(loss_alphas=loss_alphas),
                'BCEFocalLoss': BCEFocalLoss(),
                'BCEWithLogitsLoss': nn.BCEWithLogitsLoss(),
                'CrossEntropyLoss': nn.CrossEntropyLoss()
                }
        self.loss_fn_nm = loss_fn_nm
        self.loss_fn = loss_dict.get(loss_fn_nm, nn.CrossEntropyLoss())
        
    def loss(self, preds_dict, target):
        if self.loss_fn_nm == 'BCEFocal2WayLoss':     
            loss_val = self.loss_fn(preds_dict, target)  #'BCEFocal2WayLoss'
        else:   # ['BCEFocalLoss', 'BCELossWithLogits','CrossEntropyLoss']
            loss_val = self.loss_fn(preds_dict['logit'], target)
        return loss_val


############################################# Model Definition  ######################################
##################################################################################################

class BirdSoundModel(pl.LightningModule):
    
    def init_layer(self, layer):
        """Initialize a Linear or Conv2d layer"""
        nn.init.xavier_uniform_(layer.weight)
        if hasattr(layer, "bias"):
            if layer.bias is not None:
                layer.bias.data.fill_(0.)

    def init_bn(self, bn):
        """Initialize a Batch Normalization layer"""
        bn.bias.data.fill_(0.)
        bn.weight.data.fill_(1.0)

    def init_weight(self):
        """Initialize layers"""
        # self.init_bn(self.bn0)
        # Initialize the new classifier layers
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                self.init_layer(layer)
        # self.classifier[-1].bias.data.fill_(0) #positive bias

    def __init__(self,
                 cfg, 
                 audio,
                 paths,
                 in_channels=3):
        super().__init__()
        self._device = cfg.DEVICE

        # self.bn0 = nn.BatchNorm2d(audio.IMAGE_WIDTH, eps=1e-5, momentum=0.1).to(self._device) #if cfg.RESHAPE_IMAGE else nn.BatchNorm2d(audio.N_MELS)
        self.base_model = timm.create_model(
            cfg.MODEL, 
            pretrained=True, 
            in_chans=in_channels
        ).to(self._device)
        
        # Remove classification head
        in_features = self.base_model.classifier.in_features
        self.base_model.classifier = nn.Identity()

        # New classification head
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1)
        ).to(self._device)
        
        self.loss_function = LossFunctions(cfg.LOSS_FUNCTION_NAME).loss
        self.init_weight()

        # layers = list(self.base_model.children())[:-2]
        # self.encoder = nn.Sequential(*layers)

        # if hasattr(self.base_model, "fc"):
        #     in_features = self.base_model.fc.in_features
        # elif cfg.MODEL == 'eca_nfnet_l0':
        #     in_features = self.base_model.head.fc.in_features
        # elif cfg.MODEL == 'convnext_tiny.in12k_ft_in1k':
        #     in_features = self.base_model.head.fc.in_features
        # else:
        #     in_features = self.base_model.classifier.in_features

        # self.fc1 = nn.Linear(in_features, in_features, bias=True)
        # self.image_shape = audio.IMAGE_SHAPE
        # self.att_block = self.AttentionBlock(in_features, 
        #                                     self.num_classes, 
        #                                     activation="sigmoid",
        #                                     image_shape=self.image_shape
        #                                     )
        # self.loss_function = LossFunctions(cfg.LOSS_FUNCTION_NAME, loss_alphas=loss_alphas).loss
        self.val_outputs = []
        self.train_outputs = []
        self.metrics_list = []
        self.val_epoch = 0
        self.epoch_to_unfreeze_backbone = cfg.EPOCHS_TO_UNFREEZE_BACKBONE,
        self.lr = cfg.LR
        self.initial_lr = cfg.INITIAL_LR
        self.min_lr = cfg.MIN_LR
        self.warmup_epochs = cfg.WARMUP_EPOCHS
        self.cycle_length = cfg.LR_CYCLE_LENGTH
        self.lr_decay = cfg.LR_DECAY
        # self.printed_shapes = False
        # self.use_mixup = cfg.USE_MIXUP
        # self.mixup_alpha = cfg.MIXUP_ALPHA
        self.temp_dir = Path(paths.temp_dir)
        self.results_dir = Path(paths.out_dir)
        # maybe eval
        if paths.EVAL_DIR:
            self.eval_dir = Path(paths.EVAL_DIR)
        else:
            self.eval_dir = None
        self.audio_cfg = audio

    def forward(self, x):
        # x = x.transpose(1, 3)
        # x = self.bn0(x)
        # x = x.transpose(1, 3)
        
        x = self.base_model(x)
        x = self.classifier(x)
        
        return {'logit': x.squeeze()}  # Remove multi-way outputs

    def configure_optimizers(self):
        def custom_lr_scheduler(epoch):
            '''CosineAnealingWarmRestarts but with a decay between cycles and a warmup'''
            initial = self.initial_lr / self.lr 
            rel_min = self.min_lr / self.lr
            step_size = (1-initial) / self.warmup_epochs
            warmup = initial + step_size * epoch if epoch <= self.warmup_epochs else 1
            cycle = epoch-self.warmup_epochs
            decay = 1 if epoch <= self.warmup_epochs else self.lr_decay ** (cycle // self.cycle_length)
            phase = np.pi * (cycle % self.cycle_length) / self.cycle_length
            cos_anneal = 1 if epoch <= self.warmup_epochs else  rel_min + (1 - rel_min) * (1 + np.cos(phase)) / 2
            return warmup * decay * cos_anneal #this value gets multipleid by the initial lr (self.lr)
        
        optimizer = Adam(self.parameters(), lr=self.lr)
        scheduler = LambdaLR(optimizer, lr_lambda=custom_lr_scheduler)
        return [optimizer], [scheduler]
    
    def training_step(self, batch, batch_idx):
        image, target = batch  

        preds_dict = self(image)
        loss = self.loss_function(preds_dict, target)

        # Get binary predictions
        probs = torch.sigmoid(preds_dict['logit'])
        preds = (probs > 0.5).float()

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        train_output = {
            "train_loss": loss,
            "logits": preds,  # Using binary predictions
            "targets": target
        }
        self.train_outputs.append(train_output)
        return loss      

    def validation_step(self, batch, batch_idx):
        image, target = batch 

        # Add input validation
        if torch.isnan(image).any():
            print(f"NaN values in input batch {batch_idx}")
            image = torch.nan_to_num(image, nan=0.0)
        
        preds_dict = self(image)
        val_loss = self.loss_function(preds_dict, target)

        # Check loss for NaN
        if torch.isnan(val_loss):
            print(f"NaN loss detected in batch {batch_idx}")
            return None
        
        # Calculate binary predictions and probabilities
        probs = torch.sigmoid(preds_dict['logit'])
        preds = (probs > 0.5).float()

        self.log("val_loss", val_loss, on_step=True, on_epoch=True, prog_bar=True)

        output = {
            "val_loss": val_loss,
            "predictions": preds,
            "probabilities": probs,
            "targets": target
        }
        self.val_outputs.append(output)
        return output

    def train_dataloader(self):
        return self._train_dataloader

    def validation_dataloader(self):
        return self._validation_dataloader
    
    def on_train_epoch_start(self):
        epoch = self.current_epoch
        train_loader = self.trainer.train_dataloader
        train_loader.dataset.reset_wave_augmentation(epoch)
        train_loader.dataset.set_epoch(epoch)

    def on_train_epoch_end(self, *args, **kwargs):  
        epoch = self.current_epoch
        if epoch == self.epoch_to_unfreeze_backbone:
            for param in self.base_model.parameters():
                param.requires_grad = True
            print(f'Unfreezing the backbone after {epoch} epochs')
        
    def on_validation_epoch_end(self):
        val_outputs = self.val_outputs
        avg_val_loss = torch.stack([x['val_loss'] for x in val_outputs]).mean().cpu().detach().numpy()
        
        # Get predictions and targets
        val_preds = torch.cat([x['predictions'] for x in val_outputs]).cpu().detach().numpy()
        val_probs = torch.cat([x['probabilities'] for x in val_outputs]).cpu().detach().numpy()
        val_targets = torch.cat([x['targets'] for x in val_outputs]).cpu().detach().numpy()

        # Calculate metrics
        precision = precision_score(val_targets, val_preds)
        recall = recall_score(val_targets, val_preds)
        f1 = f1_score(val_targets, val_preds)
        auc = roc_auc_score(val_targets, val_probs)

        # Get training metrics if available
        train_outputs = self.train_outputs
        if train_outputs:
            train_losses = [x['train_loss'].cpu().detach().numpy() for x in train_outputs]
            avg_train_loss = sum(train_losses) / len(train_losses) if train_losses else 0.0
            train_preds = torch.cat([x['logits'] for x in train_outputs]).cpu().detach().numpy()
            train_targets = torch.cat([x['targets'] for x in train_outputs]).cpu().detach().numpy()
        else: 
            avg_train_loss = avg_val_loss
            train_preds = np.zeros_like(val_preds)
            train_targets = np.zeros_like(val_targets)

        # Store metrics
        metrics = {
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'val_precision': precision,
            'val_recall': recall,
            'val_f1': f1,
            'val_auc': auc,
        }

        if self.eval_dir:
            real_life_metrics = real_life_evaluate(self, self.eval_dir, self.audio_cfg)
            metrics.update(real_life_metrics)

            self.log("rl_auc", real_life_metrics['rl-auc'], on_epoch=True, prog_bar=True)
            self.log("rl_f1", real_life_metrics['rl-f1'], on_epoch=True, prog_bar=True)
            self.log("rl_precision", real_life_metrics['rl-precision'], on_epoch=True, prog_bar=True)
            self.log("rl_recall", real_life_metrics['rl-recall'], on_epoch=True, prog_bar=True)

        print(Stop.S + f'Epoch {self.current_epoch} train loss {avg_train_loss:.4f}')
        print(f'Epoch {self.current_epoch} validation loss {avg_val_loss:.4f}')
        print(f'Epoch {self.current_epoch} validation metrics:')
        print(f'  Precision: {precision:.4f}')
        print(f'  Recall: {recall:.4f}')
        print(f'  F1: {f1:.4f}')
        print(f'  AUC: {auc:.4f}')

        # Log metrics
        self.log("val_precision", precision, on_epoch=True, prog_bar=True)
        self.log("val_recall", recall, on_epoch=True, prog_bar=True)
        self.log("val_f1", f1, on_epoch=True, prog_bar=True)
        self.log("val_auc", auc, on_epoch=True, prog_bar=True)

        self.metrics_list.append(metrics)
        
        # Clear outputs
        self.val_outputs = []
        self.train_outputs = []
        self.val_epoch += 1

        return
    
    def get_my_metrics_list(self):
        return self.metrics_list


############################################# Training Functions ######################################
######################################################################################################

def get_model(ckpt_path, cfg, audio, paths, classes):
    model = BirdSoundModel(cfg, audio, paths)
    model_state_dict = torch.load(ckpt_path)
    model.load_state_dict(model_state_dict['state_dict'])  
    model.to(cfg.DEVICE)
    return model

def save_models(paths, train_cfg, audio_cfg, classes, deploy_ckpt_selection=1):
    '''This is overkill, but I imagine wanting to modify to pickle 
    the whole model instead of just the checkpoints'''
    checkpoints = [path for path in Path(paths.chkpt_dir).glob('*.ckpt')]
    latest_ckpt_first = sorted(checkpoints, key=lambda p: p.stat().st_ctime, reverse=True)
    selection_idx = max(deploy_ckpt_selection, len(latest_ckpt_first))-1
    
    for idx, ckpt_path in tqdm(enumerate(latest_ckpt_first)) :
        model = get_model(ckpt_path, train_cfg, audio_cfg, paths, classes)
        save_path = str(Path(paths.out_dir) / ckpt_path.name)
        deploy_path = str(Path(paths.model_deploy) / ckpt_path.name)
        torch.save(model.state_dict(), save_path)
        if idx == selection_idx:
            torch.save(model.state_dict(), deploy_path)
        print(Blue.S + 'Weights checkpoint saved to: ' + Blue.E + save_path)

    return save_path  #just returns what ever came last, to check for functionality

def get_class_weights(df):
    df = df.iloc[:, 2:] # removing the 'filepath' and 'primary_label' columns
    col_sums = df.sum()
    counts_array = col_sums.values
    counts_array = np.sqrt(300 + counts_array) 
    class_weights = counts_array.tolist()
    sample_idxs = np.argmax(df.values, axis=1).tolist()
    sampling_weights = [1 / class_weights[idx] for idx in sample_idxs] 
    return sampling_weights


def get_dataloaders(df_train, 
                    df_valid, 
                    paths,
                    audio, 
                    batch_size, 
                    num_workers, 
                    weighted_sampling=False,
                    augmentation_updates=[6,12],
                    device=None):

    ds_train = WaveformDataset(df_train, 
                               audio,
                               paths,
                               train=True, 
                               augmentation_updates=augmentation_updates)
    ds_val = WaveformDataset(df_valid, 
                             audio,
                             paths,
                             train=False)
    
    if weighted_sampling:
        # Update for binary case
        pos_weight = (df_train['label'] == 0).sum() / (df_train['label'] == 1).sum()
        sample_weights = [pos_weight if label == 1 else 1.0 for label in df_train['label']]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(ds_train),
        )
        dl_train = DataLoader(
            ds_train, 
            batch_size=batch_size, 
            sampler=sampler, 
            num_workers=num_workers,
            persistent_workers=True
        )
    else:
        dl_train = DataLoader(
            ds_train, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers,
            persistent_workers=True
        )

    dl_val = DataLoader(
        ds_val, 
        batch_size=batch_size, 
        num_workers=num_workers, 
        persistent_workers=True,
        shuffle=False
        )

    return dl_train, dl_val, ds_train, ds_val

def get_loss_alphas(classes, data_cfg, cfg):
    '''Returns individualised alpha parameters for BCE focal loss'''
    low_indices = [classes.index(x) for x in data_cfg.LOW_ALPHA_CLASSES if x in classes]
    high_indices = [classes.index(x) for x in data_cfg.HIGH_ALPHA_CLASSES if x in classes]
    alphas = np.full(len(classes), cfg.MID_ALPHA)
    alphas[low_indices] = cfg.LOW_ALPHA
    alphas[high_indices] = cfg.HIGH_ALPHA
    return alphas


def test_forward_pass(model, train_loader):
    print("Testing forward pass...")
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx > 0:  # Only test first batch
                break
            images, targets = batch
            images = images.to(model._device)
            targets = targets.to(model._device)
            print(f"Batch shapes - Images: {images.shape}, Targets: {targets.shape}")
            print(f"Batch devices - Images: {images.device}, Targets: {targets.device}")
            try:
                outputs = model(images)
                print("Forward pass successful!")
                print(f"Output shape: {outputs['logit'].shape}")
                break
            except Exception as e:
                print(f"Forward pass failed: {e}")
                raise  # Re-raise the exception to see the full traceback
            break
    model.train()

def real_life_evaluate(model, eval_dir, audio_cfg):
        '''Evaluate the model on the real-life data'''
        model.eval()

        positive_path = eval_dir / 'positive'
        negative_path = eval_dir / 'negative'
        positive_files = [os.path.join(positive_path, f) for f in os.listdir(positive_path) if f.lower().endswith('.wav')]
        negative_files = [os.path.join(negative_path, f) for f in os.listdir(negative_path) if f.lower().endswith('.wav')]
        files = positive_files + negative_files
        labels = [1] * len(positive_files) + [0] * len(negative_files)
        preds = []
        probs = []
        losses = [] # Add list to store losses

        # wave transforms normalise p 1
        wave_transforms = Compose([Normalize(p=1)])
        prep_image = PrepareImage(height=audio_cfg.N_MELS, width=audio_cfg.CHUNK_WIDTH)
        image_transform = AbluTransforms(audio_cfg).valid

        for file in files:
            audio = load_sf(file)
            audio = wave_transforms(audio, sr=audio_cfg.SR)
            image = compute_melspec(audio, audio_cfg)
            image = prep_image.prep(image=image)['image']
            image = mono_to_color(image, use_deltas=audio_cfg.USE_DELTAS)
            image = image_transform(image=image)['image']
            image = image.transpose(2, 0, 1).astype(np.float32)
            image = torch.tensor(image).unsqueeze(0).to(model._device)

            output = model(image)
            prob = torch.sigmoid(output['logit']).item()
            pred = 1 if prob > 0.5 else 0

            # Calculate loss for this sample
            target_label = labels[files.index(file)]
            target = torch.tensor(target_label).float().to(model._device)
            loss = model.loss_function(output, target)
            losses.append(loss.item()) # Store loss

            preds.append(pred)
            probs.append(prob)

        model.train()

        precision = precision_score(labels, preds)
        recall = recall_score(labels, preds)
        f1 = f1_score(labels, preds)
        auc = roc_auc_score(labels, probs)
        avg_loss = sum(losses) / len(losses) if losses else 0.0 # Calculate average loss
        print(f'Real-life evaluation:')
        print(f'  Loss: {avg_loss:.4f}') # Print loss
        print(f'  Precision: {precision:.4f}')
        print(f'  Recall: {recall:.4f}')
        print(f'  F1: {f1:.4f}')
        print(f'  AUC: {auc:.4f}')
        return {'rl-precision': precision, 'rl-recall': recall, 'rl-f1': f1, 'rl-auc': auc, 'rl-loss': avg_loss} # Return loss

def real_life_evaluate_with_roc(models, eval_dir, audio_cfg, consensus=False, plot=True):
    '''Evaluate the model on the real-life data and plot ROC curve'''
    if consensus:
        model1, model2 = models
        model1.eval()
        model2.eval()
    else:
        model1 = models
        model1.eval()

    positive_path = eval_dir / 'positive'
    negative_path = eval_dir / 'negative'
    positive_files = [os.path.join(positive_path, f) for f in os.listdir(positive_path) if f.lower().endswith('.wav')]
    negative_files = [os.path.join(negative_path, f) for f in os.listdir(negative_path) if f.lower().endswith('.wav')]
    files = positive_files + negative_files
    labels = [1] * len(positive_files) + [0] * len(negative_files)
    preds = []
    probs = []

    # wave transforms normalise p 1
    wave_transforms = Compose([Normalize(p=1)])
    prep_image = PrepareImage(height=audio_cfg.N_MELS, width=audio_cfg.CHUNK_WIDTH)
    image_transform = AbluTransforms(audio_cfg).valid

    for file in files:
        audio = load_sf(file)
        audio = wave_transforms(audio, sr=audio_cfg.SR)
        image = compute_melspec(audio, audio_cfg)
        image = prep_image.prep(image=image)['image']
        image = mono_to_color(image, use_deltas=audio_cfg.USE_DELTAS)
        image = image_transform(image=image)['image']
        image = image.transpose(2, 0, 1).astype(np.float32)
        image = torch.tensor(image).unsqueeze(0).to(model1._device)

        output = model1(image)
        prob = torch.sigmoid(output['logit']).item()
        pred = 1 if prob > 0.5 else 0
        
        if consensus:
            output2 = model2(image)
            prob2 = torch.sigmoid(output2['logit']).item()
            prob = (prob + prob2) / 2
            pred2 = 1 if prob2 > 0.5 else 0
            pred = 1 if (pred==1 and pred2==1) else 0

        preds.append(pred)
        probs.append(prob)


    model1.train()
    if consensus:
        model2.train()

    # Calculate metrics
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    f1 = f1_score(labels, preds)
    auc_score = roc_auc_score(labels, probs)
    print(f'Real-life evaluation:')
    print(f'  Precision: {precision:.4f}')
    print(f'  Recall: {recall:.4f}')
    print(f'  F1: {f1:.4f}')
    print(f'  AUC: {auc_score:.4f}')
    
    # Calculate ROC curve points
    fpr, tpr, thresholds = roc_curve(labels, probs)
    
    # Find points for 50% and 98% thresholds
    threshold_50 = 0.5
    threshold_98 = 0.98
    
    # Find indices of closest thresholds
    idx_50 = np.argmin(np.abs(thresholds - threshold_50))
    idx_98 = np.argmin(np.abs(thresholds - threshold_98))
    
    # Calculate percentage (true positives, false positives)
    fpr_50 = fpr[idx_50] * 100
    fpr_98 = fpr[idx_98] * 100
    tpr_50 = tpr[idx_50] * 100
    tpr_98 = tpr[idx_98] * 100

    if plot:
        # Plot ROC curve
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='#0000ff', lw=2, label=f'ROC curve (area = {auc_score:.3f})')
        plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--', label='Random Classifier')
        
        # Add markers for 50% and 98% thresholds with both TPR and FPR
        plt.plot(fpr[idx_50], tpr[idx_50], 'rx', markersize=10, 
                label=f'50% threshold\n(TPR: {tpr_50:.1f}%, FPR: {fpr_50:.1f}%)')
        plt.plot(fpr[idx_98], tpr[idx_98], 'go', markersize=10, 
                label=f'98% threshold\n(TPR: {tpr_98:.1f}%, FPR: {fpr_98:.1f}%)')

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=14)
        plt.ylabel('True Positive Rate', fontsize=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.legend(loc="lower right", fontsize=14)
        
        
        # Add a grid for better readability
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.show()
    
    # Return metrics and ROC data
    return {
        'rl-precision': precision, 
        'rl-recall': recall, 
        'rl-f1': f1, 
        'rl-auc': auc_score,
        'roc_data': {
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds
        },
        'tpr_50': tpr_50,
        'tpr_98': tpr_98,
        'fpr_50': fpr_50,
        'fpr_98': fpr_98
    }

activation = {}
def get_activation_hook(name):
    def hook(model, input, output):
        # Store the output tensor. Detach and clone to avoid holding onto the graph.
        activation[name] = output.detach().clone()
    return hook

def eval_real_life_attention_maps(model, eval_dir, audio_cfg,
                                  cmap_spec='viridis', # New argument for spectrogram colormap
                                  contour_levels=[0.5, 0.75], # Activation levels for contours
                                  contour_colors='white', # Color for contour lines
                                  pos_idx=None, neg_idx=None, plot=True):
    """
    Evaluates the model on one positive and one negative example from the real-life dataset,
    plotting the input spectrogram overlaid with the model's activation map.

    Args:
        model: The trained BirdSoundModel instance.
        eval_dir: Path to the evaluation directory containing 'positive' and 'negative' subfolders.
        audio_cfg: An object containing audio configuration (SR, N_MELS, etc.).
        pos_idx (int, optional): Index of the positive file to use. If None, selects randomly. Defaults to None.
        neg_idx (int, optional): Index of the negative file to use. If None, selects randomly. Defaults to None.
        plot (bool, optional): Whether to display the plot. Defaults to True.
    """
    model.eval()
    device = model._device # Assuming model has _device attribute
    eval_dir = Path(eval_dir)
    positive_path = eval_dir / 'positive'
    negative_path = eval_dir / 'negative'

    try:
        positive_files = sorted([os.path.join(positive_path, f) for f in os.listdir(positive_path) if f.lower().endswith(('.wav', '.ogg', '.mp3', '.flac'))])
        negative_files = sorted([os.path.join(negative_path, f) for f in os.listdir(negative_path) if f.lower().endswith(('.wav', '.ogg', '.mp3', '.flac'))])
    except FileNotFoundError:
        print(f"Error: Evaluation directories not found in {eval_dir}")
        return

    if not positive_files:
        print(f"Error: No positive audio files found in {positive_path}")
        return
    if not negative_files:
        print(f"Error: No negative audio files found in {negative_path}")
        return

    # --- Select Files ---
    if pos_idx is None:
        pos_idx = random.randrange(len(positive_files))
    else:
        pos_idx = min(pos_idx, len(positive_files) - 1)

    if neg_idx is None:
        neg_idx = random.randrange(len(negative_files))
    else:
        neg_idx = min(neg_idx, len(negative_files) - 1)

    pos_file = positive_files[pos_idx]
    neg_file = negative_files[neg_idx]

    print(f"Selected Positive Example ({pos_idx}): {Path(pos_file).name}")
    print(f"Selected Negative Example ({neg_idx}): {Path(neg_file).name}")

    # --- Prepare Image and Activation Map Extraction ---
    results = {}
    target_layer_name = 'conv_head' # Common name for final conv layer in EfficientNet before pooling
    hook_handle = None

    # Find the target layer within the base_model
    target_layer = None
    for name, layer in model.base_model.named_modules():
        # Adjust this check if your layer name is different or nested differently
        if name == target_layer_name:
            target_layer = layer
            break

    if target_layer is None:
        print(f"Warning: Layer '{target_layer_name}' not found in model.base_model. Attempting to find a fallback layer.")
         # Fallback: try finding the layer often preceding global pool/classifier
        potential_last_conv = None
        # Iterate backwards through blocks if they exist
        if hasattr(model.base_model, 'blocks'):
            for block in reversed(model.base_model.blocks):
                 # Common pattern: check for conv layers within the block
                 # This might need adjustment based on specific EfficientNet variant
                if hasattr(block, 'conv_pwl'): # Example check
                    potential_last_conv = block.conv_pwl
                    break
                elif hasattr(block, 'conv'): # Another common name
                     potential_last_conv = block.conv
                     break
        if potential_last_conv:
             target_layer = potential_last_conv
             print(f"Warning: Layer '{target_layer_name}' not found. Using fallback layer: {potential_last_conv}")
        else:
            print(f"Error: Could not find target layer '{target_layer_name}' or a fallback in model.base_model. Cannot extract activation maps.")
            # You might want to print model structure here for debugging: print(model.base_model)
            return
        
    prep_image_util = PrepareImage(height=audio_cfg.N_MELS, width=audio_cfg.CHUNK_WIDTH)
    image_transform_valid = AbluTransforms(audio_cfg).valid

    for label, file_path in [("Positive", pos_file), ("Negative", neg_file)]:
        activation.clear() # Clear previous activation
        hook_registered = False
        try:
            # --- Load and Preprocess Audio ---
            y = load_sf(file_path)
            target_len = int(audio_cfg.DURATION * audio_cfg.SR)
            if len(y) > target_len:
                # Center crop for visualization consistency if longer
                start = (len(y) - target_len) // 2
                window = y[start : start + target_len]
                print(f"Warning: Audio clip is longer than {target_len} samples. Center-cropping to fit.")
            elif len(y) < target_len:
                # Pad if shorter - using existing logic, 'val' ensures no random start
                window = crop_or_pad(y, target_len, train='val')
                print(f"Warning: Audio clip is shorter than {target_len} samples. Padding to fit.")
            else:
                window = y

            # Use validation-style normalization
            window = Normalize(p=1)(window, sr=audio_cfg.SR)
            # Compute Spectrogram (raw for visualization)
            audio_cfg_for_plot = DefaultAudio()
            audio_cfg_for_plot.N_MELS = 256
            audio_cfg_for_plot.CHUNK_WIDTH = 128
            if audio_cfg.PCEN:
                spec_raw = compute_pcen(window, audio_cfg)
                spec_correct_res = compute_pcen(window, audio_cfg_for_plot)
            else:
                spec_raw = compute_melspec(window, audio_cfg)
                spec_correct_res = compute_melspec(window, audio_cfg_for_plot)  
            print(f"Spectrogram shape: {spec_raw.shape}")
            print(f"Correct resolution shape: {spec_correct_res.shape}")
            prep_result = prep_image_util.prep(image=spec_raw.copy())
            image_prepped = prep_result['image'] # This is the spec the model sees

            # Keep color conversion and transforms for model input tensor
            image_color = mono_to_color(image_prepped.copy(), use_deltas=audio_cfg.USE_DELTAS)
            image_transformed = image_transform_valid(image=image_color)['image']
            image_tensor = torch.tensor(image_transformed.transpose(2, 0, 1).astype(np.float32)).unsqueeze(0).to(device)

            # --- Run Model and Get Activation ---
            if target_layer:
                hook_handle = target_layer.register_forward_hook(get_activation_hook('feat'))
                hook_registered = True
            with torch.no_grad():
                outputs = model(image_tensor)
                prob = torch.sigmoid(outputs['logit']).item()
            act_map = activation.get('feat')

            # --- Process Activation Map ---
            act_map = act_map.squeeze(0) # Remove batch dim
            act_map = act_map.mean(dim=0) # Average over channels -> (h_act, w_act)
            act_map_np = act_map.cpu().numpy()

            # --- Resize activation map FIRST to the square shape the model saw ---
            # Assuming model input is square (e.g., 386x386)
            image_shape_h, image_shape_w = spec_correct_res.shape
            act_map_resized_square = cv2.resize(act_map_np, (image_shape_w, image_shape_h), interpolation=cv2.INTER_CUBIC)

            # Normalize the map
            map_min, map_max = act_map_resized_square.min(), act_map_resized_square.max()
            if map_max > map_min:
                act_map_resized_square = (act_map_resized_square - map_min) / (map_max - map_min)
            else:
                # Ensure placeholder has the correct cropped shape
                act_map_resized_square = np.zeros_like(spec_correct_res) # Use image_prepped shape

            # --- Store results for plotting ---
            results[label] = {
                'spec_for_plot': spec_correct_res,           # The (256, 128) content spec
                'act_map_for_plot': act_map_resized_square, # The map resized to (256,256)
                'prob': prob,
                'file': Path(file_path).name
            }

        except Exception as e:
            print(f"Error processing {label} file {file_path}: {e}")
            if label not in results: # Ensure placeholder exists if error occurred early
                 results[label] = {'spec': None, 'act_map': None, 'prob': -1, 'file': Path(file_path).name}
        finally:
            if hook_handle and hook_registered:
                hook_handle.remove()
                hook_registered = False # Prevent double removal

    # --- Plotting ---
    if plot:
        final_contour_colors = contour_colors
        if isinstance(contour_colors, (list, tuple)):
            if len(contour_colors) != len(contour_levels):
                print(f"Warning: Length of contour_colors ({len(contour_colors)}) "
                      f"does not match length of contour_levels ({len(contour_levels)}). "
                      f"Falling back to default color 'white'.")
                final_contour_colors = 'white' # Fallback
        elif not isinstance(contour_colors, str):
             print(f"Warning: Invalid type for contour_colors ({type(contour_colors)}). "
                   f"Expected string or list/tuple. Falling back to 'white'.")
             final_contour_colors = 'white' # Fallback

        fig, axes = plt.subplots(1, 2, figsize=(7.5*2, 4.5))
        # Add contour level info to title if levels are defined
        contour_info = f" (Contours at {contour_levels})" if contour_levels else ""
        fig.suptitle(f"Activation Contours{contour_info} (Layer: {'Identified Layer' if target_layer else 'Not Found'})", fontsize=16)

        for i, label in enumerate(["Positive", "Negative"]):
            ax = axes[i]
            # Use the new keys from results dict
            data = results.get(label, {'spec_for_plot': None, 'act_map_for_plot': None, 'prob': -1, 'file': 'N/A'})
            spec_to_plot = data['spec_for_plot']       # Use the prepared spectrogram
            act_map_norm = data['act_map_for_plot'] # Use the map resized to match spec_to_plot
            prob = data['prob']
            filename = data['file']

            if spec_to_plot is None or act_map_norm is None:
                 ax.text(0.5, 0.5, 'Error Processing', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
                 ax.set_title(f"{label} Example\n{filename}\n(Error)", fontsize=12)
                 continue

            # 1. Display the spectrogram that was input to the model
            spec_to_plot_safe = np.nan_to_num(spec_to_plot)
            # Extent now matches the dimensions of the model input spec
            img_extent = (0, spec_to_plot_safe.shape[1], 0, spec_to_plot_safe.shape[0])
            im = ax.imshow(spec_to_plot_safe, aspect='auto', origin='lower',
                           cmap=cmap_spec, extent=img_extent)

            # 2. Draw contour lines for the activation map (already resized to match)
            if contour_levels and np.any(act_map_norm) and act_map_norm.max() > min(contour_levels):
                # Create coordinates matching spec_to_plot dimensions
                x_coords = np.arange(act_map_norm.shape[1]) # Width (e.g., CHUNK_WIDTH)
                y_coords = np.arange(act_map_norm.shape[0]) # Height (e.g., N_MELS)
                X, Y = np.meshgrid(x_coords, y_coords)

                # Draw contours on top of the image
                ax.contour(X, Y, act_map_norm, levels=contour_levels,
                           colors=final_contour_colors, linewidths=1.5,
                           origin='lower', extent=img_extent) # Extent ensures alignment

            ax.set_title(f"{label} Example\n{filename}\nPrediction Prob: {prob:.3f}", fontsize=12)
            ax.set_xlabel("Time Frame", fontsize=12)
            ax.set_ylabel("Mel Frequency", fontsize=12)
            ax.tick_params(axis='both', which='major', labelsize=12)
            # Create the colorbar and capture the returned object
            cbar = fig.colorbar(im, ax=ax, label='PCEN')
            # Increase the font size of the colorbar label
            cbar.set_label('PCEN', fontsize=12)


        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    model.train() # Put model back in training mode if necessary

def run_training(dl_train, dl_val, paths, data_cfg, train_cfg, audio_cfg, checkpoint_dir):
    pl.seed_everything(train_cfg.SEED, workers=True)
    torch.set_flush_denormal(True)
    torch.set_float32_matmul_precision('medium')  
    logger = None
    
    audio_model = BirdSoundModel(
        train_cfg, 
        audio_cfg, 
        paths, 
        in_channels=3).to(train_cfg.DEVICE)

    # Set up dataloaders
    audio_model._train_dataloader = dl_train
    audio_model._validation_dataloader = dl_val

    print(f'model loaded. Using {train_cfg.LOSS_FUNCTION_NAME} loss function')

    # swapped call-backs to REAL LIFE from LITTLE OWL dataset (this will break if im not training on binary little owl)
    early_stop_callback = EarlyStopping(
        monitor="rl_auc",
        min_delta=train_cfg.MIN_DELTA, 
        patience=train_cfg.PATIENCE, 
        verbose=True, 
        mode="max"
    )

    checkpoint_callback = ModelCheckpoint(
        save_top_k=train_cfg.KEEP_LAST,
        monitor="rl_auc",  # Match with early stopping metric
        mode="max",
        dirpath=checkpoint_dir,
        save_last=True,
        save_weights_only=True, 
        verbose=True,
        filename='binary_classifier-{epoch:02d}-{rl_auc:.3f}'
    )

    callbacks_to_use = [checkpoint_callback, early_stop_callback]

    trainer = pl.Trainer(
        val_check_interval=0.5,
        deterministic=True,
        max_epochs=train_cfg.EPOCHS,
        logger=logger,
        callbacks=callbacks_to_use,
        precision=train_cfg.PRECISION, 
        accelerator=train_cfg.GPU,
        reload_dataloaders_every_n_epochs=1,
        enable_progress_bar=True,       # Add this
        # log_every_n_steps=1,           # Add this
        # detect_anomaly=True,            # Add this for debugging,
        # gradient_clip_val=1.0,
        # gradient_clip_algorithm='norm',
    )

    print("Running trainer.fit")

    trainer.fit(audio_model, dl_train, dl_val)
           
    gc.collect()
    torch.cuda.empty_cache()

    return audio_model.get_my_metrics_list()

def extract_results(metrics, paths):
    train_losses = [x['train_loss'] for x in metrics]
    val_losses = [x['val_loss'] for x in metrics]
    
    # Binary classification metrics
    val_precision = [x['val_precision'] for x in metrics]
    val_recall = [x['val_recall'] for x in metrics]
    val_f1 = [x['val_f1'] for x in metrics]
    val_auc = [x['val_auc'] for x in metrics]
    
    # Plot metrics
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Loss plot
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Val Loss')
    if paths.EVAL_DIR and any('rl-loss' in x for x in metrics): # Check if rl-loss exists
        rl_losses = [x['rl-loss'] for x in metrics if 'rl-loss' in x]
        ax1.plot(rl_losses, label='Real-Life Eval Loss', linestyle=':', color='orange') # Add RL loss plot
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_ylim(0, 1)
    ax1.legend()

    # Metrics plot
    ax2.plot(val_precision, label='Precision', linestyle='--', color='green')
    ax2.plot(val_recall, label='Recall', linestyle='--', color='purple')
    ax2.plot(val_f1, label='F1', linestyle='--', color='blue')
    ax2.plot(val_auc, label='AUC', linestyle='--', color='red')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Score')

    # real life metrics plot
    if paths.EVAL_DIR:
        rl_precision = [x['rl-precision'] for x in metrics]
        rl_recall = [x['rl-recall'] for x in metrics]
        rl_f1 = [x['rl-f1'] for x in metrics]
        rl_auc = [x['rl-auc'] for x in metrics]
        ax2.plot(rl_precision, label='RL Precision', color='green')
        ax2.plot(rl_recall, label='RL Recall', color='purple')
        ax2.plot(rl_f1, label='RL F1', color='blue')
        ax2.plot(rl_auc, label='RL AUC', color='red')

    ax2.set_ylim(0, 1)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(Path(paths.out_dir) / f"training_metrics.jpg")
    plt.close()

def test_inference(model, audio_cfg, dir_path, save_to=None):
    """
    Load audio files from dir_path and run inference
    """
    # Convert dir_path to Path object if it's a string
    dir_path = Path(dir_path) if not isinstance(dir_path, Path) else dir_path
    
    print("\n" + Blue.S + f"testing on {dir_path} data..." + Blue.E)
    
    test_files = [f for f in dir_path.glob('*') if f.suffix in {'.ogg', '.flac', '.wav', '.mp3', '.WAV'} and not f.name.startswith('.')]
    if not test_files:
        print(Stop.S + "No audio files found in test directory!" + Stop.E)
        return
    print(f"Found {len(test_files)} audio files to test")
    
    # Put model in evaluation mode
    model.eval()
    device = model._device
    positives = []
    high_positives = []
    corrupted_files = []

    with torch.no_grad():
        for audio_path in test_files:
            try:
                # Load audio
                try:
                    y = load_sf(str(audio_path))
                except Exception as e:
                    print(f"Error loading {audio_path.name}: {e}")
                    corrupted_files.append(audio_path.name)
                    continue
                
                if len(y) == 0:
                    print(f"Empty audio file: {audio_path.name}")
                    corrupted_files.append(audio_path.name)
                    continue
                
                # Process audio into segments
                window_samples = int(audio_cfg.DURATION * audio_cfg.SR)

                for i in range(0, len(y), int(window_samples*audio_cfg.OVERLAP)):
                    try:
                        start_sample = i
                        if start_sample + window_samples > len(y):
                            window = crop_or_pad(y, window_samples, train='val')
                        else:
                            window = y[start_sample:start_sample + window_samples]
                        
                        # Normalize
                        window = Normalize(p=1)(window, sr=audio_cfg.SR)
                        
                        # Create spectrogram
                        if audio_cfg.PCEN:
                            image = compute_pcen(window, audio_cfg)
                        else:
                            image = compute_melspec(window, audio_cfg)
                        
                        # Prepare image
                        prep_image = PrepareImage(height=audio_cfg.N_MELS, width=audio_cfg.CHUNK_WIDTH)
                        image = prep_image.prep(image=image)['image']
                        image = mono_to_color(image, use_deltas=audio_cfg.USE_DELTAS)
                        
                        # Apply validation transforms (no augmentation)
                        image_transform = AbluTransforms(audio_cfg).valid
                        image = image_transform(image=image)['image']
                        image = image.transpose(2, 0, 1).astype(np.float32)
                        
                        # Convert to tensor and add batch dimension
                        image = torch.tensor(image).unsqueeze(0).to(device)
                        
                        # Get prediction
                        outputs = model(image)
                        prob = torch.sigmoid(outputs['logit']).item()
                        
                        if prob > 0.5:
                            start_seconds = i / audio_cfg.SR
                            in_formatted_time = str(datetime.timedelta(seconds=start_seconds))
                            positives.append([audio_path.name, prob, start_seconds, in_formatted_time])
                            if prob >= 0.98:
                                print(f"+h {audio_path.name} : {in_formatted_time} : {prob:.2f}")
                                high_positives.append([audio_path.name, prob, start_seconds, in_formatted_time])
                    except Exception as e:
                        print(f"Error processing segment in {audio_path.name} at position {i}: {e}")
                        continue  # Skip this segment but continue with others
                
            except Exception as e:
                print(f"Unexpected error with {audio_path.name}: {e}")
                corrupted_files.append(audio_path.name)
    
    if save_to:
        # Create directory if it doesn't exist
        save_dir = Path(save_to)
        save_dir.mkdir(exist_ok=True, parents=True)
        
        # Save positives to CSV
        positives_path = save_dir / "positive_results.csv"
        with open(positives_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['filename', 'probability', 'start_time_seconds', 'start_time'])
            for row in positives:
                writer.writerow(row)
        high_positives_path = save_dir / "high_positive_results.csv"
        with open(high_positives_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['filename', 'probability', 'start_time_seconds', 'start_time'])
            for row in high_positives:
                writer.writerow(row)
        
        # Save corrupted files list
        if corrupted_files:
            corrupted_path = save_dir / "corrupted_files.csv"
            with open(corrupted_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['filename'])
                for filename in corrupted_files:
                    writer.writerow([filename])
            print(Stop.S + f"List of {len(corrupted_files)} corrupted files saved to {corrupted_path}" + Stop.E)
        
        print(Blue.S + f"Results saved to {positives_path}" + Blue.E)

############################################# Prepare Data  ######################################
##################################################################################################

def load_training_data(paths, excluded_classes, use_secondary):
    '''Load the datframe from csv, clean any irrelevent secondary labels, and verify that all the files actually exist'''
    use_cols = ['filename', 'primary_label']  #'secondary_labels'  'latitude', 'longitude'
    in_df = pd.read_csv(paths.LABELS_PATH,  usecols=use_cols)
    in_df['filepath'] = paths.TRAIN_AUDIO_DIR + '/' + in_df['filename']
    in_df = in_df[~in_df['primary_label'].isin(excluded_classes)]
    unique_birds = sorted(in_df['primary_label'].unique()) 

    def remove_unused_birds(second_bird_list):
        return [string for string in second_bird_list if string in unique_birds]
    if use_secondary:
        in_df['secondary_labels'] = in_df['secondary_labels'].apply(ast.literal_eval)
        in_df['secondary_labels'] = in_df['secondary_labels'].apply(remove_unused_birds)

    # check that all the training samples in the dataframe exist.  Remove any rows that can't be found.
    original_length = len(in_df)
    training_samples = set([path for path in Path(paths.TRAIN_AUDIO_DIR).rglob('*') if path.suffix in {'.ogg', '.flac', '.wav', '.WAV', '.mp3'}])
    in_df['filepath'] = in_df['filepath'].apply(Path)
    in_df = in_df[in_df['filepath'].isin(training_samples)]
    in_df['filepath'] = in_df['filepath'].apply(str)
    new_length = len(in_df)
 
    if original_length > new_length:
        print(Blue.S + 'removed ' +  str(original_length - new_length) + ' leaving ' + str(new_length) + ' audio files' + Blue.E)
    else:
        print(Blue.S + 'All training samples found' + Blue.E)

    return in_df

def segment_training_data(in_df, audio_cfg):
    """
    Creates a new DataFrame with window information for each audio file.
    
    Args:
        df: Original DataFrame with filepath column
        window_duration: Duration of each window in seconds
        overlap: Overlap between windows (0.5 = 50% overlap)
    
    Returns:
        DataFrame with windows information
    """
    window_info_list = []
    sr = audio_cfg.SR
    window_samples = int(audio_cfg.DURATION * sr)
    hop_samples = int(window_samples * (1 - audio_cfg.OVERLAP))

    for idx, row in tqdm(in_df.iterrows(), total=len(in_df), desc="Creating windows"):
        try:
            # Get audio length
            audio_path = row['filepath']
            audio_len = len(load_sf(audio_path))
            
            # Calculate number of windows
            num_windows = max(1, (audio_len - window_samples) // hop_samples + 1)
            
            # Create entry for each window
            for window_idx in range(num_windows):
                start_sample = window_idx * hop_samples
                window_info = row.to_dict()  # Copy all original columns
                window_info.update({
                    'window_idx': window_idx,
                    'start_sample': start_sample,
                    'audio_len': audio_len
                })
                window_info_list.append(window_info)
                
        except Exception as e:
            print(f"Warning: Could not process {audio_path}: {str(e)}")
            continue
    
    windows_df = pd.DataFrame(window_info_list)
    return windows_df

def filter_by_location(df, limits=None):
    if limits is not None:
        df = df[
                (df['longitude'] >= limits['WEST']) &
                (df['longitude'] <= limits['EAST']) &
                (df['latitude'] >= limits['SOUTH']) &
                (df['latitude'] <= limits['NORTH'])]
    return df


def limit_max_per_class(df, max_per_class = None):
    '''Put an upper limit on class size to prevent extreme class imbalance'''
    if max_per_class is not None:
        class_counts = df['primary_label'].value_counts()
        classes_to_reduce = class_counts[class_counts > max_per_class].index
        def downsample_class(df, class_label, max_rows):
            df_class = df[df['primary_label'] == class_label]
            return df_class.sample(n=max_rows)
        df_list = [downsample_class(in_df, class_label, max_per_class) if class_label in classes_to_reduce 
                else df[df['primary_label'] == class_label]
                for class_label in df['primary_label'].unique()]
        df = pd.concat(df_list)
    return df


def split_classes_by_size(df, threshold):
    '''Temporarily drop any super rare classes from the dataframe, so they don't end up 
    loosing precious samples from training due to location or splitting.'''
    mask = df['primary_label'].map(df['primary_label'].value_counts()) > threshold
    common_df = df[mask]
    common_df = common_df.reset_index(drop=True)
    mask = df['primary_label'].map(df['primary_label'].value_counts()) <= threshold
    rare_df = df[mask]
    rare_df = rare_df.reset_index(drop=True)
    return common_df, rare_df


def duplicate_rare_rows(df, min_samples):
    '''Upsample the super-rare classes to some minimum'''
    value_counts = df['primary_label'].value_counts()
    duplication_needed = {label: min_samples - count for label, count in value_counts.items()}

    duplicated_rows = []
    for label, count in duplication_needed.items():
        label_rows = df[df['primary_label'] == label]
        num_duplicates = count // len(label_rows)  # Number of full duplications needed
        remainder = count % len(label_rows)        # Remaining duplications needed

        if num_duplicates > 0:
            duplicated_full_sets = pd.concat([label_rows] * num_duplicates, ignore_index=True)
            duplicated_rows.append(duplicated_full_sets)

        if remainder > 0:
            duplicated_remainder = label_rows.sample(n=remainder, replace=True)
            duplicated_rows.append(duplicated_remainder)

    df = pd.concat([df] + duplicated_rows, ignore_index=True)
    final_counts = df['primary_label'].value_counts()
    print(final_counts[-10:])
    return df


def train_val_split(in_df, data_cfg, audio_cfg):
    '''Split training and validation samples, but limiting the max in the validation set
    and also not using any super-rare sample in the validation set'''
    # skf =StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=2024)
    # target = common_df['primary_label'] 
    if data_cfg.DO_WINDOWING_BEFORE_DATASET_CREATION:
        prepared_df = segment_training_data(in_df, audio_cfg)
    else: prepared_df = in_df
    skf = StratifiedKFold(n_splits=data_cfg.N_FOLDS, shuffle=True, random_state=2024)
    target = prepared_df['primary_label']

    for train_index, val_index in skf.split(prepared_df, target):
        tn_df, val_df = prepared_df.iloc[train_index].copy(), prepared_df.iloc[val_index].copy()
        
    return tn_df, val_df

#unused
def multi_binarize(df, unique_birds, secondary_weights):
    keep_cols = ['primary_label', 'filepath']
    # clean labels (breaks secondary?)
    df['secondary_labels'] = df['secondary_labels'].apply(lambda x: [label for label in x if label not in ['[', ']']])
    mlb = MultiLabelBinarizer(classes=unique_birds)
    df_primary = pd.concat([df, pd.get_dummies(df['primary_label']).astype('uint8')], axis=1)
    missing_birds = list(set(unique_birds).difference(list(df.primary_label.unique())))
    df_primary = pd.concat([df_primary, pd.DataFrame(0, index=df_primary.index, columns=missing_birds)], axis=1)
    df_primary = df_primary[unique_birds] # To synchronise the column order
    #df['combined_labels'] = df.apply(lambda row: [row['primary_label']] + row['secondary_labels'], axis=1)
    secondary_array = mlb.fit_transform(df['secondary_labels']).astype('uint8')
    combined_array = secondary_array * secondary_weights + df_primary[unique_birds].values
    label_df = pd.DataFrame(combined_array, columns=unique_birds)
    df = df.reset_index(drop=True)
    df = pd.concat([df[keep_cols], label_df], axis=1)

    return df


def encode_data(train_df, val_df):
    """
    Convert labels to binary format
    """    
    # Convert labels to binary
    train_df['label'] = train_df['label'].astype('uint8')
    val_df['label'] = val_df['label'].astype('uint8')
    
    return train_df, val_df


def prepare_binary_data(df):
    """
    Convert labels to binary format while preserving window information.
    """
    # Create new DataFrame with required columns
    binary_df = pd.DataFrame()
    
    # Preserve all the window-related columns
    window_cols = ['filepath', 'window_idx', 'start_sample', 'audio_len']
    for col in window_cols:
        if col in df.columns:
            binary_df[col] = df[col]
    
    # Add binary label
    binary_df['label'] = (df['primary_label'] == 1).astype(int)
    
    return binary_df

############################################# Main Script  #######################################
##################################################################################################

def main_training_function(use_case=None):
    """Main function to run the training pipeline with given parameters"""
    if use_case is None:
        # Load default use_case if none provided
        with open('classifiers/use_case.yaml') as f:
            use_case = yaml.safe_load(f)

    train_cfg = TrainingParameters(options=use_case)
    data_cfg = NzBirdData()
    audio_cfg = DefaultAudio()
    paths = FilePaths(options=use_case)

    Path(paths.out_dir).mkdir(parents=True, exist_ok=True)
    Path(paths.temp_dir).mkdir(parents=True, exist_ok=True)
    Path(paths.chkpt_dir).mkdir(parents=True, exist_ok=True)
    Path(paths.model_deploy).mkdir(parents=True, exist_ok=True)

    print(Blue.S + f'GPU set to: ' + Blue.E + train_cfg.GPU)
    # print(Blue.S + 'CPUs for available for dataloading: ' + Blue.E + str(train_cfg.NUM_WORKERS))

    #This could all be moved into a data prep class
    in_df = load_training_data(paths, data_cfg.EXCLUDED_CLASSES, data_cfg.USE_SECONDARY) #MAYBE GET RID OF USE_SECONDARY?  HANDLE WITH WEIGHTS INSTEAD
    # in_df = filter_by_location(in_df, limits=data_cfg.SPATIAL_LIMITS)
    # in_df = limit_max_per_class(in_df, data_cfg.MAX_PER_CLASS)
    unique_birds = list(in_df['primary_label'].unique())
    print(f'Unique birds: {unique_birds}')
    # common_df, rare_df = split_classes_by_size(in_df, data_cfg.RARE_THRESHOLD)  
    # rare_df = duplicate_rare_rows(rare_df, data_cfg.RARE_THRESHOLD)
    # print(Blue.S + 'after duplicating for rarity the dataframe has ' + Blue.E + str(len(common_df)) + ' common samples and ' + str(len(rare_df)) + ' rare samples')

    train_df, val_df = train_val_split(in_df, data_cfg, audio_cfg)
    train_df = prepare_binary_data(train_df)
    val_df = prepare_binary_data(val_df)

    train_df_0, val_df = encode_data(
        train_df, 
        val_df
    )

    augmentation_updates = [train_cfg.FIRST_AUGMENTATION_UPDATE, train_cfg.SECOND_AUGMENTATION_UPDATE]

    dl_train, dl_val, ds_train, ds_val = get_dataloaders(
        train_df_0, 
        val_df,
        paths,
        audio_cfg,
        batch_size=train_cfg.BATCH_SIZE,
        num_workers=train_cfg.NUM_WORKERS,
        weighted_sampling=train_cfg.WEIGHTED_SAMPLING,
        augmentation_updates=augmentation_updates,
        device=train_cfg.DEVICE
    )

    save_naming_scheme(train_df, val_df, paths.bird_map_for_model)

    save_model_config(paths, audio_cfg, train_cfg)

    metrics = run_training(dl_train, dl_val, paths, data_cfg, train_cfg, audio_cfg, paths.chkpt_dir)
    extract_results(metrics, paths)
    last_path = save_models(paths, train_cfg, audio_cfg, unique_birds)

    # Clean up the temporary checkpoint directory after saving final models
    try:
        checkpoint_dir_path = Path(paths.chkpt_dir)
        if checkpoint_dir_path.exists() and checkpoint_dir_path.is_dir():
            shutil.rmtree(checkpoint_dir_path)
            print(f"Successfully removed temporary checkpoint directory: {checkpoint_dir_path}")
        else:
             print(f"Temporary checkpoint directory not found or not a directory: {checkpoint_dir_path}")
    except Exception as e:
        print(f"Error removing temporary checkpoint directory {paths.chkpt_dir}: {e}")

    final_model = BirdSoundModel(train_cfg, audio_cfg, paths, in_channels=3)
    # Ensure last_path points to a file in the Results directory now
    last_path_in_results = Path(paths.out_dir) / Path(last_path).name
    if last_path_in_results.exists():
        model_state_dict = torch.load(last_path_in_results)
        final_model.load_state_dict(model_state_dict)
    else:
        print(f"Warning: Could not find final model state dict at {last_path_in_results} to load.")
    print(Stop.S + 'Model loaded OK' + Stop.E)
    print('')

    return metrics

if __name__ == "__main__":
    main_training_function()