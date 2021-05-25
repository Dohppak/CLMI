import os
import random
import torch
import pickle
import pandas as pd
import numpy as np
from pydub import AudioSegment
from torch.utils.data import Dataset, DataLoader
import torchaudio
from PIL import Image


class CLMIDataset(Dataset):
    def __init__(self, feature_path, split, transform=None):
        self.feature_path = feature_path
        self.transform = transform
        self.split = split
        self.get_fl()
        
    def get_fl(self):
        if self.split == "TRAIN":
            self.fl = torch.load("./dataset/split/tr_list.pt")
        elif self.split == "VALID":
            self.fl = torch.load("./dataset/split/va_list.pt")
        elif self.split == "TEST":
            self.fl = torch.load("./dataset/split/te_list.pt")
        else:
            print("Split should be one of [TRAIN, VALID, TEST]")
    
    def __getitem__(self, index):
        audio_path = self.fl[index]
        thumbnail_path = audio_path.replace("audio","thumbnail").replace(".mp3",".jpg")

        metadata = torchaudio.info(audio_path)
        duration_sec = (metadata[0].length / 2) / metadata[0].rate
        rand_idx = random.choice(range(100))

        audio = AudioSegment.from_file(audio_path , "mp3", start_second=rand_idx, duration=3).set_channels(1)._data
        image = Image.open(thumbnail_path).convert('RGB')
        audio = np.frombuffer(audio, dtype=np.int16) / 32768
        audio = torch.Tensor(audio).unsqueeze(0)
        return audio, self.transform(image)
    
    def __len__(self):
        return len(self.fl)