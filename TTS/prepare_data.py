import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import os
from utils import get_spectrograms

from .config import sample_rate, data_path

import librosa

import argparse

def get_parser():
    parser = argparse.ArgumentParser(description="n_samples")
    parser.add_argument("--n_samples", type=int, default=-1, help="number of samples to consider")
    return parser

class PrepareDataset(Dataset):
    """LJSpeech dataset."""

    def __init__(self, csv_file, root_dir):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the wavs.

        """
        self.landmarks_frame = pd.read_csv(csv_file, sep='|', header=None)
        self.root_dir = root_dir

    def load_wav(self, filename):
        return librosa.load(filename, sr=sample_rate)

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        wav_name = os.path.join(self.root_dir, self.landmarks_frame.loc[idx, 0]) + '.wav'
        mel, mag = get_spectrograms(wav_name)
        
        np.save(wav_name[:-4] + '.pt', mel)
        np.save(wav_name[:-4] + '.mag', mag)

        sample = {'mel':mel, 'mag': mag}

        return sample
    
if __name__ == '__main__':
    parser = get_parser()
    params = parser.parse_args()

    dataset = PrepareDataset(os.path.join(data_path,'metadata.csv'), os.path.join(data_path,'wavs'))
    
    if params.n_samples < 0 : 
        dataloader = DataLoader(dataset, batch_size=1, drop_last=False, num_workers=8)
    else :
        indices = list(range(len(dataset)))
        sampler = SubsetRandomSampler(indices[:params.n_samples])
        dataloader = DataLoader(dataset, batch_size=1, sampler = sampler, drop_last=False, num_workers=8)
        
    from tqdm import tqdm
    pbar = tqdm(dataloader)
    for d in pbar:
        pass
        
    torch.save(params.n_samples, data_path+'/.n_samples')
    
