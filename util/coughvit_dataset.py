from curses import meta
import os
import cv2
import pickle
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

import librosa
import torch
from torch.utils.data import Dataset
from copy import deepcopy
from PIL import Image

from .icbhi_util import get_annotations, save_image, generate_fbank, get_individual_cycles_librosa, split_pad_sample, generate_mel_spectrogram, concat_augmentation
from .icbhi_util import get_individual_cycles_torchaudio, cut_pad_sample_torchaudio
from .augmentation import augment_raw_audio

import librosa
import torch
import torchaudio
from torchaudio import transforms as T


class COUGHVIDdataset(Dataset):
    def __init__(self, train_flag, transform, args, metadata_path, print_flag=True, mean_std=False):
  
        self.data_folder = os.path.join("public_dataset")
        self.train_flag = train_flag
        self.split = 'train' if train_flag else 'test'
        self.transform = transform
        self.args = args
        self.mean_std = mean_std

        # parameters for spectrograms
        self.sample_rate = args.sample_rate
        self.desired_length = args.desired_length
        self.pad_types = args.pad_types
        self.nfft = args.nfft
        self.hop = self.nfft // 2
        self.n_mels = args.n_mels
        self.f_min = 50
        self.f_max = 2000
        self.dump_images = False

        # ==========================================================================
        data_pandas = pd.read_csv(metadata_path, sep='\t')
        sample_data = []
        for index, row in tqdm(data_pandas.iterrows(), total=data_pandas.shape[0]):
            fpath = os.path.join(self.data_folder, row['uuid'] + '.wav')
                
            sr = librosa.get_samplerate(fpath)
            data, _ = torchaudio.load(fpath)
            
            if sr != self.sample_rate:
                resample = T.Resample(sr, self.sample_rate)
                data = resample(data)

            fade_samples_ratio = 16
            fade_samples = int(self.sample_rate / fade_samples_ratio)

            fade = T.Fade(fade_in_len=fade_samples, fade_out_len=fade_samples, fade_shape='linear')
            data = fade(data)

            sample_data.append((data, row['status']))
        
        padded_sample_data = []
        for data, label in sample_data:
            data = cut_pad_sample_torchaudio(data, args)
            padded_sample_data.append((data, label))
        
        self.audio_data = padded_sample_data  # each sample is a tuple with (audio_data, label, filename)
        self.labels = []
        
        self.class_nums = np.zeros(args.n_cls)
        for sample in self.audio_data:
            self.class_nums[sample[1]] += 1
            self.labels.append(sample[1])
        self.class_ratio = self.class_nums / sum(self.class_nums) * 100
        
        if print_flag:
            print('[Preprocessed {} dataset information]'.format(self.split))
            print('total number of audio data: {}'.format(len(self.audio_data)))
            for i, (n, p) in enumerate(zip(self.class_nums, self.class_ratio)):
                print('Class {} {:<9}: {:<4} ({:.1f}%)'.format(i, '('+args.cls_list[i]+')', int(n), p))    
        # ==========================================================================
        """ convert mel-spectrogram """
        self.audio_images = []
        for index in tqdm(range(len(self.audio_data))):
            audio, label = self.audio_data[index][0], self.audio_data[index][1]

            audio_image = []
            # self.aug_times = 1 + 5 * self.args.augment_times  # original + five naa augmentations * augment_times (optional)
            for aug_idx in range(self.args.raw_augment+1): 
                if aug_idx > 0:
                    if self.train_flag and not mean_std:
                        audio = augment_raw_audio(audio, self.sample_rate, self.args)
                        
                        # "RespireNet" version: pad incase smaller than desired length
                        # audio = split_pad_sample([audio, 0,0], self.desired_length, self.sample_rate, types=self.pad_types)[0][0]

                        # "SCL" version: cut longer sample or pad sample
                        audio = cut_pad_sample_torchaudio(torch.tensor(audio), args)
                    else:
                        audio_image.append(None)
                        continue
                
                image = generate_fbank(audio, self.sample_rate, n_mels=self.n_mels)
                # image = generate_mel_spectrogram(audio.squeeze(0).numpy(), self.sample_rate, n_mels=self.n_mels, f_max=self.f_max, nfft=self.nfft, hop=self.hop, args=self.args) # image [n_mels, 251, 1]

                # blank region clipping from "RespireNet" paper..
                if self.args.blank_region_clip:     
                    image_copy = deepcopy(generate_fbank(audio, self.sample_rate, n_mels=self.n_mels))
                    # image_copy = deepcopy(generate_mel_spectrogram(audio.squeeze(0).numpy(), self.sample_rate, n_mels=self.n_mels, f_max=self.f_max, nfft=self.nfft, hop=self.hop, args=self.args)) # image [n_mels, 251, 1]                    

                    image_copy[image_copy < 10] = 0
                    for row in range(image_copy.shape[0]):
                        black_percent = len(np.where(image_copy[row,:] == 0)[0]) / len(image_copy[row,:])
                        # if there is row that is filled by more than 20% regions, stop and remember that `row`
                        if black_percent < 0.80:
                            break

                    # delete black percent
                    if row + 1 < image.shape[0]:
                        image = image[row+1:,:,:]
                    image = cv2.resize(image, (image.shape[1], self.n_mels), interpolation=cv2.INTER_LINEAR)
                    image = image[..., np.newaxis]

                audio_image.append(image)
            self.audio_images.append((audio_image, label))
            
            if self.dump_images:
                save_image(audio_image, './')
                self.dump_images = False

        self.h, self.w, _ = self.audio_images[0][0][0].shape
        # ==========================================================================

    def __getitem__(self, index):
        audio_images, label = self.audio_images[index][0], self.audio_images[index][1]

        if self.args.raw_augment and self.train_flag and not self.mean_std:
            aug_idx = random.randint(0, self.args.raw_augment)
            audio_image = audio_images[aug_idx]
        else:
            audio_image = audio_images[0]
        
        if self.transform is not None:
            audio_image = self.transform(audio_image)
        
        return audio_image, label

    def __len__(self):
        return len(self.audio_data)