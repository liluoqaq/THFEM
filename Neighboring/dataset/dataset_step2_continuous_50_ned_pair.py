import os
import sys
sys.path.append(os.getcwd())
from skimage import io, img_as_float32
from skimage.color import gray2rgb
from sklearn.model_selection import train_test_split
from imageio import mimread
from os.path import dirname, join, basename, isfile
import json
import gzip
import copy
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
from Utils.augmentation import AllAugmentationTransform
import glob
import torch
import torchaudio
import random
import glob
import soundfile as sf
import math
import cv2
import yaml
from argparse import ArgumentParser
from scipy.io import wavfile
import python_speech_features
# import pyworld
from scipy.interpolate import interp1d

### wav2lip read wav
from scipy import signal
import librosa

syncnet_mel_step_size = 16

###### CHANGE THE MEAD PATHS HERE ######

MEL_PARAMS_25 = {"n_mels": 80, "n_fft": 2048, "win_length": 540, "hop_length": 540}
train_video_list = "info/train.txt"
raleation_align = "info/re_align.json"
class FramesWavsDatasetMEL25VoxBoxQG2ImgPrompt(Dataset):
    """
    Dataset of videos and wavs, each video can be represented as:
      - an image of concatenated frames
      - wavs
      - folder with all frames
    """

    def __init__(self,
                 root_dir,
                 mead_path,
                 real_gt,
                 frame_shape=(256, 256, 3),
                 id_sampling=False,
                 is_train=True,
                 random_seed=0,
                 pairs_list=None,
                 augmentation_params=None,
                 syncnet_T=24,
                 region=10,
                 use_vox=False):
        self.root_dir = root_dir
        self.mead_path = mead_path
        self.poseimg_path = mead_path
        self.real_gt = real_gt

        self.frame_shape = tuple(frame_shape)
        self.pairs_list = pairs_list
        self.id_sampling = id_sampling
        with open(raleation_align,'r') as f:
            self.pair_info = json.loads(f.read())
            f.close()

        self.length = 0
        if is_train:
            with open(train_video_list,'r') as f:
                train_videos = f.read().splitlines()
                f.close()
            self.videos = train_videos
            self.length += len(self.videos)
            print(self.length)
            self.length = self.length


        self.is_train = is_train
        self.latent_dim = 16
        self.num_w = 5

        if self.is_train:
            self.transform = AllAugmentationTransform(**augmentation_params)
        else:
            self.transform = None
        self.emo_label = [
            'angry', 'disgusted', 'fear', 'happy', 'neutral', 'sad',
            'surprised'
        ]
        self.emo_label_full = [
            'angry', 'disgusted', 'fear', 'happy', 'neutral', 'sad',
            'surprised'
        ]
        self.to_melspec = torchaudio.transforms.MelSpectrogram(**MEL_PARAMS_25)
        self.mean, self.std = -4, 4
        self.syncnet_T = syncnet_T
        self.region = region

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.is_train:
            return self.getitem_neu(idx)
        else:
            return self.getitem_neu(idx)

    def get_frame_id(self, frame):
        return int(basename(frame).split('.')[0])

    def get_window(self, start_frame, he_driving, poseimg):
        start_id = self.get_frame_id(start_frame)
        vidname = dirname(start_frame)

        he_d = {'yaw': [], 'pitch': [], 'roll': [], 't': [], 'exp': []}

        window_fnames = []
        for frame_id in range(start_id, start_id + self.syncnet_T):

            frame = join(vidname, '{:06}.png'.format(frame_id))
            if not isfile(frame):
                return None
            window_fnames.append(frame)

        for k in he_driving.keys():
            he_driving[k] = torch.from_numpy(he_driving[k])
            he = torch.cat(
                [he_driving[k][start_id - 1:start_id + self.syncnet_T - 1]])
            he_d[k] = he

        # draw headpose
        poseboxs = []
        for i in range((start_id - self.num_w),
                       (start_id + self.syncnet_T + self.num_w)):
            if i < 0:
                poseboxs.append(poseimg[0])
            elif i >= poseimg.shape[0]:
                poseboxs.append(poseimg[-1])
            else:
                poseboxs.append(poseimg[i])



        return window_fnames, he_d, poseboxs, start_id

    def crop_audio_phoneme_window(self, mel, poses, deeps, start_frame,
                                  num_frames):

        start_frame_num = self.get_frame_id(start_frame)
        pad = torch.zeros((mel.shape[0]))

        audio_frames = []
        pose_frames = []
        deep_frames = []

        zero_pad = np.zeros([self.num_w, deeps.shape[1],
                             deeps.shape[2]])  
        deeps = np.concatenate((zero_pad, deeps[:num_frames], zero_pad),
                               axis=0) 

        for rid in range(
                start_frame_num, start_frame_num +
                self.syncnet_T):  
            audio = []
            for i in range(rid - self.num_w, rid + self.num_w + 1):
                if i < 0:
                    audio.append(pad)
                elif i >= num_frames:
                    audio.append(pad)
                else:
                    audio.append(mel[:, i])
            deep_frames.append(deeps[rid:rid + 2 * self.num_w + 1])

            audio_frames.append(torch.stack(audio, dim=1))

            pose_frames.append(
                poses[(rid -
                       start_frame_num):(rid + 2 * self.num_w + 1 -
                                         start_frame_num)])  

        audio_f = torch.stack(audio_frames, dim=0)
        poses_f = torch.from_numpy(np.array(pose_frames, dtype=np.float32))
  
        deep_frames = torch.from_numpy(np.array(deep_frames)).to(torch.float)

        return audio_f, poses_f, deep_frames

    def get_sync_mel(self, wav, start_frame):

        def preemphasis(wav, k, preemphasize=True):
            if preemphasize:
                return signal.lfilter([1, -k], [1], wav)
            return wav

        def _stft(y):
            return librosa.stft(y=y, n_fft=800, hop_length=200, win_length=800)

        def _amp_to_db(x):
            min_level = np.exp(-100 / 20 * np.log(10))
            return 20 * np.log10(np.maximum(min_level, x))

        def _linear_to_mel(spectogram):
            _mel_basis = _build_mel_basis()
            return np.dot(_mel_basis, spectogram)

        def _build_mel_basis():
            # assert 7600 <= 16000 // 2
            return librosa.filters.mel(16000,
                                       800,
                                       n_mels=80,
                                       fmin=55,
                                       fmax=7600)

        def _normalize(S, max_abs_value=4., min_level_db=-100):
            return np.clip(
                (2 * max_abs_value) * ((S - min_level_db) / (-min_level_db)) -
                max_abs_value, -max_abs_value, max_abs_value)

        D = _stft(preemphasis(wav, 0.97))
        S = _amp_to_db(_linear_to_mel(np.abs(D))) - 20

        S = _normalize(S).T

        # fps = 25
        fps = 30   #修改
        start_idx = int(80. * (start_frame / float(fps)))
        end_idx = start_idx + syncnet_mel_step_size
        return S[start_idx:end_idx, :]

    def _load_tensor(self, data):
        wave_path = data
        wave, sr = sf.read(wave_path)
        wave_tensor = torch.from_numpy(wave).float()
        return wave_tensor

    def remove_reconstruction(self, out_put):
        for k in out_put.keys():
            print(1)

    def getitem_neu(self, idx):
        while 1:
            name = self.videos[idx]
            path = os.path.join(self.root_dir, name)
            video_name = os.path.basename(path)
            vsp = video_name.split('_')

            out = {}

            out['y_trg'] = self.emo_label.index(vsp[-1])
            z_trg = torch.randn(self.latent_dim)
            out['z_trg'] = z_trg

            deep_path = f'{self.mead_path}/deepfeature32/{video_name}.npy'  #首位为帧数
            deeps = np.load(deep_path)

            wave_path = f'{self.mead_path}/wav_16000/{video_name}.wav'
            out['wave_path'] = wave_path
            wave_tensor = self._load_tensor(wave_path)
            if len(wave_tensor.shape) > 1:
                wave_tensor = wave_tensor[:, 0]
            mel_tensor = self.to_melspec(wave_tensor)
            mel_tensor = (torch.log(1e-5 + mel_tensor) -
                          self.mean) / self.std  #(seq,num)

            lable_index = self.emo_label.index(vsp[-1])

            frames = os.listdir(path)
            num_frames = len(frames)
            num_frames = min(num_frames, len(deeps),
                             mel_tensor.shape[1])  
            if num_frames - self.region - self.syncnet_T + 1 <= 1:
                idx += 1
                idx = idx % len(self.videos)
                continue



            source_range = np.arange(
                0, num_frames - self.region - self.syncnet_T + 1)
            # print(source_range)
            # print("\n")
            source_idx = np.random.choice(source_range)
            source_frame = join(path, '{:06}.png'.format(source_idx))
            source_path = source_frame[:-11]
            source_latent = np.load(source_path.replace('images', 'latent') +
                                    '.npy',
                                    allow_pickle=True)
            video_array_source = img_as_float32(io.imread(source_frame))

            # neutral source latent with pretrained
            he_source = {}
            for k in source_latent[1].keys():
                he_source[k] = source_latent[1][k][source_idx - 1]
            out['he_source'] = he_source

            out['source'] = video_array_source.transpose((2, 0, 1))

            # select gt frames

            if num_frames - self.syncnet_T + 1 <= 0:
                # print(num_frames)
                idx += 1
                idx = idx % len(self.videos)
                continue

            region_list = list(range(1, self.region + 1))

            driving_idx = np.random.choice(region_list, replace=True,
                                           size=1)[0]
            frame_idx = source_idx + driving_idx
            choose = join(path, '{:06}.png'.format(frame_idx))


            driving_latent = np.load(path.replace('images', 'latent') + '.npy',
                                     allow_pickle=True)
            he_driving = driving_latent[1]

            ### get syncmel refer to: wav2lip
            ## The syncnet_T length should be 5 or don't use the sync_mel_tensor
            ## fps: 25 wav: 16k Hz
            ## sync_mel_tensor shape: [16, 80]
            ## cost: 0.05s
            sync_mel_tensor = self.get_sync_mel(wave_tensor, frame_idx).T  #根据第一帧图像得到sync
            out['sync_mel'] = np.expand_dims(sync_mel_tensor,
                                             axis=0).astype(np.float32)

            fposeimg = gzip.GzipFile(
                f'{self.poseimg_path}/poseimg/{video_name}.npy.gz', "r")
            poseimg = np.load(fposeimg)

            ### bboxs files extracted
            bboxs = np.load(f'{self.mead_path}/bboxs/{video_name}.npy',
                            allow_pickle=True)

            try:
                window_fnames, he_d, poses, source_idx = self.get_window(
                    choose, he_driving,
                    poseimg)  
            except:
                print(source_idx, choose)
                print(choose, path)
                idx += 1
                idx = idx % len(self.videos)
                continue

            out['he_driving'] = he_d

            ### read img
            ## cost 0.2s for 5 frames
            window = []
            boxs = []
            all_read = True
            count = 0
            for fname in sorted(window_fnames):
                img = img_as_float32(io.imread(fname)).transpose((2, 0, 1))
                box = bboxs[frame_idx + count]
                if box is None:
                    all_read = False
                    break
                if img is None:
                    all_read = False
                    break
                window.append(img)
                boxs.append(box)
                count += 1
            real_gt_list,real_su = self.get_real_gt(driving_list=window_fnames)
            real_pair_gts = []
            if  real_su:
                for fname in sorted(real_gt_list):
                    real_gt = img_as_float32(io.imread(fname)).transpose((2, 0, 1))
                    real_pair_gts.append(real_gt)
            if not all_read or not real_su:
                idx += 1
                idx = idx % len(self.videos)

            out['fake_driving'] = np.stack(window, axis=0)
            out['driving'] = np.stack(real_pair_gts, axis=0)
            out['bboxs'] = np.stack(boxs, axis=0)

            mel, poses_f, deep_frames = self.crop_audio_phoneme_window(
                mel_tensor, poses, deeps, choose, num_frames)
            out['mel'] = mel.unsqueeze(1)
            out['pose'] = poses_f
            out['name'] = video_name
            out['deep'] = deep_frames

            return out

    def get_real_gt(self,driving_list):
        reat_gt = []
        for driving in driving_list:
            d_info = driving.split("/")
            frame_id =int(d_info[-1][:-4])
            vid =d_info[-2]
            act,emo,num,edit_emo = vid.split("_")
            source_vid = f'{act}_{emo}_{num}'
            if edit_emo == "neutral":
                tgt_frame = f'{self.real_gt}/{source_vid}/{d_info[-1]}'
            else:
                tgt_info = self.pair_info[source_vid][edit_emo]
                tgt_vid = tgt_info["path"]
                relation = tgt_info["relation"]
                lacation = (random.sample(list(np.where(np.array(relation[0]) == frame_id)),1)[0]).tolist()

                if len(lacation) != 0:
                    tgt_index= "{:06d}.png".format(relation[1][lacation[0]])
                    tgt_frame = f'{self.real_gt}/{tgt_vid}/{tgt_index}'
                else:
                    return reat_gt,False     
            if os.path.exists(tgt_frame):
                reat_gt.append(tgt_frame)
            else:
                return reat_gt,False
        return reat_gt , True
        

