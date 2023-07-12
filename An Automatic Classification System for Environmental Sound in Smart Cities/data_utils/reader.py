import random
import sys

import warnings
from datetime import datetime

import torch
from paddle.dataset.image import cv2

warnings.filterwarnings("ignore")

import librosa
import numpy as np
from torch.utils.data import Dataset


# 加载并预处理音频
def load_audio(audio_path, feature_method='features_concat', mode='train', sr=16000, chunk_duration=4, augmentors=None):
    # 读取音频数据
    wav, sr_ret = librosa.load(audio_path, sr=sr)
    a=np.zeros(64000)
    cnt = 0
    if mode == 'train':
        # 随机裁剪
        num_wav_samples = wav.shape[0]
        # 数据太短不利于训练
        num_chunk_samples = int(chunk_duration * sr)
        if num_wav_samples > num_chunk_samples + 1:
            wav = wav[:num_chunk_samples]
        else:
            new_wav = np.zeros(num_chunk_samples)
            new_wav[:num_wav_samples] = wav[:num_wav_samples]
            # for i in range(num_wav_samples, num_chunk_samples):
            #     new_wav[i] = wav[cnt]
            #     new_wav[i] = wav[cnt]
            #     cnt = cnt + 1
            #     if cnt == num_wav_samples:
            #         cnt = 0
            for i in range(num_wav_samples, num_chunk_samples):
                new_wav[i] = a[cnt]
                cnt = cnt + 1
                if cnt == num_wav_samples:
                    cnt = 0
            wav = new_wav
            # print(wav.shape)
        # if num_wav_samples > num_chunk_samples + 1:
        #    start = random.randint(0, num_wav_samples - num_chunk_samples - 1)
        #   stop = start + num_chunk_samples
        #  wav = wav[start:stop]
        # 对每次都满长度的再次裁剪
        # if random.random() > 0.5:
        #     wav[:random.randint(1, sr // 4)] = 0
        #     wav = wav[:-random.randint(1, sr // 4)]
        # else:
        #     t=num_chunk_samples - num_wav_samples
        #     for i in range(t):
        #         wav=np.c_(wav,0)
        # 数据增强

        # wav = new_wav

        if augmentors is not None:
            for key, augmentor in augmentors.items():
                if key == 'specaug': continue
                wav = augmentor(wav)
            #print(wav.shape)

    elif mode == 'eval':
        # 为避免显存溢出，只裁剪指定长度
        num_wav_samples = wav.shape[0]
        # 数据太短不利于训练
        num_chunk_samples = int(chunk_duration * sr)
        if num_wav_samples > num_chunk_samples + 1:
            wav = wav[:num_chunk_samples]
        else:
            new_wav = np.zeros(num_chunk_samples)
            new_wav[:num_wav_samples] = wav[:num_wav_samples]
            # for i in range(num_wav_samples , num_chunk_samples):
            #     new_wav[i] = wav[cnt]
            #     cnt = cnt + 1
            #     if cnt == num_wav_samples:
            #         cnt = 0
            for i in range(num_wav_samples, num_chunk_samples):
                new_wav[i] = a[cnt]
                cnt = cnt + 1
                if cnt == num_wav_samples:
                    cnt = 0
            wav = new_wav

    if mode == 'infer':
        # 随机裁剪
        num_wav_samples = wav.shape[0]
        # 数据太短不利于训练
        num_chunk_samples = int(chunk_duration * sr)
        if num_wav_samples > num_chunk_samples + 1:
            wav = wav[:num_chunk_samples]
        else:
            new_wav = np.zeros(num_chunk_samples)
            new_wav[:num_wav_samples] = wav[:num_wav_samples]
            # for i in range(num_wav_samples, num_chunk_samples):
            #     new_wav[i] = wav[cnt]
            #     cnt = cnt + 1
            #     if cnt == num_wav_samples:
            #         cnt = 0
            for i in range(num_wav_samples, num_chunk_samples):
                new_wav[i] = a[cnt]
                cnt = cnt + 1
                if cnt == num_wav_samples:
                    cnt = 0
            wav = new_wav


        # else:
        #     wav.extend(np.zeros(shape=[num_chunk_samples - num_wav_samples], dtype=np.float32))
        #     wav = np.array(wav)
        #     获取音频特征

    if feature_method=='features_fusion':
        # 计算梅尔频谱
        features1 = librosa.feature.melspectrogram(y=wav, sr=sr, n_fft=400, n_mels=80, hop_length=160, win_length=400)
        features2 = librosa.power_to_db(features1, ref=1.0, amin=1e-10, top_db=None) #取对数
        #features2 = librosa.power_to_db(features1_1, ref=1.0, amin=1e-10, top_db=None)
        mean = np.mean(features2, 0, keepdims=True)
        std = np.std(features2, 0, keepdims=True)
        features_mel= (features2 - mean) / (std + 1e-5)
        # print(features_mel.shape)
        #features_mel=np.pad(features3, ((60, 61), (0, 1)), 'constant', constant_values=0)
        # 计算声谱图
        # features4 = librosa.power_to_db(features, ref=1.0, amin=1e-10, top_db=None)
        # features4 = librosa.power_to_db(features3, ref=1.0, amin=1e-10, top_db=None)
        # 归一化
        # features4 = librosa.feature.chroma_cqt(y=wav, sr=16000, C=None, hop_length=160, fmin=None, threshold=0.0, tuning=None, n_chroma=12, n_octaves=7, window=None, bins_per_octave=36, cqt_mode='full')
        # librosa.feature.chroma_stft(y=wav, sr=16000, S=None, norm=inf, n_fft=2048, hop_length=512, win_length=None,window='hann', center=True, pad_mode='constant', tuning=None, n_chroma=12)
        # mean1 = np.mean(features4, 0, keepdims=True)
        # std1 = np.std(features4, 0, keepdims=True)
        # features_CQT = (features4 - mean1) / (std1 + 1e-5)
        # features_stack= torch.stack((features_mel,features_CQT), dim=2)
        linear = librosa.stft(y=wav, n_fft=400, win_length=400, hop_length=160)
        # features, _ = librosa.magphase(linear)  # 计算复数图谱的幅度值和相位值
        features4 = librosa.power_to_db(linear, ref=1.0, amin=1e-10, top_db=None)
        mean1 = np.mean(features4, 0, keepdims=True)
        std1 = np.std(features4, 0, keepdims=True)
        features_spec = (features4 - mean1) / (std1 + 1e-5)
        # print(features_spec.shape)
        features_concat = np.concatenate((features_mel,features_spec),axis=0)
        # print(features_concat.shape)
        return features_concat
    else:
       raise Exception(f'预处理方法 {feature_method} 不存在！')


# 数据加载器
class CustomDataset(Dataset):
    def __init__(self, data_list_path, feature_method='features_concat', mode='train', sr=16000, chunk_duration=4,augmentors=None):
        super(CustomDataset, self).__init__()
        # 当预测时不需要获取数据
        if data_list_path is not None:
            with open(data_list_path, 'r') as f:
                self.lines = f.readlines()
        self.feature_method = feature_method
        self.mode = mode
        self.sr = sr
        self.chunk_duration = chunk_duration
        self.augmentors = augmentors

    def __getitem__(self, idx):
        try:
            audio_path, label = self.lines[idx].replace('\n', '').split('\t')
            # 加载并预处理音频
            features = load_audio(audio_path, feature_method=self.feature_method, mode=self.mode, sr=self.sr,
                                  chunk_duration=self.chunk_duration, augmentors=self.augmentors)
            return features, np.array(int(label), dtype=np.int64)
        except Exception as ex:
            print(f"[{datetime.now()}] 数据: {self.lines[idx]} 出错，错误信息: {ex}", file=sys.stderr)
            rnd_idx = np.random.randint(self.__len__())
            return self.__getitem__(rnd_idx)

    def __len__(self):
        return len(self.lines)

    @property
    def input_size(self):
        if self.feature_method == 'features_concat':
            return 281
        else:
            raise Exception(f'预处理方法 {self.feature_method} 不存在！')


# 对一个batch的数据处理
def collate_fn(batch):
    # 找出音频长度最长的
    batch = sorted(batch, key=lambda sample: sample[0].shape[1], reverse=True)
    freq_size = batch[0][0].shape[0]
    max_audio_length = batch[0][0].shape[1]
    batch_size = len(batch)
    # 以最大的长度创建0张量
    inputs = np.zeros((batch_size, freq_size, max_audio_length), dtype='float32')
    labels = []
    for x in range(batch_size):
        sample = batch[x]
        tensor = sample[0]
        labels.append(sample[1])
        seq_length = tensor.shape[1]
        # 将数据插入都0张量中，实现了padding
        inputs[x, :, :seq_length] = tensor[:, :]
    labels = np.array(labels, dtype='int64')
    # 打乱数据
    return torch.tensor(inputs), torch.tensor(labels)
