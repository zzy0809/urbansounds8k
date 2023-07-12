import os
import random
import warnings

import numpy as np

warnings.filterwarnings("ignore")
import librosa


class WhitenoiseAugmentor(object):


    def __init__(self, min_snr_dB=10, max_snr_dB=50, sr=16000,prob=0.5):
        self.prob = prob
        self.sr = sr
        self._min_snr_dB = min_snr_dB
        self._max_snr_dB = max_snr_dB



    def __call__(self, wav):
        if random.random() > self.prob: return wav
        snr_dB = random.uniform(self._min_snr_dB, self._max_snr_dB)
        noise = snr_dB * np.random.normal(0, 1, len(wav))
        wav = wav + noise
        return wav



