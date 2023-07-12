import random

import numpy as np



class SpeedPerturbAugmentor(object):


    def __init__(self, min_speed_rate=0.9, max_speed_rate=1.1, num_rates=3, prob=0.5):
        if min_speed_rate < 0.9:
            raise ValueError("Sampling speed below 0.9 can cause unnatural effects")
        if max_speed_rate > 1.1:
            raise ValueError("Sampling speed above 1.1 can cause unnatural effects")
        self.prob = prob
        self._min_speed_rate = min_speed_rate
        self._max_speed_rate = max_speed_rate
        self._num_rates = num_rates
        if num_rates > 0:
            self._rates = np.linspace(self._min_speed_rate, self._max_speed_rate, self._num_rates, endpoint=True)

    def __call__(self, wav):

        if random.random() > self.prob: return wav
        if self._num_rates < 0:
            speed_rate = random.uniform(self._min_speed_rate, self._max_speed_rate)
        else:
            speed_rate = random.choice(self._rates)
        if speed_rate == 1.0: return wav

        old_length = wav.shape[0]
        new_length = int(old_length / speed_rate)
        old_indices = np.arange(old_length)
        new_indices = np.linspace(start=0, stop=old_length, num=new_length)
        wav = np.interp(new_indices, old_indices, wav)
        num_wav_samples = wav.shape[0]
        # 数据太短不利于训练
        chunk_duration=4
        sr=16000
        cnt=0
        num_chunk_samples = int(chunk_duration * sr)
        if num_wav_samples > num_chunk_samples + 1:
            start = random.randint(0, num_wav_samples - num_chunk_samples - 1)
            stop = start + num_chunk_samples
            wav = wav[start:stop]
        else:
            new_wav = np.zeros(num_chunk_samples)
            new_wav[:num_wav_samples] = wav[:num_wav_samples]
            for i in range(num_wav_samples, num_chunk_samples):
                new_wav[i] = wav[cnt]
                cnt = cnt + 1
                if cnt == num_wav_samples:
                    cnt = 0
            wav = new_wav
        return wav
