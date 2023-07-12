import random

import librosa
import librosa.display
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft



def displayWaveform(): # 显示语音时域波形


    samples, sr = librosa.load(r'dataset\24965-3-0-0.wav', sr=16000)
    # samples = samples[6000:16000]

    snr_dB =10
    noise = snr_dB * np.random.normal(0, 1, len(samples))
    samples= samples + noise
    print(len(samples), sr)
    time = np.arange(0, len(samples)) * (1.0 / sr)


    # min_speed_rate = 0.9, max_speed_rate = 1.1, num_rates = 3, prob = 0.5
    # speed_rate = random.uniform(0.9,1.1)
    # old_length = samples.shape[0]
    # new_length = int(old_length / speed_rate)
    # old_indices = np.arange(old_length)
    # new_indices = np.linspace(start=0, stop=old_length, num=new_length)
    # samples = np.interp(new_indices, old_indices, samples)
    #

    # gain = random.uniform(-15, 15)
    # samples *= 10. ** (gain / 20.)
    # time = np.arange(0, len(samples)) * (1.0 / sr)
    plt.plot(time, samples)
    plt.title("waveform")
    plt.xlabel("time")
    plt.ylabel("amplitude")
    plt.savefig("", dpi=1000)
    plt.show()



def displayspectrogram():

    x, sr = librosa.load(r'dataset\24965-3-0-0.wav', sr=16000)

    # compute power spectrogram with stft(short-time fourier transform):
    # 基于stft，计算power spectrogram

    # linear = librosa.stft(x, n_fft=400, win_length=400, hop_length=160)
    # features, _ = librosa.magphase(linear)  # 计算复数图谱的幅度值和相位值
    # features4 = librosa.power_to_db(features, ref=1.0, amin=1e-10, top_db=None)
    # mean1 = np.mean(features4, 0, keepdims=True)
    # std1 = np.std(features4, 0, keepdims=True)
    # features_spec = (features4 - mean1) / (std1 + 1e-5)
    features1 = librosa.feature.melspectrogram(x, sr=sr, n_fft=400, n_mels=80, hop_length=160, win_length=400)
    features2 = librosa.power_to_db(features1, ref=1.0, amin=1e-10, top_db=None) #取对数
    #features2 = librosa.power_to_db(features1_1, ref=1.0, amin=1e-10, top_db=None)
    mean = np.mean(features2, 0, keepdims=True)
    std = np.std(features2, 0, keepdims=True)
    features_mel= (features2 - mean) / (std + 1e-5)

    # show
    librosa.display.specshow(features_mel, y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('log-Mel spectrogram')
    plt.xlabel('time(s)')
    plt.ylabel('Frequency (hz)')
    plt.show()


if __name__ == '__main__':
    displayWaveform()
    # displayspectrogram()