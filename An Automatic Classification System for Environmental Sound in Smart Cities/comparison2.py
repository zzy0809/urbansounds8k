import librosa
import librosa.display
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft



def displayWaveform( mode='train', sr=16000, chunk_duration=4):
    # 读取音频数据
    wav, sr_ret = librosa.load(r'dataset\24965-3-0-0.wav', sr=sr)
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
            for i in range(num_wav_samples, num_chunk_samples):
               new_wav[i] = (a[cnt])
               cnt = cnt + 1
               if cnt == num_wav_samples:
                    cnt = 0
            wav = new_wav
    features1 = librosa.feature.melspectrogram(wav, sr=sr, n_fft=400, n_mels=80, hop_length=160, win_length=400)
    features2 = librosa.power_to_db(features1, ref=1.0, amin=1e-10, top_db=None)  # 取对数
    # features2 = librosa.power_to_db(features1_1, ref=1.0, amin=1e-10, top_db=None)
    mean = np.mean(features2, 0, keepdims=True)
    std = np.std(features2, 0, keepdims=True)
    features_mel = (features2 - mean) / (std + 1e-5)

    print(len(wav), sr)
    time = np.arange(0, len(wav)) * (1.0 / sr)
    plt.plot(time, wav)
    plt.title("waveform")
    plt.xlabel("time")
    plt.ylabel("amplitude")
    plt.savefig("***.jpg", dpi=1000)
    plt.show()
    # show
    librosa.display.specshow(features_mel, y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('log-Mel spectrogram')
    plt.xlabel('time(s)')
    plt.ylabel('Frequency (hz)')
    plt.savefig( "***.jpg", dpi=1000)
    plt.show()


if __name__ == '__main__':
    displayWaveform()
    # displayspectrogram()