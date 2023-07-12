# 前言

本章我们来介绍如何使用Pytorch训练一个区分不同音频的分类模型

# 环境准备

主要介绍libsora，PyAudio的安装，其他的依赖包根据需要自行安装。

- Python 3.7
- Pytorch 

## 安装libsora

最简单的方式就是使用pip命令安装，如下：
```shell
pip install librosa==0.9.1
```

## 安装PyAudio

使用pip安装命令，如下：

```shell
pip install pyaudio
```

# 数据增强
使用随机添加高斯白噪声
```python
snr_dB = random.uniform(self._min_snr_dB, self._max_snr_dB)
noise = snr_dB * np.random.normal(0, 1, len(wav))
wav = wav + noise
```
使用线性插值进行改变音频速度
```python
 old_length = wav.shape[0]
        new_length = int(old_length / speed_rate)
        old_indices = np.arange(old_length)
        new_indices = np.linspace(start=0, stop=old_length, num=new_length)
        wav = np.interp(new_indices, old_indices, wav)
        num_wav_samples = wav.shape[0]
```
随机添加 音量大小
```python
gain = random.uniform(self._min_gain_dBFS, self._max_gain_dBFS)
wav *= 10.**(gain / 20.)
```
# 训练分类模型

使用librosa可以很方便得到音频的频谱（Spectrogram），使用的API为 `librosa.stft（）`，再进行取对数，使用的API为 `librosa.power_to_db（）`，得到log-Spectrogram.
使用librosa可以很方便得到音频的梅尔频谱（Mel Spectrogram），使用的API为 `librosa.feature.melspectrogram()`，输出的是numpy值。再进行取对数，使用的API为 `librosa.power_to_db（）`，得到log-Mel Spectrogram.

## 生成数据列表

生成数据列表，用于下一步的读取需要，`audio_path`为音频文件路径，用户需要提前把音频数据集存放在`dataset/audio`目录下，每个文件夹存放一个类别的音频数据，每条音频数据长度在3秒以上，如 `dataset/audio/air_conditioner/······`。`audio`是数据列表存放的位置，生成的数据类别的格式为 `音频路径\t音频对应的类别标签`。

下面是在训练时或者测试时读取音频数据进行统一裁剪

```python
def load_audio(audio_path, mode='train', sr=16000, chunk_duration=4):
    # 读取音频数据
    wav, sr_ret = librosa.load(audio_path, sr=sr)
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
                new_wav[i] = a[cnt]
                cnt = cnt + 1
                if cnt == num_wav_samples:
                    cnt = 0
            wav = new_wav
        elif mode == 'eval':
        num_wav_samples = wav.shape[0]
        num_chunk_samples = int(chunk_duration * sr)
        if num_wav_samples > num_chunk_samples + 1:
            wav = wav[:num_chunk_samples]
        else:
            new_wav = np.zeros(num_chunk_samples)
            new_wav[:num_wav_samples] = wav[:num_wav_samples]
            for i in range(num_wav_samples, num_chunk_samples):
                new_wav[i] = a[cnt]
                cnt = cnt + 1
                if cnt == num_wav_samples:
                    cnt = 0
            wav = new_wav
   

## 训练

接着就可以开始训练模型了，创建 `train.py`。我们搭建双分支残差网络，通过把音频数据转换成log-梅尔频谱、log-频谱。然后定义优化方法和获取训练和测试数据。

# 预测

在训练结束之后，我们得到了一个模型参数文件，我们使用这个模型预测音频，在执行预测之前，需要把音频转换为log-梅尔频谱、log-频谱数据，最后输出的结果即为预测概率最大的标签。
```python
def infer():
    data = load_audio(args.audio_path, mode='infer')
    data = data[np.newaxis, :]
    data = torch.tensor(data, dtype=torch.float32, device=device)
    # 执行预测
    output = model(data)
    result = torch.nn.functional.softmax(output, dim=-1)
    result = result.data.cpu().numpy()
    # 显示图片并输出结果最大的label
    lab = np.argsort(result)[0][-1]
    print(f'音频：{args.audio_path} 的预测结果标签为：{class_labels[lab]}')


if __name__ == '__main__':
    infer()
```
