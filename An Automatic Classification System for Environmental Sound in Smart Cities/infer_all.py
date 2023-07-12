import argparse
import functools

import numpy as np
import torch

from data_utils.reader import CustomDataset
from data_utils.reader import load_audio
from modules.se_resnet import se_resnet
from utils.utility import add_arguments
from torchsummary import summary
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('use_model',        str,    'se_resnet',             '所使用的模型')
add_arg('audio_path',       str,    r'',                        '音频路径')
add_arg('num_classes',      int,    10,                        '分类的类别数量')
add_arg('label_list_path',  str,    'dataset/label_list.txt',  '标签列表路径')
add_arg('model_path',       str,    'output/',                   '模型保存的路径')
add_arg('feature_method',   str,    'features_fusion',          '音频特征提取方法')
args = parser.parse_args()


train_dataset = CustomDataset(data_list_path=None, feature_method=args.feature_method)
# 获取分类标签
with open(args.label_list_path, 'r', encoding='GBK') as f:
    lines = f.readlines()
class_labels = [l.replace('\n', '') for l in lines]
# 获取模型
device = torch.device("cuda")
if args.use_model == 'se_resnet':
    model = se_resnet(num_classes=args.num_classes, input_size=train_dataset.input_size)
else:
    raise Exception(f'{args.use_model} 模型不存在!')
model.to(device)
model.load_state_dict(torch.load(args.model_path))
model.eval()


def infer(audio_path):
    data = load_audio(audio_path, mode='infer')
    data = data[np.newaxis, :]
    data = torch.tensor(data, dtype=torch.float32, device=device)
    # 执行预测
    output = model(data)
    result = torch.nn.functional.softmax(output, dim=-1)
    result = result.data.cpu().numpy()
    # 显示图片并输出结果最大的label
    lab = np.argsort(result)[0][-1]
    return lab

cnt_all = [0,0,0,0,0,0,0,0,0,0]
cnt_match = [0,0,0,0,0,0,0,0,0,0]

cnt = 0
if __name__ == '__main__':
    # 要预测的音频文件
    # path = r"G:\zzy1\test1"  # 'UrbanSound8K/audio/fold8/193699-2-0-46.wav'
    path=r"G:\zzy\two-stream\dataset\test"
    rel_label = -1
    cnt_result = 0
    cnt_all2 = 0
    for root1, dirs1, files1 in os.walk(path):
        for file in files1:
            file_path = os.path.join(root1,  file)
            f_list = file.split('-')
            rel_label = int(f_list[1])
            cnt_all[rel_label] = cnt_all[rel_label] + 1
            cnt = cnt + 1
            pre_label = infer(file_path)
            print(str(cnt), file_path, pre_label)
            if pre_label == rel_label:
                cnt_match[rel_label] = cnt_match[rel_label] + 1



        for i in range(10):
            if cnt_all[i] != 0:
                print(class_labels[i] + '类正确率:' + str(100 * cnt_match[i] / cnt_all[i])[0:4] + '%')
            cnt_result =cnt_result+cnt_match[i]
            cnt_all2 = cnt_all2 + cnt_all[i]


        print(cnt_match)
        print(cnt_all)
        print('总正确率:' + str(100 * cnt_result / cnt_all2)[0:4] + '%')

    # infer()

