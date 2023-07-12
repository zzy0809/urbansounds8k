import argparse
import functools
import os
from datetime import datetime

import numpy as np
import torch
import yaml
from sklearn.metrics import confusion_matrix
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchsummary import summary

from data_utils.reader import CustomDataset, collate_fn
from data_utils.speed_perturb import SpeedPerturbAugmentor
from data_utils.volume_perturb import VolumePerturbAugmentor
from data_utils.gaussian_white_noise import WhitenoiseAugmentor
from modules.se_resnet import se_resnet, Res2Conv1dReluBn
from utils.utility import add_arguments, print_arguments, plot_confusion_matrix

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('use_model',        str,    'se_resnet',             '所使用的模型')
add_arg('batch_size',       int,    64,                       '训练的批量大小')
add_arg('num_workers',      int,    4,                        '读取数据的线程数量')
add_arg('num_epoch',        int,    100,                       '训练的轮数')
add_arg('num_classes',      int,    10,                       '分类的类别数量')
add_arg('learning_rate',    float,  1e-3,                     '初始学习率的大小')
add_arg('train_list_path',  str,    'dataset/train_list.txt', '训练数据的数据列表路径')
add_arg('test_list_path',   str,    'dataset/test_list.txt',  '测试数据的数据列表路径')
add_arg('save_model_dir',   str,    'output/',                    '模型保存的路径')
add_arg('feature_method',   str,    'features_fusion',         '音频特征提取方法')
add_arg('augment_conf_path',str,    'configs/augment.yml',    '数据增强的配置文件，为json格式')
add_arg('resume',           str,    None,                     '恢复训练的模型文件夹，当为None则不使用恢复模型')
args = parser.parse_args()


# 评估模型
@torch.no_grad()
def evaluate(model, test_loader, device):
    model.eval()
    accuracies, preds, labels = [], [], []
    for batch_id, (spec_mag, label) in enumerate(test_loader):
        spec_mag = spec_mag.to(device)
        label = label.numpy()
        output = model(spec_mag)
        output = torch.nn.functional.softmax(output, dim=-1)
        output = output.data.cpu().numpy()
        pred = np.argmax(output, axis=1)
        preds.extend(pred.tolist())
        labels.extend(label.tolist())
        acc = np.mean((pred == label).astype(int))
        accuracies.append(acc.item())
    model.train()
    acc = float(sum(accuracies) / len(accuracies))
    return acc


def train(args):
    # 获取数据增强器
    augmentors = None
    if args.augment_conf_path is not None:
        augmentors = {}
        with open(args.augment_conf_path, encoding="utf-8") as fp:
            configs = yaml.load(fp, Loader=yaml.FullLoader)
        augmentors['noise'] = WhitenoiseAugmentor(**configs['noise'])
        augmentors['speed'] = SpeedPerturbAugmentor(**configs['speed'])
        augmentors['volume'] = VolumePerturbAugmentor(**configs['volume'])
    # 获取数据
    train_dataset = CustomDataset(args.train_list_path,
                                  feature_method=args.feature_method,
                                  mode='train',
                                  sr=16000,
                                  chunk_duration=4,
                                  augmentors=augmentors)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=args.num_workers,drop_last=True)

    test_dataset = CustomDataset(args.test_list_path,
                                 feature_method=args.feature_method,
                                 mode='eval',
                                 sr=16000,
                                 chunk_duration=4)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, collate_fn=collate_fn, num_workers=args.num_workers)
    # 获取模型
    device = torch.device("cuda")
    if args.use_model == 'se_resnet':
        model =se_resnet(num_classes=args.num_classes, input_size=train_dataset.input_size)
    else:
        raise Exception(f'{args.use_model} 模型不存在!')
    model.to(device)
    summary(model, (281,401))

    # 获取优化方法
    optimizer = torch.optim.Adam(params=model.parameters(),
                                 lr=args.learning_rate,
                                 weight_decay=5e-4)


    # 获取学习率衰减函数
    scheduler = CosineAnnealingLR(optimizer, T_max=4)

    # 恢复训练
    if args.resume is not None:
        model.load_state_dict(torch.load(os.path.join(args.resume, 'model.pth')))
        state = torch.load(os.path.join(args.resume, 'model.state'))
        last_epoch = state['last_epoch']
        optimizer_state = torch.load(os.path.join(args.resume, 'optimizer.pth'))
        optimizer.load_state_dict(optimizer_state)
        print(f'成功加载第 {last_epoch} 轮的模型参数和优化方法参数')

    # 获取损失函数
    loss = torch.nn.CrossEntropyLoss()

    # 开始训练
    for epoch in range(args.num_epoch):
        loss_sum = []
        accuracies = []
        for batch_id, (spec_mag, label) in enumerate(train_loader):
            spec_mag = spec_mag.to(device)
            label = label.to(device).long()
            try:
               output = model(spec_mag)
            except:
                continue
            # 计算损失值
            los = loss(output, label)
            optimizer.zero_grad()
            los.backward()
            optimizer.step()

            # 计算准确率
            output = torch.nn.functional.softmax(output, dim=-1)
            output = output.data.cpu().numpy()
            output = np.argmax(output, axis=1)
            label = label.data.cpu().numpy()
            acc = np.mean((output == label).astype(int))
            accuracies.append(acc)
            loss_sum.append(los)
            if batch_id % 10 == 0:
                print(f'[{datetime.now()}] Train epoch [{epoch}/{args.num_epoch}], batch: {batch_id}/{len(train_loader)}, '
                      f'lr: {scheduler.get_last_lr()[0]:.8f}, loss: {sum(loss_sum) / len(loss_sum):.8f}, '
                      f'accuracy: {sum(accuracies) / len(accuracies):.8f}')
        scheduler.step()
        # 评估模型
        acc = evaluate(model, test_loader, device)
        print('='*70)
        print(f'[{datetime.now()}] Test {epoch}, Accuracy: {acc}')
        print('='*70)
        # 保存模型
        os.makedirs(args.save_model_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(args.save_model_dir, 'model.pth'))
        torch.save({'last_epoch': torch.tensor(epoch)}, os.path.join(args.save_model_dir, 'model.state'))
        torch.save(optimizer.state_dict(), os.path.join(args.save_model_dir, 'optimizer.pth'))


if __name__ == '__main__':
    print_arguments(args)
    train(args)
