# -*- coding:utf-8 -*-
# @author: ShuleHao
# @contact: 2571540718@qq.com
# @Blog:https://blog.csdn.net/hubuhgyf?type=blog
"""
    文件说明：训练文件
"""
import torch
import random
from model import Mymodel
from mysvm.utils import save
from torch.utils.data import  DataLoader
import argparse
import numpy as np
from tqdm import tqdm, trange#和range一样但是增加了进度条
import torch.optim as optim
import torch.nn.functional as F
from data_set import SentimentDataSet
import os
import logging
#显示log
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

def acc(p,t):
    '''
    评测函数,目前只有acc评价指标，没有将召回率和f-1作为评价指标
    :param p: 预测值
    :param t: 真实值
    :return: acc
    '''
    a = np.array(p)
    b = np.array(t)
    temp = 0
    for i in a == b:
        if i == True:
            temp += 1
    result=temp/len(a)
    return result

def set_args():
    """设置训练模型所需参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='1', type=str, help='设置训练或测试时使用的显卡')
    parser.add_argument('--train_file_path', default='data/aclImdb/train', type=str, help='新闻标题生成的训练数据')
    parser.add_argument('--test_file_path', default='data/aclImdb/test', type=str, help='新闻标题生成的测试数据')
    parser.add_argument('--data_dir', default='data', type=str, help='生成缓存数据的存放路径')
    parser.add_argument('--num_train_epochs', default=1000, type=int, help='模型训练的轮数')
    parser.add_argument('--train_batch_size', default=100, type=int, help='训练时每个batch的大小')
    parser.add_argument('--test_batch_size', default=100, type=int, help='测试时每个batch的大小')
    parser.add_argument('--learning_rate', default=0.01, type=float, help='模型训练时的学习率')
    parser.add_argument('--eval_steps', default=10, type=int, help='训练时，多少步进行一次测试')
    parser.add_argument('--output_dir', default='output_dir/', type=str, help='模型输出路径')
    parser.add_argument('--seed', type=int, default=1, help='随机种子')
    parser.add_argument('--max_len', type=int, default=100, help='输入模型的最大长度，要比config中n_ctx小')
    return parser.parse_args()




model=Mymodel(vocab_size=10000,embed_size=10,n_layer=1,hidden_dim=128)

def train(model,device,train_loader, valid_loader, args):
    '''

    :param model: 定义的模型写在model.py文件中
    :param train_loader: 训练集
    :param valid_loader: 测试机
    :param args:
    :return:
    '''
    #随机种子方便模型复现
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    model.to(device)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(parameters, lr=args.learning_rate, momentum=0.9, weight_decay=5e-4)
    count = 0
    for i in trange(args.num_train_epochs, desc="Epoch", disable=False):
        sum_loss = 0
        iter_bar = tqdm(train_loader, desc="Iter (loss=X.XXX)", disable=False)
        for d, l in iter_bar:
            d=d.to(device)
            l=l.to(device)
            optimizer.zero_grad()
            output = model.forward(d)
            loss = F.cross_entropy(output, l)
            sum_loss += loss
            iter_bar.set_description("Iter (loss=%5.3f)" % loss.item())
            loss.backward()
            optimizer.step()
            true_result = []
            count += 1
            if count % args.eval_steps == 0:
                for dv, lv in valid_loader:
                    dv=dv.to(device)
                    lv=lv.to(device)
                    with torch.no_grad():#防止反向传参
                        output = model.forward(dv)
                    y = torch.argmax(output, dim=1)
                    predict_result = list(y.numpy())
                    true_result = list(lv.numpy())
                print()
                print('acc:', acc(true_result, predict_result))
    save(model,args.output_dir)
def main():
    args=set_args()
    # 设置显卡信息
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICE"] = args.device
    # 获取device信息，用于模型训练
    device = torch.device("cuda" if torch.cuda.is_available() and int(args.device) >= 0 else "cpu")
    train_data = SentimentDataSet(maxlen=args.max_len, max_words=10000, data_dir='data', data_set_name='train',
                                  path_file=args.train_file_path,num=200)["data_set"]
    valid_data = SentimentDataSet(maxlen=args.max_len, max_words=10000, data_dir='data', data_set_name='valid',
                                  path_file=args.test_file_path)["data_set"]
    train_loader = DataLoader(train_data,
                              shuffle=True,
                              batch_size=args.train_batch_size,
                              drop_last=True)
    valid_loader = DataLoader(valid_data,
                              shuffle=True,
                              batch_size=args.test_batch_size,
                              drop_last=True)
    train(model,device,train_loader,valid_loader,args)

if __name__ == '__main__':
    main()














