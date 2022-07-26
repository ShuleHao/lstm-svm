# -*- coding:utf-8 -*-
# @author: ShuleHao
# @contact: 2571540718@qq.com
# @Blog:https://blog.csdn.net/hubuhgyf?type=blog
"""
    文件说明：
    数据类文件，定义模型所需的数据类，方便模型训练使用
"""
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import torch
import os
import logging
import numpy as np
from torch.utils.data import TensorDataset
logger = logging.getLogger(__name__)


class SentimentDataSet():
    def __init__(self, maxlen, max_words, data_dir, data_set_name, path_file,num=None):
        '''
        情感分析模型所需要的数据类
        :param maxlen: 句子的最大长度
        :param max_words: 词典大小
        :param data_dir: 保存缓存的数据集路径
        :param data_set_name: 保存缓存数据的名称
        :param path_file: 本地数据集路径
        :param num:取数量多少的数据集，默认为None
        '''
        self.maxlen = maxlen
        self.max_words=max_words
        self.num=num
        cached_feature_file = os.path.join(data_dir, "cached{}".format(data_set_name))
        # 判断缓存文件是否存在，如果存在，则直接加载处理后数据
        if os.path.exists(cached_feature_file):
            logger.info("已经存在缓存文件{}，直接加载".format(cached_feature_file))
            self.data_set = torch.load(cached_feature_file)
        # 如果缓存数据不存在，则对原始数据进行数据处理操作，并将处理后的数据存成缓存文件
        else:
            logger.info("不存在缓存文件{}，进行数据预处理操作".format(cached_feature_file))
            self.data_set = self.load_data(path_file)
            logger.info("数据预处理操作完成，将处理后的数据存到{}中，作为缓存文件".format(cached_feature_file))
            torch.save(self.data_set, cached_feature_file)
    def load_data(self, path_file):
        '''
        加载原始数据，生成数据处理后的数据
        :param path_file:
        :return: dict类型将特征与标签，打包到一起成为tensordata
        '''
        labels = []
        texts = []
        for label_type in ['neg', 'pos']:
            dir_name = os.path.join(path_file, label_type)
            for fname in os.listdir(dir_name):
                if fname[-4:] == '.txt':
                    f = open(os.path.join(dir_name, fname),
                             encoding='utf-8')  # 只能加encoding='utf-8'UnicodeDecodeError: 'gbk' codec can't decode byte 0x93 in position 130: illegal multibyte sequence
                    texts.append(f.read())
                    f.close()
                    if label_type == 'neg':
                        labels.append(0)
                    else:
                        labels.append(1)
        tokenizer = Tokenizer(num_words=self.max_words)
        tokenizer.fit_on_texts(texts[:1])
        sequences = tokenizer.texts_to_sequences(texts)
        feature = pad_sequences(sequences, maxlen=self.maxlen)
        labels = np.asarray(labels)
        indices = np.arange(feature.shape[0])
        np.random.shuffle(indices)
        feature = feature[indices]
        labels = labels[indices]
        if self.num:
            feature =feature[:self.num]
            labels= labels[:self.num]
        xt = torch.LongTensor(feature)  # 将Numpy数据转化为张量
        yt = torch.LongTensor(labels)
        data = TensorDataset(xt, yt)
        return {"data_set": data}
    def __getitem__(self, key):
        '''
        类返回数据的方法
        :param key:
        :return:
        '''
        return self.data_set.get(key, "None")



