# -*- coding:utf-8 -*-
# @author: ShuleHao
# @contact: 2571540718@qq.com
# @Blog:https://blog.csdn.net/hubuhgyf?type=blog
"""
    文件说明：里面写了一些自定义的损势函数（加入了惩罚机制正则化l1,l2，）；保存模型，加载模型参数的方法
"""
import torch
import torch.nn.functional as F


def svm_l1loss(a, y, weight, C=1, batch_fraction=1):
    '''
    计算 SVM-L1 loss:
    l(w) = 0.5 * w^T w + C \sum_{i=1}^N max(1 - w^Tx_n y_n, 0)
    :param a: 输入的特征
    :param y: 标签
    :param weight: 上一层的权重张量
    :param C: 正则化常数
    :param batch_fraction: 小批量中的样本分数,与整个样本相比
    :return:
    '''

    relu = F.relu(1 - a * y)
    loss = 0.5 * batch_fraction * (weight * weight).sum() + C * relu.sum()
    return loss


def svm_l1loss2(prediction, target, weight, C=1, batch_fraction=1):
    '''
     计算 SVM-L1 loss:
    l(w) = 0.5 * w^T w + C \sum_{i=1}^N max(1 - w^Tx_n y_n, 0)
    :param prediction: 输入的特征
    :param target: 标签
    :param weight: 上一层的权重张量
    :param C: 正则化常数
    :param batch_fraction: 小批量中的样本分数,与整个样本相比
    :return:
    '''
    relu = F.relu(1 + prediction - prediction[target].view(-1, 1)).sum(dim=1) - 1
    loss = 0.5 * batch_fraction * (weight * weight).sum() + C * relu.sum()
    return loss


def svm_l2loss(a, y, weight, C=1, batch_fraction=1):
    """
    计算 SVM-L2 loss:
    l(w) = 0.5 * w^T w + C \sum_{i=1}^N max(1 - w^Tx_n y_n, 0)^2
    参数介绍与之前相似
    """
    relu = F.relu(1 - a * y)**2
    return 0.5 * batch_fraction * (weight * weight).sum() + C * relu.sum()


def svm_l2loss2(prediction, target, weight, C=1, batch_fraction=1):
    '''
    计算 SVM-L1 loss:
    l(w) = 0.5 * w^T w + C \sum_{i=1}^N max(1 - w^Tx_n y_n, 0)

    参数介绍与之前相似
    '''
    relu = (F.relu(1 + prediction - prediction[target].view(-1, 1))**2).sum(dim=1) - 1
    loss = 0.5 * batch_fraction * (weight * weight).sum() + C * relu.sum()
    return loss



def save(model, path):
    print('Saving..')
    state = model.state_dict()
    torch.save(state, '{}'.format(path))

def load_model(basic_model, path):
    checkpoint = torch.load('{}'.format(path))
    basic_model.load_state_dict(checkpoint)


