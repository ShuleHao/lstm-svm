# -*- coding:utf-8 -*-
# @author: ShuleHao
# @contact: 2571540718@qq.com
# @Blog:https://blog.csdn.net/hubuhgyf?type=blog
"""
    文件说明：模型文件
"""
import torch.nn as nn
from mysvm import layers

class Mymodel(nn.Module):
    def __init__(self,vocab_size,embed_size,hidden_dim,n_layer):
        super(Mymodel,self).__init__()
        self.embedding1=nn.Embedding(vocab_size,embed_size)
        self.lism1=nn.LSTM(embed_size,hidden_dim,n_layer,batch_first=True)
        self.line1=nn.Linear(hidden_dim,int(hidden_dim/2))
        self.line2=nn.Linear(int(hidden_dim/2),int(hidden_dim/4))
        self.line3 = nn.Linear(int(hidden_dim / 4), int(hidden_dim / 8))
        self.svm = layers.KernelLayer(int(hidden_dim/8), 2)
        self.activition=nn.ReLU()
    def forward(self,x):
        x=self.embedding1(x)
        lstm_out, (hidden_last, cn_last)=self.lism1(x)
        x=self.line1(hidden_last)
        x=self.activition(x)
        x=self.line2(x)
        x=self.line3(x)[0,:,:]
        x=self.svm(x)
        return x
