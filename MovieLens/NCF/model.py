import os
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn
# from tensorboardX import SummaryWriter

import pandas as pd
import scipy.sparse as sp


class NCF(nn.Module):
  def __init__(self, user_num, item_num, factor_num, num_layers, dropout, model):
    super(NCF, self).__init__()
    """
    user_num: number of users
    item_num: number of items
    factor_num: number of predictive factors
    num_layers: the number of layers in MLP model
    dropout: dropout rate between fully connected layers
    model: MLP, GMF, NeuMF-end, NeuMF-pre
    """

    self.dropout = dropout
    self.model = model 

    # 임베딩 저장공간 확보; (num_embeddings, embedding_dim)
    self.embed_user_MLP = nn.Embedding(
      user_num, factor_num * ( 2 ** (num_layers -1) )
    )
    self.embed_item_MLP = nn.Embedding(
      item_num, factor_num * ( 2 ** (num_layers -1) )
    )

    MLP_modules = []
    for i in range(num_layers):
      input_size = factor_num * ( 2 ** (num_layers - i ))
      MLP_modules.append(nn.Dropout(p=self.dropout))
      MLP_modules.append(nn.Linear(input_size, input_size // 2))
      MLP_modules.append(nn.ReLU())

    self.MLP_layers = nn.Sequential(*MLP_modules)
    predict_size = factor_num
    self.predict_layer = nn.Linear(predict_size, 1)
    self._init_weight_()

  def _init_weight_(self):
    # weight init
    nn.init.normal_(self.embed_user_MLP.weight, std=0.01)
    nn.init.normal_(self.embed_item_MLP.weight, std=0.01)
    for m in self.MLP_layers:
      if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
    
    nn.init.kaiming_uniform_(self.predict_layer.weight, a=1, nonlinearity='sigmoid')

    # bias init
    for m in self.modules():
      if isinstance(m, nn.Linear) and m.bias is not None:
        m.bias.data.zero_()

  def forward(self, user, item):
    embed_user_MLP = self.embed_user_MLP(user)
    embed_item_MLP = self.embed_item_MLP(item)
    # 임베딩 벡터 합치기
    interaction = torch.cat(
      (embed_user_MLP, embed_item_MLP), -1
    )
    output_MLP = self.MLP_layers(interaction)
    concat = output_MLP 

    # 예측하기
    prediction = self.predict_layer(concat)
    return prediction.view(-1)
