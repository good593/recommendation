import pandas as pd 
import numpy as np 
import scipy.sparse as sp

from torch.utils.data import Dataset

__all__ = ['load_all', 'NCFData']

def load_all(config):
  """
  We load all the three file here to save time in each epoch.
  """
  train_data = pd.read_csv(
    config['train_rating'],
    sep='\t',
    header=None,
    names=['user', 'item'],
    usecols=[0,1],
    dtype={0: np.int32, 1:np.int32}
  )
  print(f'train_data: {train_data.shape}')
  user_num = train_data["user"].max() +1
  item_num = train_data["item"].max() +1

  # dok matrix 형식으로 저장
  train_data = train_data.values.tolist()

  train_mat = sp.dok_matrix((user_num, item_num), dtype=np.float32)
  for _data in train_data:
    train_mat[_data[0], _data[1]] = 1.0
  print(f'train_mat end')

  test_data = []
  with open(config['test_negative'], 'r') as fd:
    line = fd.readline()
    while line:
      arr = line.split("\t")
      u = eval(arr[0])[0]
      test_data.append(
        [ u, eval(arr[0])[1] ]
      )
      for i in arr[1:]:
        test_data.append(
          [ u, int(i) ]
        )
      line = fd.readline()
  print(f'test_negative end')

  return train_data, test_data, user_num, item_num, train_mat


class NCFData(Dataset):
  def __init__(self, features, num_item, train_mat=None, num_ng=0, is_training=False):
    super(NCFData, self).__init__()
    """
    Note that the labels are only useful when training, we thus add them in the ng_sample() function.
    """
    self.features_ps = features 
    self.num_item = num_item
    self.train_mat = train_mat
    self.num_ng = num_ng
    self.is_training = is_training
    self.labels = [0] * len(features)

  def set_ng_sample(self):
    assert self.is_training, "no need to sampling when testing"

    # add negative sample 
    self.feaures_ng = [] 
    for ps in self.features_ps:
      # user 
      u = ps[0]
      for _ in range(self.num_ng):
        # item
        i = np.random.randint(self.num_item)
        # train set에 있는 경우 다시 뽑기
        while (u, i) in self.train_mat:
          i = np.random.randint(self.num_item)
        
        self.feaures_ng.append(
          [u, i]
        )

    labels_ps = [1] * len(self.features_ps)
    labels_ng = [0] * len(self.feaures_ng)

    self.features_fill = self.features_ps + self.feaures_ng
    self.labels_fill = labels_ps + labels_ng

  def __len__(self):
    return (self.num_ng +1) * len(self.labels)

  def __getitem__(self, idx):
    features = self.features_fill if self.is_training else self.features_ps
    labels = self.labels_fill if self.is_training else self.labels 

    user = features[idx][0]
    item = features[idx][1]
    label = labels[idx]
    return user, item, label 



