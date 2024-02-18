import numpy as np

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def calculate(list_t):
  try:
    if((len(list_t)!=9)):
      raise ValueError()
    else:
      x = np.reshape(list_t,(3,3))
  except ValueError as error:
    return 'List must contain nine numbers.'

  l = [
    [x[:,0].mean(),x[:,1].mean(),x[:,2].mean()],
    [x[0].mean(),x[1].mean(),x[2].mean()],
    [x[:, 0].var(), x[:, 1].var(), x[:, 2].var()],
    [x[0].var(), x[1].var(), x[2].var()],
    [x[:, 0].std(), x[:, 1].std(), x[:, 2].std()],
    [x[0].std(), x[1].std(), x[2].std()],
    [x[:, 0].max(), x[:, 1].max(), x[:, 2].max()],
    [x[0].max(), x[1].max(), x[2].max()],
    [x[:, 0].min(), x[:, 1].min(), x[:, 2].min()],
    [x[0].min(), x[1].min(), x[2].min()],
    [x[:, 0].sum(), x[:, 1].sum(), x[:, 2].sum()],
    [x[0].sum(), x[1].sum(), x[2].sum()]
     ]

  dic = {
    'mean' : [l[0],l[1],np.mean(list_t)],
    'variance': [l[2], l[3],np.var(list_t)],
    'standard deviation': [l[4], l[5],np.std(list_t)],
    'max':[l[6],l[7],np.max(list_t)],
    'min':[l[8],l[9],np.min(list_t)],
    'sum':[l[10],l[11],np.sum(list_t)]
  }
  return dic

inn = [0,1,2,3,4,5,6,7,8]
print(calculate(inn))
