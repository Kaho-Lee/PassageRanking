import os
import numpy as np
import json
import matplotlib.pyplot as plt
import tqdm
import csv
import pandas as pd
import re
from utils import *
from nltk.stem import PorterStemmer, SnowballStemmer
from sklearn.model_selection import KFold
import xgboost as xgb
import pickle
from random import shuffle

def batchDMatrix(reader, sampled_data_qid, embedding, idf):
    group = []
    for i, qid in enumerate(sampled_data_qid):
        # print(i, qid)
        
        cur_qid_label, cur_qid_em = Data_Embedding(embedding, reader, qid, idf[str(qid)], mode='Train')
        if i ==0:
            batch_label = cur_qid_label
            batch_em = cur_qid_em[:, 1:]
            group.append(cur_qid_em.shape[0])
        else:
            batch_label = np.vstack((batch_label, cur_qid_label))
            batch_em = np.vstack((batch_em, cur_qid_em[:, 1:]))
            group.append(cur_qid_em.shape[0])

    dmatrix = xgb.DMatrix(batch_em, label = batch_label)
    dmatrix.set_group(group)
    print('Gnerating Data Shape em: {}, label: {}'.format(batch_em.shape, batch_label.shape))
    return dmatrix

