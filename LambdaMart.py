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
from nltk.tokenize import word_tokenize 
import xgboost as xgb
import pickle

def batchDMatrix(reader, sampled_data_qid, embedding, idf):
    group = []
    for i, qid in enumerate(sampled_data_qid):
        cur_qid_label, cur_qid_em = Data_Embedding(embedding, reader, qid, idf[str(qid)], mode='train')
        if i ==0:
            batch_label = cur_qid_label
            batch_em = cur_qid_em[:, 1:]
            group.append(cur_qid_em.shape[0])
        else:
            batch_label = np.vstatck((batch_label, cur_qid_label))
            batch_em = np.vstack((batch_em, cur_qid_em[:, 1:]))
            group.append(cur_qid_em.shape[0])

    dmatrix = xgd.DMatrix(batch_em, label = batch_label)
    dmatrix.set_group(group)

    return dmatrix

if __name__=="__main__":

    train_path = '/Users/leekaho/Desktop/part2/train_data.tsv'
    val_path = '/Users/leekaho/Desktop/part2/validation_data.tsv'
    gloveEmbedding_path = '/Users/leekaho/Desktop/part2/gloveEmbedding.json'
    idf_train = '/Users/leekaho/Desktop/part2/train_idf.json'

    train_reader = pd.read_csv(train_path, sep='\t')
    train_qid_unique = train_reader.qid.unique().tolist()

    #from xgb sample code https://github.com/dmlc/xgboost/blob/master/demo/rank/rank.py
    params = {'objective': 'rank:ndcg', 'eta': 0.1, 'gamma': 1.0,
          'min_child_weight': 0.1, 'max_depth': 6}

    kf = KFold(n_splits=2, random_state=None, shuffle=False)


    with open(gloveEmbedding_path, 'r') as readFile:
        embedding = json.load(readFile)
    readFile.close()

    with open(idf_train, 'r') as readFile:
        train_idf = json.load(readFile)
    readFile.close()

    i=0
    for train_qid_index, eval_qid_index in kf.split(train_qid_unique):
        i += 1
        sampled_train_index = train_qid[np.random.choice(len(train_qid), int(0.1*len(train_qid_index)))]
        sampled_eval_index = eval_qid[np.random.choice(len(eval_qid), int(0.1*len(eval_qid_index)))]
        print(sampled_train_qid_index, sampled_eval_qid_index )

        
        train_data_qid = [train_qid_unique[x] for x in sampled_train_qid_index]
        eval_data_qid = [train_qid_unique[x] for x in sampled_eval_qid_index]
        print(train_data_qid, eval_data_qid)

        train_dmatrix = batchDMatrix(train_reader, train_data_qid, embedding, train_idf)
        eval_dmatrix = betchDMatrix(train_reader, eval_data_qid, embedding, train_idf)


        break



    # batch_train_label, batch_train_em = valData_Embedding(embedding, train_reader, train_qid_unique[0], \
    # train_idf[str(train_qid_unique[0])], mode='Train')

    # train_dmatrix = xgd.DMatrix(batch_train_em, batch_train_label)

    '''
    xgb_model = xgb.train(params, train_dmatrix, num_boost_round=4,
                      evals=[(valid_dmatrix, 'validation')])

    pred = xgb_model.predict(test_dmatrix)
    '''



