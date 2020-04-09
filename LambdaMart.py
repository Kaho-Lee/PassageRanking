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
from random import shuffle
import itertools

def batchDMatrix(reader, sampled_data_qid, embedding, idf, downSampling=False):
    group = []
    for i, qid in enumerate(sampled_data_qid):
        # print(i, qid)
        
        cur_qid_label, cur_qid_em = Data_Embedding(embedding, reader, qid, idf[str(qid)], mode='Train', downSampling=downSampling)
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

def randomSampleKfold(qid_unique, qid_index, ratio=0.05):
    # print(len(qid_index))
    sampled_index = qid_index[np.random.choice(len(qid_index), int(ratio*len(qid_index)), replace=False)]
    # print(len(sampled_index) )
  
    data_qid = [train_qid_unique[x] for x in sampled_index]
    # print(train_data_qid, eval_data_qid)

    return data_qid

def paramGenerator(param_dict):
    param = {'objective': 'rank:ndcg', 'eval_metric':'ndcg'}

    eta_lst = param_dict['eta']
    gamma_lst = param_dict['gamma']
    depth_lst = param_dict['max_depth']
    child_lst = param_dict['min_child_weight']
    # boostRound_lst = param_dict['num_boost_round']

    for combi in itertools.product(eta_lst,  gamma_lst, depth_lst, child_lst):
        param['eta'] = combi[0]
        param['gamma'] = combi[1]
        param['max_depth'] = combi[2]
        param['min_child_weight'] = combi[3]
        # param['num_boost_round'] = combi[4]
        yield param

# def paramSelection(parameters, )

if __name__=="__main__":

    parameters = {
    'max_depth': [6, 10, 15],
    'eta': [ 0.01, 0.1, 0.5],
    'min_child_weight': [0.1, 5, 10],
    'gamma': [0.1, 0.5, 1.0]
    }

    # parameters = {
    # 'max_depth': [6, 10],
    # 'eta': [0.1, 0.5],
    # 'min_child_weight': [0.1],
    # 'gamma': [0.5, 1.0]
    # }

    paramGen = paramGenerator(parameters)
    print('Candidate Parameters Setting are:')
    for p in paramGen:
        print(p)
    # a = t

    train_path = '/Users/leekaho/Desktop/part2/train_data.tsv'
    val_path = '/Users/leekaho/Desktop/part2/validation_data.tsv'
    gloveEmbedding_path = '/Users/leekaho/Desktop/part2/gloveEmbedding.json'
    idf_train = '/Users/leekaho/Desktop/part2/train_idf.json'

    train_reader = pd.read_csv(train_path, sep='\t')
    train_qid_unique = train_reader.qid.unique().tolist()

    #from xgb sample code https://github.com/dmlc/xgboost/blob/master/demo/rank/rank.py
    # params_ndcg_e1 = {'objective': 'rank:ndcg', 'eta': 0.1, 'gamma': 1.0,
    #       'min_child_weight': 0.1, 'max_depth': 6, 'eval_metric':['ndcg', 'map']}


    kf = KFold(n_splits=5, random_state=None, shuffle=False)


    with open(gloveEmbedding_path, 'r') as readFile:
        embedding = json.load(readFile)
    readFile.close()

    with open(idf_train, 'r') as readFile:
        train_idf = json.load(readFile)
    readFile.close()

     #from xgb sample code https://github.com/dmlc/xgboost/blob/master/demo/rank/rank_sklearn.py
    # params = {'objective': 'rank:ndcg', 'learning_rate': 0.1,
    #       'gamma': 1.0, 'min_child_weight': 0.1,
    #       'max_depth': 6, 'n_estimators': 4}
    # passRank = xgb.sklearn.XGBRanker(**params) 
    
    maxIteration = 2
    batchsize = 300
    
    best_model = None
    best_param = None
    cur_bestNDGC = 0.0

    paramGen = paramGenerator(parameters)

    for params in paramGen:
        print('current setting ', params)
        batch = 0
        passRank = None
        avg_ndcg = 0
        for train_qid_index, eval_qid_index in kf.split(train_qid_unique):
            # print('Train Fold Length: {} Eval Fold Length: {}'.format(len(train_qid_index), len(eval_qid_index)))
            batch += 1
            eval_ndcg = {}
            print('Batch {}'.format(batch))
            
            train_data_qid = randomSampleKfold(train_qid_unique, train_qid_index, ratio=0.15)
            eval_data_qid = randomSampleKfold(train_qid_unique, eval_qid_index)

            # train_data_qid = [train_qid_unique[x] for x in train_qid_index]
            # eval_data_qid = [train_qid_unique[x] for x in eval_qid_index]

            print('Train DMatrix, num ', len(train_data_qid))
            train_dmatrix = batchDMatrix(train_reader, train_data_qid, embedding, train_idf, downSampling=True)
            print('Eval DMatrix, num ', len(eval_data_qid))
            eval_dmatrix = batchDMatrix(train_reader, eval_data_qid, embedding, train_idf, downSampling=False)

            passRank = xgb.train(params, train_dmatrix, num_boost_round=200, early_stopping_rounds =10,
                    evals=[(eval_dmatrix, 'validation')], verbose_eval=False, evals_result = eval_ndcg, xgb_model=None )
            
            print('Metrics History are {} '.format( eval_ndcg['validation']['ndcg']))
            avg_ndcg += eval_ndcg['validation']['ndcg'][-1]
        
        avg_ndcg = avg_ndcg/5
        print('current avg ndcg {}, best avg ndcg {}'.format(avg_ndcg, cur_bestNDGC))
        if avg_ndcg > cur_bestNDGC:
            print('Best Model Update: ', params)
            # best_model = passRank
            best_param = params
            cur_bestNDGC = avg_ndcg
        print('\n')  

            # break

    print('Best params setting ', best_param)
    print('Best NDCG ', cur_bestNDGC)
    # print('Best xgb ranker model ', best_model)
    '''

    pred = xgb_model.predict(test_dmatrix)
    '''



