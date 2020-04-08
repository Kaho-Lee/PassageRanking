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

def randomSampleKfold(qid_unique, qid_index, ratio=0.05):
    # print(len(qid_index))
    # sampled_train_index = train_qid_index[np.random.choice(len(train_qid_index), int(0.05*len(train_qid_index)), replace=True)]
    sampled_index = qid_index[np.random.choice(len(qid_index), int(ratio*len(qid_index)), replace=False)]
    # print(len(sampled_index) )

    
    # train_data_qid = [train_qid_unique[x] for x in sampled_train_index]
    data_qid = [train_qid_unique[x] for x in sampled_index]
    # print(train_data_qid, eval_data_qid)

    return data_qid

if __name__=="__main__":

    # test = '-lepsy-leptic-leptic-less-less-less-lessly-lessness-let-let-let-let-let-leukemia-leukin-lexia-lexis-lexy-like-like-limbed-limbed-limbed-lined-ling-ling-ling-ling-lipoma-lipotropin-lipped'
    # passage_lst = word_tokenize(test.lower())
    # print(passage_lst)
    # aa= tt

    train_path = '/Users/leekaho/Desktop/part2/train_data.tsv'
    val_path = '/Users/leekaho/Desktop/part2/validation_data.tsv'
    gloveEmbedding_path = '/Users/leekaho/Desktop/part2/gloveEmbedding.json'
    idf_train = '/Users/leekaho/Desktop/part2/train_idf.json'

    train_reader = pd.read_csv(train_path, sep='\t')
    train_qid_unique = train_reader.qid.unique().tolist()

    #from xgb sample code https://github.com/dmlc/xgboost/blob/master/demo/rank/rank.py
    params_ndcg_e1 = {'objective': 'rank:ndcg', 'eta': 0.1, 'gamma': 1.0,
          'min_child_weight': 0.1, 'max_depth': 6, 'eval_metric'='ndcg'}
    params_ndcg_e2 = {'objective': 'rank:ndcg', 'eta': 0.05, 'gamma': 1.0,
          'min_child_weight': 0.1, 'max_depth': 6, 'eval_metric'='ndcg'}
    params_ndcg_e = {'objective': 'rank:ndcg', 'eta': 1, 'gamma': 1.0,
          'min_child_weight': 0.1, 'max_depth': 6, 'eval_metric'='ndcg'}
    params_ndcg_d9 = {'objective': 'rank:ndcg', 'eta': 0.1, 'gamma': 0.1,
        'min_child_weight': 0.1, 'max_depth': 9, 'eval_metric'='ndcg'}
    params_ndcg_g_e1 = {'objective': 'rank:ndcg', 'eta': 1, 'gamma': 0.1,
          'min_child_weight': 0.1, 'max_depth': 6, 'eval_metric'='ndcg'}
    
    params_pairwise_e1 = {'objective': 'rank:pairwise', 'eta': 0.1, 'gamma': 1.0,
          'min_child_weight': 0.1, 'max_depth': 6}
    params_pairwise_e2 = {'objective': 'rank:pairwise', 'eta': 0.05, 'gamma': 1.0,
          'min_child_weight': 0.1, 'max_depth': 6}


    # params_dict = {'params_ndcg_e': params_ndcg_e, 'params_ndcg_e1': params_ndcg_e1, 
    # 'params_ndcg_e2':params_ndcg_e2, 'params_pairwise_e1':params_pairwise_e1, 'params_pairwise_e2':params_pairwise_e2}

    params_dict = {'params_ndcg_e': params_ndcg_e, 'params_ndcg_e1': params_ndcg_e1, 
    'params_ndcg_e2':params_ndcg_e2, 'params_ndcg_d9': params_ndcg_d9, 'params_ndcg_g_e1': params_ndcg_g_e1}

    # params_dict = {'params_ndcg_e': params_ndcg_e}

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
    cur_acc = 0.0

    for params_name in params_dict:
        print('current setting ', params_name)
        params = params_dict[params_name]
        batch = 0
        passRank = None
        for train_qid_index, eval_qid_index in kf.split(train_qid_unique):
            # print('Train Fold Length: {} Eval Fold Length: {}'.format(len(train_qid_index), len(eval_qid_index)))
            batch += 1
            eval_ndcg = {}
            for i in range(maxIteration):
                print('Batch {}, Iteration {}'.format(batch, i+1))
                # shuffle(train_qid_index)
                # for start in range(0, len(train_qid_index), batchsize):

                    # if start + batchsize > len(train_qid_index):
                    #     train_data_qid = [train_qid_unique[x] for x in train_qid_index[start:]]
                    #     # print('len of train {}'.format(len(train_data_qid))
                    # else:
                    #     train_data_qid = [train_qid_unique[x] for x in train_qid_index[start:start+batchsize]]
                    #     # print('len of train {}'.format(len(train_data_qid))
                
                train_data_qid = randomSampleKfold(train_qid_unique, train_qid_index)
                eval_data_qid = randomSampleKfold(train_qid_unique, eval_qid_index)

                print('Train DMatrix')
                train_dmatrix = batchDMatrix(train_reader, train_data_qid, embedding, train_idf)
                print('Eval DMatrix')
                eval_dmatrix = batchDMatrix(train_reader, eval_data_qid, embedding, train_idf)

                passRank = xgb.train(params, train_dmatrix,
                        evals=[(eval_dmatrix, 'validation')], evals_result = eval_ndcg, xgb_model=passRank )
            
            print('Batch {}, Metrics History are {} '.format(i, eval_ndcg['validation'], eval_ndcg['validation']['map'][-1]))
        
        if eval_ndcg['validation']['map'][-1] > cur_acc:
            best_model = passRank
            best_param = params


            # break

    print('Best params setting ', best_param)
    print('Best xgb ranker model ', best_model)
    '''
    

    pred = xgb_model.predict(test_dmatrix)
    '''



