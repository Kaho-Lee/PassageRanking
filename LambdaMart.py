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
from DataGenerator import dataPipeLine

def batchDMatrix(sampled_data_qid, data, downSampling=False):
    group = []
    batch_label = []
    batch_em = []
    for i, qid in enumerate(sampled_data_qid):
        # print(i, qid)     
        cur_qid_label, cur_qid_em = data.getItemByGroup(qid, mode='Train', downSampling=downSampling, raw=False)
        # print('shape ', cur_qid_em.shape)
        batch_label.append( cur_qid_label)
        batch_em.append( cur_qid_em[:, 1:])
        group.append(cur_qid_em.shape[0])

    # batch_em = np.stack(batch_em, axis=0)
    batch_em = np.concatenate(batch_em, axis=0)
    print('shape em ', batch_em.shape)
    # batch_label = np.stack(batch_label, axis=0)
    batch_label = np.concatenate(batch_label, axis=0)
    print('shape label ', batch_label.shape)

    dmatrix = xgb.DMatrix(batch_em, label = batch_label)
    dmatrix.set_group(group)
    print('Gnerating Data Shape em: {}, label: {}'.format(batch_em.shape, batch_label.shape))
    return dmatrix

def randomSampleKfold(train_qid_unique, qid_index, ratio=0.05):
    # print(len(qid_index))
    # print(len(qid_index), int(ratio*len(qid_index)), np.random.choice(len(qid_index), int(ratio*len(qid_index)), replace=False))
    choice = np.random.choice(len(qid_index), int(ratio*len(qid_index)), replace=False)
    sampled_index = [qid_index[x] for x in choice]
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

    for combi in itertools.product(eta_lst,  gamma_lst, depth_lst, child_lst):
        param['eta'] = combi[0]
        param['gamma'] = combi[1]
        param['max_depth'] = combi[2]
        param['min_child_weight'] = combi[3]
        # param['num_boost_round'] = combi[4]
        yield param

# def paramSelection(parameters, train_path, gloveEmbedding_path, idf_train):
def paramSelection(parameters, train_data):
    paramGen = paramGenerator(parameters)
    print('Candidate Parameters Setting are:')
    for p in paramGen:
        print(p)

    train_qid_unique = train_data.getReader().qid.unique().tolist()

    #from xgb sample code https://github.com/dmlc/xgboost/blob/master/demo/rank/rank.py
    # params_ndcg_e1 = {'objective': 'rank:ndcg', 'eta': 0.1, 'gamma': 1.0,
    #       'min_child_weight': 0.1, 'max_depth': 6, 'eval_metric':['ndcg', 'map']}


    kf = KFold(n_splits=5, random_state=None, shuffle=False)


    # with open(gloveEmbedding_path, 'r') as readFile:
    #     embedding = json.load(readFile)
    # readFile.close()

    # with open(idf_train, 'r') as readFile:
    #     train_idf = json.load(readFile)
    # readFile.close()

     #from xgb sample code https://github.com/dmlc/xgboost/blob/master/demo/rank/rank_sklearn.py
    # params = {'objective': 'rank:ndcg', 'learning_rate': 0.1,
    #       'gamma': 1.0, 'min_child_weight': 0.1,
    #       'max_depth': 6, 'n_estimators': 4}
    # passRank = xgb.sklearn.XGBRanker(**params) 
    
    maxIteration = 2
    batchsize = 300
    
    best_model = None
    best_param = None
    cur_bestNDGC = 0

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
            train_dmatrix = batchDMatrix(train_data_qid, train_data, downSampling=True)
            print('Eval DMatrix, num ', len(eval_data_qid))
            eval_dmatrix = batchDMatrix(eval_data_qid, train_data, downSampling=False)

            passRank = xgb.train(params, train_dmatrix, num_boost_round=500, early_stopping_rounds =10,
                    evals=[(eval_dmatrix, 'validation')], verbose_eval=False, evals_result = eval_ndcg, xgb_model=None )
            
            print('Metrics History are {} '.format( eval_ndcg['validation']['ndcg']))
            #avg_ndcg += eval_ndcg['validation']['ndcg'][-1]
            avg_ndcg += passRank.best_score
        
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

    return best_param

# def train(parameter, train_path, gloveEmbedding_path, idf_train):
def train(parameter, train_data):

    print('Training, param is ', parameter)

    # train_reader = pd.read_csv(train_path, sep='\t')


    qid_unique = train_data.getReader().qid.unique().tolist()
    test_index = list(np.random.choice(len(qid_unique), int(len(qid_unique)*0.1), replace=False))
    # print(test_index)
    all_choice = np.arange(len(qid_unique))
    train_index = [int(x) for x in all_choice if x not in test_index]
    # print(train_index)
    
    #from xgb sample code https://github.com/dmlc/xgboost/blob/master/demo/rank/rank.py
    # params_ndcg_e1 = {'objective': 'rank:ndcg', 'eta': 0.1, 'gamma': 1.0,
    #       'min_child_weight': 0.1, 'max_depth': 6, 'eval_metric':['ndcg', 'map']}


    # with open(gloveEmbedding_path, 'r') as readFile:
    #     embedding = json.load(readFile)
    # readFile.close()

    # with open(idf_train, 'r') as readFile:
    #     train_idf = json.load(readFile)
    # readFile.close()

     #from xgb sample code https://github.com/dmlc/xgboost/blob/master/demo/rank/rank_sklearn.py
    # params = {'objective': 'rank:ndcg', 'learning_rate': 0.1,
    #       'gamma': 1.0, 'min_child_weight': 0.1,
    #       'max_depth': 6, 'n_estimators': 4}
    # passRank = xgb.sklearn.XGBRanker(**params) 
    
    maxIteration = 1
    batchsize = 300
    
    batch = 0
    passRank = None
    avg_ndcg = 0
    for i in range(maxIteration):
        # print('Train Fold Length: {} Eval Fold Length: {}'.format(len(train_qid_index), len(eval_qid_index)))
        batch += 1
        eval_ndcg = {}
        print('Step {}'.format(i))
        
        train_data_qid = randomSampleKfold(qid_unique, train_index, ratio=0.3)
        eval_data_qid = randomSampleKfold(qid_unique, test_index, ratio=0.3)

        # train_data_qid = [train_qid_unique[x] for x in train_qid_index]
        # eval_data_qid = [train_qid_unique[x] for x in eval_qid_index]

        print('Train DMatrix, num ', len(train_data_qid))
        train_dmatrix = batchDMatrix(train_data_qid, train_data, downSampling=False)
        print('Eval DMatrix, num ', len(eval_data_qid))
        eval_dmatrix = batchDMatrix(eval_data_qid, train_data, downSampling=False)
        # if passRank != None:


        passRank = xgb.train(parameter, train_dmatrix, num_boost_round=1000, early_stopping_rounds =30,
                evals=[(train_dmatrix, 'train'), (eval_dmatrix, 'validation')], verbose_eval=True, evals_result = eval_ndcg, xgb_model=passRank)
        
        # print('Metrics History are {} '.format( eval_ndcg['validation']['ndcg']))
        print('early stop return Iter {} Score {} num_tree {}'.format(passRank.best_iteration, passRank.best_score, passRank.best_ntree_limit))

    passRank.save_model('LambdaMart.model')

    return passRank



if __name__=="__main__":

    parameters = {
    'max_depth': [6, 10, 15],
    'eta': [ 0.01, 0.1, 0.5],
    'min_child_weight': [0.1, 5, 10],
    'gamma': [0.1, 0.5, 1.0]
    }

    parameters = {
    'max_depth': [6, 15],
    'eta': [ 0.15],
    'min_child_weight': [1, 5 ],
    'gamma': [1.0]
    }#discard eta=0.5 above

    # parameters = {
    # 'max_depth': [6, 10],
    # 'eta': [0.1, 0.5],
    # 'min_child_weight': [0.1],
    # 'gamma': [0.5, 1.0]
    # }

    
    # a = t

    gloveEmbedding_path = '/Users/leekaho/Desktop/part2/gloveEmbedding.json'
    train_path = '/Users/leekaho/Desktop/part2/train_data.tsv'
    val_path = '/Users/leekaho/Desktop/part2/validation_data.tsv'
    idf_val = '/Users/leekaho/Desktop/part2/val_idf.json'
    idf_train = '/Users/leekaho/Desktop/part2/train_idf.json'

    train_data = dataPipeLine(gloveEmbedding_path, train_path, idf_train)

    selectModel = False
    if selectModel:
        # best_param = paramSelection(parameters, train_path, gloveEmbedding_path, idf_train)
        best_param = paramSelection(parameters, train_data)
    else:
        best_param =  {'objective': 'rank:ndcg', 'eta': 0.5, 'gamma': 1.0,
          'min_child_weight': 5, 'max_depth': 6, 'eval_metric':['map', 'ndcg']} #to be determined

    
    isTrain = False
    if isTrain:
        train(best_param , train_data)

    passRank = xgb.Booster()  # init model
    passRank.load_model('LambdaMart_all.model')  # load data

    # print('Best xgb ranker model ', best_model)
    '''
    pred = xgb_model.predict(test_dmatrix)
    '''

    val_data = dataPipeLine(gloveEmbedding_path, val_path, idf_val)

    #reranking
    src_path = '/Users/leekaho/Desktop/part2/'
    raw_path = 'query_passage_raw.json'
    with open(src_path + raw_path, 'r') as readFile:
        temp = json.load(readFile)
        query_raw = temp['query']
        labels = temp['query_pass']
    readFile.close()
    del temp

    avg_precision = {}
    ndcg = {}

    try:
        os.remove('LM.txt')
    except OSError:
        pass

    for query_id, query_value in zip(query_raw.keys(), query_raw.values()):
        print(query_id, query_value)
        query_pass_pid, query_pass_em = val_data.getItemByGroup(query_id)
        dmatrix = xgb.DMatrix(query_pass_em[:, 1:])
        y_pred = np.atleast_2d( passRank.predict(dmatrix)).T
        # print(y_pred)
        reranked_candidate = saveToText('LM', query_id, query_pass_pid, y_pred)
        avg_precision[query_id] = avg_Precision(reranked_candidate, labels[query_id])
        ndcg[query_id] = NDCG(query_id, reranked_candidate, labels[query_id])
        # break
        

    print('Mean value of the average precision is {}'.format(np.mean(list(avg_precision.values()))))
    avg_precision['mean'] = np.mean(list(avg_precision.values()))
    metricToText(avg_precision, 'LM_Average_Precision')
    print('Mean value of NDCG is {}'.format(np.mean(list(ndcg.values()))))
    ndcg['mean'] = np.mean(list(ndcg.values()))
    metricToText(ndcg, 'LM_NDCG')



