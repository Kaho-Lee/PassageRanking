import os
import numpy as np
import json
import matplotlib.pyplot as plt
import tqdm
import csv
import pandas as pd
from torch.utils import data
import time
import re
from utils import *
from nltk.stem import PorterStemmer, SnowballStemmer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from nltk.tokenize import word_tokenize 
from DataGenerator import dataPipeLine
#nltk.download('punkt')

class LogisticRegressor:
    def __init__(self, parameters):
        self.parameters = {}
        self.name = 'LR'
        self.parameters['weights'] = np.random.normal(0, 1, 3)
        #self.parameters['weights'] = np.zeros(3)
        updateDict(parameters, self.parameters)

    def sigmoid_loss(self, y, theta):
        loss = np.negative(y*np.log(sigmoid(theta)) + (1-y)*np.log(1-sigmoid(theta)))
        # print(loss.shape)
        
        return np.sum(loss)

    def sigmoid_loss_grad(self, y, x, theta):
        grad = x.T @ (sigmoid(theta) - y)
        return np.sum(grad)

    def train(self, dataGenerator, batch_size = 300):
        
        #self.weights = np.zeros(100)
        
        maxIteration = 100
        tolerance = 0.001
        prev_loss = 0
        cur_iteration = 0
        eta = 0.0001
        print('Learning rate: {} maxIteration: {}'.format(eta, maxIteration))

        allRows = np.arange(1, dataGenerator.getLength()+1)
        choices = dataGenerator.randomChoice_unbalancedData(batch_size)
        x, y = dataGenerator.getitem(choices)
        theta = x @ np.atleast_2d(self.parameters['weights']).T
        loss = self.sigmoid_loss(y, theta)
        loss_grad = self.sigmoid_loss_grad(y, x, theta)
        loss_lst = [loss]

        y_pred = np.round(self.predict(x))
        print('Batch: x {} y {}'.format(x.shape, y.shape))
        acc = accuracy(y, y_pred)
        print('Iteration {}: current loss is {}, accuracy is {}'.format(cur_iteration, loss, acc))
        cur_iteration += 1
    

        while abs(prev_loss - loss)/abs(loss) and cur_iteration <= maxIteration:
            
            self.parameters['weights'] = self.parameters['weights'] - eta*loss_grad
            prev_loss = loss

            choices = dataGenerator.randomChoice_unbalancedData(batch_size)
            x, y = dataGenerator.getitem(choices)
            theta = x @ np.atleast_2d(self.parameters['weights']).T
            loss = self.sigmoid_loss(y, theta)
            loss_grad = self.sigmoid_loss_grad(y, x, theta)

            if cur_iteration % 10 == 0:
                choices = dataGenerator.randomChoice_unbalancedData(batch_size)
                x, y = dataGenerator.getitem(choices)
                y_pred = np.round(self.predict(x))
                print('Batch: x {} y {}'.format(x.shape, y.shape))
                acc = accuracy(y, y_pred)
                print('Iteration {}: current loss is {}, accuracy is {}'.format(cur_iteration, loss, acc))

            cur_iteration += 1
            loss_lst.append(loss)

        temp = {}
        temp[eta] = {}
        temp[eta]['weights'] = list(self.parameters['weights'])
        temp[eta]['loss'] = loss_lst
        
        with open('LR_weight.json', 'w') as writeFile:
            json.dump(temp, writeFile)
        writeFile.close()

    def train2(self, dataGenerator, batch_size = 500):
        #using sklearn Logistic Regression API to check correctness of LR implementation above
        
        #self.weights = np.zeros(100)
        
        maxIteration = 100
        tolerance = 0.001
        prev_loss = 0
        cur_iteration = 0
        eta = 0.01
        clf = LogisticRegression(random_state=0, max_iter=10, multi_class='ovr', warm_start=True)

        allRows = np.arange(1, dataGenerator.getLength()+1)
        choices = dataGenerator.randomChoice_unbalancedData(batch_size)
        x, y = dataGenerator.getitem(choices)
        clf.fit(x,y)
        # loss = np.sum(clf.predict_log_proba(x))
        loss_lst = []
        i = 0
        while  i <= maxIteration:
            # print(cur_iteration)
            
            print('check param estimator')
            print(clf.coef_)

            choices = dataGenerator.randomChoice_unbalancedData(batch_size)
            x, y = dataGenerator.getitem(choices)
            y_pred = np.atleast_2d(clf.predict(x)).T
            acc = accuracy(y, y_pred)
            loss = log_loss(y.T, y_pred.T)
            print('Iteration {}:  loss is {}, accuracy is {}'.format(i, loss, acc))

            choices = dataGenerator.randomChoice_unbalancedData(batch_size)
            x, y = dataGenerator.getitem(choices)
            clf.fit(x,y)
            # loss = np.sum(clf.predict_log_proba(x))
            i+= 10
            loss_lst.append(loss)

            print('pnew aram estimator')
            print(clf.coef_)

        temp = {}
        temp[eta] = {}
        temp[eta]['weights'] = list(self.parameters['weights'])
        temp[eta]['loss'] = loss_lst
        
    
    def predict(self, x):
        # print('x.shape ', x.shape)
        # print('weight shape', self.parameters['weights'].shape)
        theta = x @ np.atleast_2d(self.parameters['weights']).T
        sigmoid_logit = sigmoid(theta)
        y_pred = sigmoid_logit

        return y_pred

    def saveToText(self, queryID, pid, y_pred):
        candidate_ranked_pass = {}
        for i, score in zip(pid, y_pred):
            # print(i, score)
            candidate_ranked_pass[str(i[0])] = score[0]

        candidate_ranked_pass = {k: v for k, v in sorted(candidate_ranked_pass.items(), \
        key=lambda item: item[1], reverse=True)}

        if len(candidate_ranked_pass.keys()) <= 100:
            reranked_candidate = candidate_ranked_pass
        else:
            reranked_candidate = {k: candidate_ranked_pass[k] for k in list(candidate_ranked_pass)[:100]}
        
        with open('{}.txt'.format(self.name), 'a') as writeFile:
            rank = 1
            for key, value in zip(reranked_candidate.keys(), reranked_candidate.values()):
                writeFile.write('<{} A1 {} {} {} {}>\n'.format(queryID, key, rank, value, self.name))
                rank += 1
        writeFile.close()
        return reranked_candidate


if __name__=="__main__":
    init = False

    
    pre_trained_model_path = '/Users/leekaho/Desktop/part2/glove/glove.6B.50d.txt'
    gloveEmbedding_path = '/Users/leekaho/Desktop/part2/gloveEmbedding.json'
    train_path = '/Users/leekaho/Desktop/part2/train_data.tsv'
    val_path = '/Users/leekaho/Desktop/part2/validation_data.tsv'
    idf_val = '/Users/leekaho/Desktop/part2/val_idf.json'
    idf_train = '/Users/leekaho/Desktop/part2/train_idf.json'

    if init:
        ExtractPreTrained(pre_trained_model_path, gloveEmbedding_path)
        IDFOfDataSet(train_path, idf_train)
        IDFOfDataSet(val_path, idf_val)

    data = dataPipeLine(gloveEmbedding_path, train_path, idf_train)
    
    parameters = {}
    mode = 'train'
    if mode == 'train':
        data.getSimProperty()
        relevancy_LR = LogisticRegressor(parameters)
        relevancy_LR.train(data)
    elif mode == 'test':
        with open('LR_weight.json', 'r') as readFile:
            saved_model = json.load(readFile)
        readFile.close()
        # print(saved_model)
        relevancy_LR = LogisticRegressor(saved_model['0.0001'])
        # a = data.randomChoice_unbalancedData( 10)
        # print(a)


    with open(gloveEmbedding_path, 'r') as readFile:
            embedding = json.load(readFile)
    readFile.close()
    val_reader = pd.read_csv(val_path, sep='\t')

    #reranking
    src_path = '/Users/leekaho/Desktop/part2/'
    raw_path = 'query_passage_raw.json'
    with open(src_path + raw_path, 'r') as readFile:
        temp = json.load(readFile)
        query_raw = temp['query']
        labels = temp['query_pass']
    readFile.close()
    del temp

    with open(idf_val, 'r') as readFile:
            val_idf = json.load(readFile)
    readFile.close()

    avg_precision = {}
    ndcg = {}


    for query_id, query_value in zip(query_raw.keys(), query_raw.values()):
        print(query_id, query_value)
        query_pass_pid, query_pass_em = Data_Embedding(embedding, val_reader, query_id, val_idf[str(query_id)])
        y_pred = relevancy_LR.predict(query_pass_em)
        reranked_candidate = relevancy_LR.saveToText(query_id, query_pass_pid, y_pred)
        avg_precision[query_id] = avg_Precision(reranked_candidate, labels[query_id])
        ndcg[query_id] = NDCG(query_id, reranked_candidate, labels[query_id])
        # break
        

    print('Mean value of the average precision is {}'.format(np.mean(list(avg_precision.values()))))
    metricToText(avg_precision, 'LR_Average_Precision')
    print('Mean value of NDCG is {}'.format(np.mean(list(ndcg.values()))))
    metricToText(ndcg, 'LR_NDCG')

