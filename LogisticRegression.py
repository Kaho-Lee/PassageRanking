import os
import numpy as np
import json
import matplotlib.pyplot as plt
import tqdm
import csv
import pandas as pd
import time
import re
from utils import *
from nltk.stem import PorterStemmer, SnowballStemmer
from DataGenerator import dataPipeLine
from Evaluation_Metrics import *

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

    def train(self, dataGenerator, batch_size = 800):
        
        maxIteration = 200
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
                y_pred = np.round(self.predict(x))
                print('Batch: x {} y {}, ratio {}'.format(x.shape, y.shape, np.sum(y)))
                acc = accuracy(y, y_pred)
                print('Iteration {}: Train current loss is {}, Training accuracy is {}'.format(cur_iteration, loss, acc))

            cur_iteration += 1
            loss_lst.append(loss)

        temp = {}
        temp[eta] = {}
        temp[eta]['weights'] = list(self.parameters['weights'])
        temp[eta]['loss'] = loss_lst

        try:
            os.remove('LR_weight.json')
        except OSError:
            pass
        
        with open('LR_weight.json', 'w') as writeFile:
            json.dump(temp, writeFile)
        writeFile.close()
        
    
    def predict(self, x):
        # print('x.shape ', x.shape)
        # print('weight shape', self.parameters['weights'].shape)
        theta = x @ np.atleast_2d(self.parameters['weights']).T
        sigmoid_logit = sigmoid(theta)
        y_pred = sigmoid_logit

        return y_pred


if __name__=="__main__":
    root_path =  '../Data/'
    pre_trained_model_path = root_path+'glove.6B.50d.txt'
    gloveEmbedding_path = root_path+'gloveEmbedding.json'
    train_path = root_path+'train_data.tsv'
    val_path = root_path+'validation_data.tsv'
    idf_val = root_path+'val_idf.json'
    idf_train = root_path+'train_idf.json'

    data = dataPipeLine(gloveEmbedding_path, train_path, idf_train)

    
    parameters = {}
    mode = 'train'
    if mode == 'train':
        # data.getSimProperty()
        relevancy_LR = LogisticRegressor(parameters)
        relevancy_LR.train(data)
    elif mode == 'test':
        with open('LR_weight.json', 'r') as readFile:
            saved_model = json.load(readFile)
        readFile.close()
        # print(saved_model)
        relevancy_LR = LogisticRegressor(saved_model['0.0001'])

    val_data = dataPipeLine(gloveEmbedding_path, val_path, idf_val)

    #reranking
    
    raw_path = 'query_passage_raw.json'
    with open(root_path + raw_path, 'r') as readFile:
        temp = json.load(readFile)
        query_raw = temp['query']
        labels = temp['query_pass']
    readFile.close()
    del temp

    avg_precision = {}
    ndcg = {}

    try:
        os.remove('LR_TFIDF.txt')
    except OSError:
        pass

    for query_id, query_value in zip(query_raw.keys(), query_raw.values()):
        print(query_id, query_value)
        query_pass_pid, query_pass_em = val_data.getItemByGroup(query_id)
        y_pred = relevancy_LR.predict(query_pass_em)
        reranked_candidate = saveToText('LR_TFIDF', query_id, query_pass_pid, y_pred)
        avg_precision[query_id] = avg_Precision(reranked_candidate, labels[query_id])
        ndcg[query_id] = NDCG(query_id, reranked_candidate, labels[query_id])
        # break
        

    print('Mean value of the average precision is {}'.format(np.mean(list(avg_precision.values()))))
    avg_precision['mean'] = np.mean(list(avg_precision.values()))
    metricToText(avg_precision, 'LR_TFIDF_Average_Precision')
    print('Mean value of NDCG is {}'.format(np.mean(list(ndcg.values()))))
    ndcg['mean'] = np.mean(list(ndcg.values()))
    metricToText(ndcg, 'LR_TFIDF_NDCG')

