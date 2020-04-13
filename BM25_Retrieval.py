import numpy as np
import os
import re
import threading
import sys
import json
import time
import csv
import matplotlib.pyplot as plt 
from Retrivel_engine import BM25
from nltk.stem import PorterStemmer, SnowballStemmer
import tqdm
from utils import *
from Evaluation_Metrics import *


if __name__=='__main__':

    #execute the reranking tasks, BM25
    pass_invIndex_path = 'pass_invIndex.json'
    raw_path = 'query_passage_raw.json'
    query_invIndex_path = 'query_invIndex.json'
    src_path = '../Data/'
    retrivel_engine = BM25(pass_invIndex_path, raw_path, query_invIndex_path, src_path)
    with open(src_path + raw_path, 'r') as readFile:
        temp = json.load(readFile)
        query_raw = temp['query']
        labels = temp['query_pass']
    readFile.close()
    del temp

    avg_precision = {}
    ndcg = {}

    for query_id, query_value in zip(query_raw.keys(), query_raw.values()):
        rerank_candidate = retrivel_engine.retrivel(query_id, query_value, show=False)
        retrivel_engine.saveToText(query_id, rerank_candidate)
        avg_precision[query_id] = avg_Precision(rerank_candidate, labels[query_id])
        ndcg[query_id] = NDCG(query_id, rerank_candidate, labels[query_id])
        #break

    print('Mean value of the average precision is {}'.format(np.mean(list(avg_precision.values()))))
    metricToText(avg_precision, 'BM25_Average_Precision')
    print('Mean value of NDCG is {}'.format(np.mean(list(ndcg.values()))))
    metricToText(ndcg, 'BM25_NDCG')
