import os
import numpy as np
import json
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import tqdm
import csv
import pandas as pd
import time
import re
from nltk.stem import PorterStemmer, SnowballStemmer
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from utils import *

class dataPipeLine:

    def __init__(self, embedding_path, train_path, idf_path):

        with open(embedding_path, 'r') as readFile:
            self.embedding = json.load(readFile)
        readFile.close()
        
        with open(train_path, 'r') as readFile:
            self.num_line = sum(1 for line in readFile) -1
        readFile.close()

        with open(idf_path, 'r') as readFile:
            self.idf = json.load(readFile)
        readFile.close()

        self.train_path = train_path
        self.num_line = self.__length__(train_path)
        print('num lines ', self.num_line)
        
        self.reader = pd.read_csv(self.train_path, sep='\t')
        relevancy_lines = self.reader[self.reader.relevancy.eq(1.0)]
        #print(relevancy_lines)
        self.relavancePass = relevancy_lines.index.tolist()
        non_relavancy_line = self.reader[self.reader.relevancy.eq(0.0)]
        self.pos = relevancy_lines.shape[0]/self.num_line
        print('pos ', self.pos)
        self.neg = non_relavancy_line.shape[0]/self.num_line
        print('neg ', self.neg)

    
    def __length__(self, train_path):
        with open(train_path, 'r') as readFile:
            num_line = sum(1 for line in readFile) -1
        readFile.close()

        return num_line

    def getLength(self):
        return self.num_line

    def getEmbedding(self):
        return self.embedding

    def getSimProperty(self):
        # reader = pd.read_csv(self.train_path, sep='\t')
        #print(reader)
        relevancy_lines = self.reader[self.reader.relevancy.eq(1.0)]
        cos_sum = 0
        count = 0 
        dist_sum = 0
        tf_idf_sum = 0
        relavancePot = []
        print('cal relavence sim')
        for i, row in relevancy_lines.iterrows():

            query_raw = row['queries'].lower()
            passage_raw = row['passage'].lower()

            qid = str(row['qid'])
            query_passage_embedding = generateEmbedding(self.embedding, query_raw, passage_raw, self.idf[qid])

            cos_sum += query_passage_embedding[1]
            # dist_sum += query_passage_embedding[0,2]
            tf_idf_sum += query_passage_embedding[2]
            count += 1
            relavancePot.append(query_passage_embedding[1:])
        relavancePot = np.array(relavancePot)
        num_relavance = relavancePot.shape[0]
        print(relavancePot.shape)
        self.relavanceCos = cos_sum/count
        # self.relavanceDist = dist_sum/count
        self.relavanceTFIDF = tf_idf_sum/count

        relevancy_lines = self.reader[self.reader.relevancy.eq(0.0)]
        cos_sum = 0
        count = 0 
        dist_sum = 0
        tf_idf_sum = 0
        non_relavancePot = []
        print('cal non relavence sim')
        sample_reader = relevancy_lines.sample(int(relevancy_lines.shape[0]*0.005))
        for i, row in sample_reader.iterrows():

            query_raw = row['queries'].lower()
            passage_raw = row['passage'].lower()

            qid = str(row['qid'])
            query_passage_embedding = generateEmbedding(self.embedding, query_raw, passage_raw, self.idf[qid])

            cos_sum += query_passage_embedding[1]
            # dist_sum += query_passage_embedding[0,2]
            tf_idf_sum += query_passage_embedding[2]
            count += 1
            non_relavancePot.append(query_passage_embedding[1:])

        non_relavancePot = np.array(non_relavancePot)
        print(non_relavancePot.shape)
        
        self.nonrelavanceCos = cos_sum/count
        # self.nonrelavanceDist = dist_sum/count
        self.nonrelavanceTFIDF = tf_idf_sum/count

        print('Cos: relevance {}, non relevance {}'.format(self.relavanceCos, self.nonrelavanceCos))
        # print('Dist: relevance {}, non relevance {}'.format(self.relavanceDist, self.nonrelavanceDist))
        print('tf: relevance {}, non relevance {}'.format(self.relavanceTFIDF, self.nonrelavanceTFIDF))

        # visual_embedding = np.vstack((relavancePot,non_relavancePot))
        # print('All embedding shape ', visual_embedding.shape)
        # visual_embedding = TSNE(n_components=2).fit_transform(visual_embedding)

        label_txt = 'relavence: average cost={}, tf={}'.format(round(self.relavanceCos, 2), round(self.relavanceTFIDF, 2))
        plt.scatter(relavancePot[:,0], relavancePot[:,1], s=5, marker='x', c='r', label=label_txt)
        label_txt = 'non-relavence: average cos={}, tf={}'.format(round(self.nonrelavanceCos, 2), round(self.nonrelavanceTFIDF, 2))
        plt.scatter(non_relavancePot[:,0], non_relavancePot[:,1], s=2, marker='o', facecolors='none', edgecolors='b', label=label_txt)
        plt.legend()
        plt.xlabel('Cos Similarity')
        plt.ylabel('Log Freq')
        plt.savefig('Visual_dist_logWeight.png')
        # plt.show()

        # print('shape check ', relavancePot[:,0].shape, non_relavancePot[:,0].shape)
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # label_txt = 'relavence: average Cos={}, tf*idf={}, dist={}'.format(round(self.relavanceCos, 2), round(self.relavanceTFIDF, 2), round(self.relavanceDist, 2))
        # ax.scatter(relavancePot[:,0], relavancePot[:,1], relavancePot[:,2], c='r', marker='x', label = label_txt )
        # label_txt = 'non-relavence: average Cos={}, tf*idf={}, dist={}'.format(round(self.nonrelavanceCos, 2), round(self.nonrelavanceTFIDF, 2), round(self.nonrelavanceDist, 2))
        # ax.scatter(non_relavancePot[:,0], non_relavancePot[:,1], non_relavancePot[:,2], edgecolors='b', marker='o', label = label_txt )
        
        # # ax.scatter(non_relavancePot[:,0], non_relavancePot[:,1], non_relavancePot[:,2], marker='o', edgecolors='b', label = label_txt )
        # ax.set_xlabel('Cos Similarity')
        # ax.set_ylabel('Euclidean Distance')
        # ax.set_zlabel('TF-IDF')
        # plt.legend()
        # plt.savefig('Visual_cos_dist_tfidf.png')
        # plt.show()

    def randomChoice_unbalancedData(self, batch_size, oversample=True):
        if oversample:
            pos = max(0.5, self.pos)
        else:
            pos = max(0.1, self.pos)
        random_pos_line = max(1, int(pos*batch_size))

        temp = np.random.choice(len(self.relavancePass), random_pos_line, replace=False)
        pos_line = [self.relavancePass[x] for x in temp] 

        allRows = np.arange(1, self.getLength()+1)
        choices = np.random.choice(allRows,  batch_size-random_pos_line, replace=False)
        choices = np.concatenate((pos_line, choices))
        np.random.shuffle(choices)
        
        
        return choices


    # def getItem_tf(self, batch_size=1):


    def getitem(self, index):
        # allRows = np.arange(1, self.num_line+1)
        vectorize_index = np.array(index)
        #print('check choices ',vectorize_index)
        # skip = [x for x in allRows if x not in vectorize_index ]
        #reader = pd.read_csv(self.train_path, sep='\t', skiprows=skip)
        reader = self.reader[self.reader.index.isin(vectorize_index)]

        batch_query_pass_embedding = []
        batch_query_pass_labels = []

        for i, row in reader.iterrows():

            query_raw = row['queries'].lower()
            passage_raw = row['passage'].lower()

            qid = str(row['qid'])
            query_passage_embedding = generateEmbedding(self.embedding, query_raw, passage_raw, self.idf[qid])
            cur_label = np.array([row['relevancy']])

            batch_query_pass_embedding.append( query_passage_embedding)
            batch_query_pass_labels.append( cur_label)

        batch_query_pass_embedding = np.stack(batch_query_pass_embedding, axis=0)
        batch_query_pass_labels = np.stack(batch_query_pass_labels, axis=0)
        # print('after stack ', batch_query_pass_embedding.shape, batch_query_pass_labels.shape)

        #print(batch_query_pass_embedding.shape, batch_query_pass_labels.shape)
        # print(batch_query_pass_labels)
        return batch_query_pass_embedding, batch_query_pass_labels


    def getItemByGroup(self, queryID, mode='val', downSampling=False, downSampleRate=0.1):
        #get query passage by group

        candidate_pass = self.reader[self.reader.qid.eq(int(queryID))]
        # print(candidate_pass.shape)
        # print(candidate_pass)
        dict_empty = True
        # print('can reader')
        # print(candidate_pass)

        if downSampling:
            num_line = candidate_pass.shape[0]
            # print('num line ', num_line)
            index = np.random.choice(num_line, int(num_line*downSampleRate), replace=False)
            row_indx_lst = candidate_pass.index.tolist()
            selected_row = [row_indx_lst[x] for x in index]
            relavance_row = candidate_pass.relevancy.eq(1.0).index.tolist()[0]
            selected_row.append(relavance_row)
            #print('check choices ',vectorize_index)
            # skip = [x for x in allRows if x not in vectorize_index ]
            candidate_pass = candidate_pass[candidate_pass.index.isin(selected_row)]
            # print(queryID)
            # print('unique ', candidate_pass.qid.unique().tolist(), candidate_pass.shape)
            # print(candidate_pass)
            # a=t
        batch_pid = []
        batch_label = []
        batch_query_pass_embedding = []
        for i, row in candidate_pass.iterrows():

            query = row['queries'].lower()
            #print(query)
            passage = row['passage'].lower()
            
            #print(passage)
            # qid = str(row['qid'])
            query_passage_embedding = generateEmbedding(self.embedding, query, passage, self.idf[str(queryID)], raw=raw)

            cur_pid = [row['pid']]
            cur_label = [row['relevancy']]
            batch_pid.append(cur_pid)
            batch_label.append(cur_label)
            batch_query_pass_embedding.append(query_passage_embedding)

        batch_pid = np.stack(batch_pid, axis=0)
        batch_label = np.stack(batch_label, axis=0)
        batch_query_pass_embedding = np.stack(batch_query_pass_embedding, axis=0)

        #print(batch_query_pass_embedding.shape)
        if mode == 'val':
            return batch_pid, batch_query_pass_embedding
        elif mode == 'Train':
            return batch_label, batch_query_pass_embedding