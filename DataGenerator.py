import os
import numpy as np
import json
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
        
        reader = pd.read_csv(self.train_path, sep='\t')
        #print(reader)
        relevancy_lines = reader[reader.relevancy.eq(1.0)]
        #print(relevancy_lines)
        self.relavancePass = relevancy_lines.index.tolist()
        non_relavancy_line = reader[reader.relevancy.eq(0.0)]
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
        reader = pd.read_csv(self.train_path, sep='\t')
        #print(reader)
        relevancy_lines = reader[reader.relevancy.eq(1.0)]
        cos_sum = 0
        count = 0 
        dist_sum = 0
        relavancePot = []
        print('cal relavence sim')
        for i, row in relevancy_lines.iterrows():
            query = re.split('(\W)', row['queries'].lower()) #sperate the string by non-word character
            # query = word_tokenize(row['queries'].lower())
            #print(query)
            passage = re.split('(\W)', row['passage'].lower())
            # passage = word_tokenize(row['passage'].lower())
            #print(passage)
            query_lst = re.sub("[^\sa-zA-Z0-9]+", ' ', row['queries'].lower()).split(' ')
            passage_lst = re.sub("[^\sa-zA-Z0-9]+", ' ', row['passage'].lower()).split(' ')

            query_passage_embedding = generateEmbedding(self.embedding, query, passage)

            qid = str(row['qid'])
            log_freq = log_freqWeighting(query_lst, passage_lst, self.idf[qid])
            query_passage_embedding = np.hstack((query_passage_embedding, log_freq))

            cos_sum += query_passage_embedding[0,1]
            dist_sum += query_passage_embedding[0,2]
            count += 1
            relavancePot.append(query_passage_embedding[0, 1:])
        relavancePot = np.array(relavancePot)
        print(relavancePot.shape)
        self.relavanceCos = cos_sum/count
        self.relavanceDist = dist_sum/count

        relevancy_lines = reader[reader.relevancy.eq(0.0)]
        cos_sum = 0
        count = 0 
        dist_sum = 0
        non_relavancePot = []
        print('cal non relavence sim')
        sample_reader = relevancy_lines.sample(int(relevancy_lines.shape[0]*0.01))
        for i, row in sample_reader.iterrows():
            query = re.split('(\W)', row['queries'].lower()) #sperate the string by non-word character
            # query = word_tokenize(row['queries'].lower())
            #print(query)
            passage = re.split('(\W)', row['passage'].lower())
            # passage = word_tokenize(row['passage'].lower())
            #print(passage)
            query_lst = re.sub("[^\sa-zA-Z0-9]+", ' ', row['queries'].lower()).split(' ')
            passage_lst = re.sub("[^\sa-zA-Z0-9]+", ' ', row['passage'].lower()).split(' ')

            query_passage_embedding = generateEmbedding(self.embedding, query, passage)
            qid = str(row['qid'])
            log_freq = log_freqWeighting(query_lst, passage_lst, self.idf[qid])
            query_passage_embedding = np.hstack((query_passage_embedding, log_freq))

            cos_sum += query_passage_embedding[0,1]
            dist_sum += query_passage_embedding[0,2]
            count += 1
            non_relavancePot.append(query_passage_embedding[0, 1:])

        non_relavancePot = np.array(non_relavancePot)
        print(non_relavancePot.shape)
        self.nonrelavanceCos = cos_sum/count
        self.nonrelavanceDist = dist_sum/count

        print('Cos: relevance {}, non relevance {}'.format(self.relavanceCos, self.nonrelavanceCos))
        print('Log Freq Weights: relevance {}, non relevance {}'.format(self.relavanceDist, self.nonrelavanceDist))
        label_txt = 'relavence: average Cos={}, log(tf)={}'.format(round(self.relavanceCos, 2), round(self.relavanceDist, 2))
        plt.scatter(relavancePot[:,0], relavancePot[:,1], s=5, marker='x', c='r', label=label_txt)
        label_txt = 'non-relavence: average Cos={}, log(tf)={}'.format(round(self.nonrelavanceCos, 2), round(self.nonrelavanceDist, 2))
        plt.scatter(non_relavancePot[:,0], non_relavancePot[:,1], s=5, marker='o', facecolors='none', edgecolors='b', label=label_txt)
        plt.legend()
        plt.xlabel('Cos Similarity')
        plt.ylabel('Log Freq')
        plt.savefig('Visual_cos_logWeight.png')
        #plt.show()

    def VisualEmbedding(self):
        reader = pd.read_csv(self.train_path, sep='\t')
        #print(reader)
        relevancy_lines = reader[reader.relevancy.eq(1.0)]
        
        relavancePot = []
        print('cal relavence sim')
        for i, row in relevancy_lines.iterrows():
            query = re.split('(\W)', row['queries'].lower()) #sperate the string by non-word character
            # query = word_tokenize(row['queries'].lower())
            #print(query)
            passage = re.split('(\W)', row['passage'].lower())
            # passage = word_tokenize(row['passage'].lower())
            #print(passage)
            query_passage_embedding = generateEmbedding(self.embedding, query, passage)
            # print(query_passage_embedding)
            relavancePot.append(query_passage_embedding[0])

        relavancePot = np.array(relavancePot)
        print(relavancePot.shape)
        num_relavance = relavancePot.shape[0]

        relevancy_lines = reader[reader.relevancy.eq(0.0)]
        
        non_relavancePot = []
        print('cal non relavence sim')
        sample_reader = relevancy_lines.sample(int(relevancy_lines.shape[0]*0.001))
        for i, row in sample_reader.iterrows():
            query = re.split('(\W)', row['queries'].lower()) #sperate the string by non-word character
            # query = word_tokenize(row['queries'].lower())
            #print(query)
            passage = re.split('(\W)', row['passage'].lower())
            # passage = word_tokenize(row['passage'].lower())
            #print(passage)
            query_passage_embedding = generateEmbedding(self.embedding, query, passage)
            non_relavancePot.append(query_passage_embedding[0])

        non_relavancePot = np.array(non_relavancePot)
        print(non_relavancePot.shape)

        tot_pot = np.vstack((relavancePot, non_relavancePot))
        print(tot_pot.shape)

        pca = PCA(n_components=50)
        pca.fit(tot_pot)
        tot_pot = pca.transform(tot_pot)
        print('dim reduct ', tot_pot.shape)
        print(pca.explained_variance_ratio_)
        X_embedded = TSNE(n_components=2).fit_transform(tot_pot)
        print(X_embedded.shape)


        plt.scatter(X_embedded[:num_relavance,0], X_embedded[:num_relavance,1], s=20, marker='o', label='relavence')
        plt.scatter(X_embedded[num_relavance:,0], X_embedded[num_relavance:,1], s=20, marker='*', label='non-relavence')
        plt.legend()
        plt.xlabel('TSNE Dim 1')
        plt.ylabel('TSNE Dim 1')
        plt.savefig('queryTSNEVisual_PCA50.png')
        #plt.show()
        aa= tt

    def randomChoice_unbalancedData(self, batch_size):
        pos = max(0.5, self.pos)
        random_pos_line = max(1, int(pos*batch_size))
        
        temp = np.random.choice(len(self.relavancePass), random_pos_line, replace=False)
        pos_line = [self.relavancePass[x]+1 for x in temp] 

        allRows = np.arange(1, self.getLength()+1)
        choices = np.random.choice(allRows,  batch_size-random_pos_line, replace=False)
        choices = np.concatenate((pos_line, choices))
        np.random.shuffle(choices)
        #print('choices ', choices)
        return choices

    def getitem(self, index):
        allRows = np.arange(1, self.num_line+1)
        vectorize_index = np.array(index)
        #print('check choices ',vectorize_index)
        skip = [x for x in allRows if x not in vectorize_index ]
        reader = pd.read_csv(self.train_path, sep='\t', skiprows=skip)
        for i, row in reader.iterrows():
            query = re.split('(\W)', row['queries'].lower()) #sperate the string by non-word character
            # query = word_tokenize(row['queries'].lower())
            #print(query)
            passage = re.split('(\W)', row['passage'].lower())
            # passage = word_tokenize(row['passage'].lower())
            #print(passage)

            query_lst = re.sub("[^\sa-zA-Z0-9]+", ' ', row['queries'].lower()).split(' ')
            passage_lst = re.sub("[^\sa-zA-Z0-9]+", ' ', row['passage'].lower()).split(' ')

            query_passage_embedding = generateEmbedding(self.embedding, query, passage)
            qid = str(row['qid'])
            log_freq = log_freqWeighting(query_lst, passage_lst, self.idf[qid])
            query_passage_embedding = np.hstack((query_passage_embedding, log_freq))

            cur_label = np.atleast_2d(np.array(row['relevancy']))
            if i == 0:
                batch_query_pass_embedding = query_passage_embedding
                batch_query_pass_labels = cur_label
            else:
                batch_query_pass_embedding = np.vstack((batch_query_pass_embedding, query_passage_embedding))
                batch_query_pass_labels = np.vstack((batch_query_pass_labels, cur_label))

        #print(batch_query_pass_embedding.shape, batch_query_pass_labels.shape)
        # print(batch_query_pass_labels)
        return batch_query_pass_embedding, batch_query_pass_labels