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
# import nltk
from nltk.tokenize import word_tokenize 
from sklearn.manifold import TSNE
#nltk.download('stopwords')


def saveToText(name, queryID, pid, y_pred):
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
    
    with open('{}.txt'.format(name), 'a') as writeFile:
        rank = 1
        for key, value in zip(reranked_candidate.keys(), reranked_candidate.values()):
            writeFile.write('<{} A1 {} {} {} {}>\n'.format(queryID, key, rank, value, name))
            rank += 1
    writeFile.close()
    return reranked_candidate
    
def ExtractPreTrained(pre_trained_model_path, stored_path):
    gloveEmbedding = {}
    count = 0
    with open(pre_trained_model_path, 'r') as readFile:
        for line in tqdm.tqdm(readFile):
            count+=1
            embedding = line.split(' ')
            values = [float(x) for x in embedding[1:]]
            gloveEmbedding[embedding[0]] = values
            # print(embedding)
            # print(gloveEmbedding)
            # break
    readFile.close()
    print(count, len(list(gloveEmbedding.keys())))
    with open(stored_path,'w') as writeFile:
        json.dump(gloveEmbedding, writeFile)
    writeFile.close()

def IDFOfDataSet(path, name):
    print('Generating IDF of current dataset!!!')
    reader = pd.read_csv(path, sep='\t')
    # stop_words = set(stopwords.words('english'))
    idf = {}
    df = {}

    qid = reader.qid.unique().tolist()
    print(len(qid))
    for id in tqdm.tqdm(qid):
        qid_reader =  reader[reader.qid.eq(int(id))]
        # print(qid_reader.qid.unique().tolist())
        idf[id] = {}
        df = {}
        for i, row in qid_reader.iterrows():
            # passage_lst = re.sub("[^\sa-zA-Z0-9]+", ' ', row['passage'].lower()).split(' ')
            passage_lst = re.split('(\W)', row['passage'].lower())
            for term in passage_lst:
                # if term not in stop_words and term != '':
                if term != '':
                    if term not in idf[id]:
                        idf[id][term] = 1
                        df[term] = set()
                        df[term].add(row['pid'])
                    else:
                        idf[id][term] += 1
                        df[term].add(row['pid'])
        
        for term in idf[id].keys():
            idf[id][term] = idf[id][term]/len(df[term])

        #break
    # print(idf)
    with open(name, 'w') as writeFile:
        json.dump(idf, writeFile)
    writeFile.close()



def metricToText(metricResults, metricName):
    with open(metricName+'.json', 'w') as writeFile:
        #writeFile.write('Query {}\n'.format(metricName))

        for key, value in metricResults.items():
            writeFile.write('<{}: {}> '.format(key, value))
    writeFile.close()
    return 0

def updateDict(dict_src, dict_des):
    for k_src in dict_src.keys():
        if k_src in dict_des:
            print('Updating: key {}'.format(k_src))
            dict_des[k_src] = dict_src[k_src]

def sigmoid(theta):
    theta[theta < -100] = -100
    return 1/(1+np.exp(-theta))

def sigmoid_grad(theta):
    sigmoid_logis = sigmoid(theta)
    return sigmoid_logis*(1-sigmoid_logis)

def accuracy(y, y_pred):
    num = y.shape[0]
    check = (y == y_pred)
    toSHow = np.hstack((check, y))
    toSHow = np.hstack((toSHow, y_pred))
    
    
    acc = np.sum(check)/num
    return acc

def generateEmbedding(embedding, query, passage, idf, raw=False):
    stemmer = SnowballStemmer("english") 
    stop_words = set(stopwords.words('english')) 
    query_term_count = 0
    query_embedding = np.zeros(50)

    rawq = query
    rawp = passage
    query = re.sub("(\W)", ' ', rawq).split(' ')
    # query_lst = re.sub("[^\sa-zA-Z0-9]+", ' ', rawq).split(' ')
    passage = re.sub("(\W)", ' ', rawp).split(' ')
    # passage_lst = re.sub("[^\sa-zA-Z0-9]+", ' ', rawp).split(' ')
    # print(rawq)
    # print(query)
    # print(rawp)
    # print(passage)
    # print('\n')

    #tf calculation
    query_tf = {}
    for term in query:
        if term not in query_tf:
            query_tf[term] = 1
        else:
            query_tf[term] += 1

    pass_tf = {}
    for term in passage:
        if term not in pass_tf:
            pass_tf[term] = 1
        else:
            pass_tf[term] += 1

    for term in query:
            
        if term != ' ' and term != '' and (term not in stop_words):#
            query_term_count += 1
        else:
            continue

        if term in embedding:
            norm_embedding = np.array(embedding[term])
            norm_embedding = (norm_embedding/np.linalg.norm(norm_embedding)) #* ( query_tf[term]*np.log10(idf[term]))
            query_embedding += norm_embedding
                
        else:
            #print('Not Found: ', term)
            new_term = stemmer.stem(term)
            if new_term in embedding:
                norm_embedding = np.array(embedding[new_term])
                norm_embedding = (norm_embedding/np.linalg.norm(norm_embedding)) #* ( query_tf[term]*np.log10(idf[term]))
                query_embedding += norm_embedding

    query_embedding = query_embedding/query_term_count

    passage_term_count = 0
    passage_embedding = np.zeros(50)
    for term in passage:
        if term != ' ' and term != '' and (term not in stop_words):#
            passage_term_count += 1
        else:
            continue

        if term in embedding:
            norm_embedding = np.array(embedding[term])
            norm_embedding = (norm_embedding/np.linalg.norm(norm_embedding)) * ( pass_tf[term]*np.log10(idf[term]))
            passage_embedding += norm_embedding
        else:
            #print('Not Found: ', term)
            new_term = stemmer.stem(term)
            if new_term in embedding:
                norm_embedding = np.array(embedding[new_term])
                norm_embedding = (norm_embedding/np.linalg.norm(norm_embedding)) * ( pass_tf[term]*np.log10(idf[term]))
                passage_embedding += norm_embedding

    passage_embedding = passage_embedding/passage_term_count

    if raw:
        embed = np.hstack((passage_embedding, query_embedding))
        # return np.atleast_2d(embed)
        return embed
    else:

        if passage_term_count == 0  or query_term_count ==0:
            # print('Count: query {} passage {}'.format(query_term_count, passage_term_count))
            # print(query)
            # print(passage)
            cos_sim = [0]
        elif np.linalg.norm(passage_embedding) == 0.0 or np.linalg.norm(query_embedding)==0.0:
            # print('Norm: query {} passage {}'.format(np.linalg.norm(query_embedding), np.linalg.norm(passage_embedding)))
            # print(query)
            # print(passage)
            cos_sim = [0]
        else:
            cos_sim = (passage_embedding @ np.atleast_2d(query_embedding).T)/(np.linalg.norm(passage_embedding)*np.linalg.norm(query_embedding))

        # dist = np.linalg.norm(passage_embedding - query_embedding)

        tf = log_freqWeighting(query, passage, idf)

        query_passage_embedding = np.array([1, cos_sim[0], tf]) 
        
        # query_passage_embedding = np.hstack((query_embedding, passage_embedding))
        # query_passage_embedding = np.hstack((query_passage_embedding, [1]))
        
        # return np.atleast_2d( query_passage_embedding)
        return  query_passage_embedding

def log_freqWeighting(query_lst, passage_lst, idf):

    stop_words = set(stopwords.words('english')) 
    score = 0
    pass_dict = {}
    for term in passage_lst:
        # if term not in stop_words:
        if term != '' and term != ' ':
            if term not in pass_dict:
                pass_dict[term] = 1
            else:
                pass_dict[term] += 1
    
    for term in query_lst:
        # if term not in stop_words:
            if term in pass_dict:
                # tf_idf.append(pass_dict[term] * np.log10(idf[term]))
                # score += pass_dict[term] * np.log10(idf[term])
                score += 1 + np.log10(pass_dict[term])
                # score += 1
    
    # cos normalize
    # cos_normalizer = 0
    # for term in passage_lst:
    #     if term != '' and term != ' ':
    #         # cos_normalizer += 1 + np.log10(pass_dict[term])
    #         cos_normalizer += (pass_dict[term] * np.log10(idf[term]))**2
    # score = score/np.sqrt(cos_normalizer)

    return score

# def Data_Embedding(embedding, val_reader, queryID, val_idf, mode='val', downSampling=False, downSampleRate=0.5, raw=False):
#     #get query passage by group

#     candidate_pass = val_reader[val_reader.qid.eq(int(queryID))]
#     # print(candidate_pass.shape)
#     # print(candidate_pass)
#     dict_empty = True
#     # print('can reader')
#     # print(candidate_pass)

#     if downSampling:
#         num_line = candidate_pass.shape[0]
#         # print('num line ', num_line)
#         index = np.random.choice(num_line, int(num_line*downSampleRate), replace=False)
#         row_indx_lst = candidate_pass.index.tolist()
#         selected_row = [row_indx_lst[x] for x in index]
#         relavance_row = candidate_pass.relevancy.eq(1.0).index.tolist()[0]
#         selected_row.append(relavance_row)
#         candidate_pass = candidate_pass[candidate_pass.index.isin(selected_row)]
        
#     batch_pid = []
#     batch_label = []
#     batch_query_pass_embedding = []
#     for i, row in candidate_pass.iterrows():

#         query = row['queries'].lower()
#         #print(query)
#         passage = row['passage'].lower()
        
#         #print(passage)
#         # qid = str(row['qid'])
#         query_passage_embedding = generateEmbedding(embedding, query, passage, val_idf, raw=raw)

#         cur_pid = [row['pid']]
#         cur_label = [row['relevancy']]
#         batch_pid.append(cur_pid)
#         batch_label.append(cur_label)
#         batch_query_pass_embedding.append(query_passage_embedding)

#     batch_pid = np.stack(batch_pid, axis=0)
#     batch_label = np.stack(batch_label, axis=0)
#     batch_query_pass_embedding = np.stack(batch_query_pass_embedding, axis=0)

#     #print(batch_query_pass_embedding.shape)
#     if mode == 'val':
#         return batch_pid, batch_query_pass_embedding
#     elif mode == 'Train':
#         return batch_label, batch_query_pass_embedding
