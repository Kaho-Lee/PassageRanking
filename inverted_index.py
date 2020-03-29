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

def stemming(s):
    ps = SnowballStemmer("english") 

    s_lst = s.split(' ')
    s_lst_stem = []
    for word in s_lst:
        if word != '':
            s_lst_stem.append(ps.stem(word))
    s_stem = ' '.join(s_lst_stem)
    return s_stem

def fileToLst(path):
    '''
    Text preproccessing
    '''
    query_dict = {}
    pass_dict = {}
    query_pass = {}
    
    print('Preprocessing and stemming, please wait!')
    with open(path) as tsvfile:
        tsvreader = csv.reader(tsvfile, delimiter="\t")

        first_row = next(tsvreader)
        for line in tsvreader:
            # query = stemming(line[2])
            # passage = stemming(line[3])

            query = line[2]
            passage = line[3]

            query = re.sub("[^\sa-zA-Z0-9]+", ' ', query)
            passage = re.sub("[^\sa-zA-Z0-9]+", ' ', passage)
            query_dict[line[0]] = query.lower()
            pass_dict[line[1]] = passage.lower()

            if line[0]  not in query_pass.keys():
                query_pass[line[0]] = {}
                
                query_pass[line[0]][line[1]] = float(line[4])
                #query_pass[line[0]].append((line[1], float(line[4])))
                
                
            else:
                #query_pass[line[0]].append(line[1])
                query_pass[line[0]][line[1]] = float(line[4])
                #query_pass[line[0]].append((line[1], float(line[4])))
                

    tsvfile.close()
    return query_dict, pass_dict, query_pass

def GenInvIndex(query_dict, pass_dict, query_pass,path):
    query_word_set = set()
    passage_word_set = {}

    with open(path+'query_passage_raw.json', 'w') as writeFile:
        json_qnp = {}
        json_qnp['query'] = query_dict
        json_qnp['passage'] = pass_dict
        json_qnp['query_pass'] = query_pass

        totQuery = len(query_dict.keys())
        print('tot query ', totQuery)
        totQuery_len = 0
        for query_val in query_dict.values():
            for item in query_val.split(' '):
                totQuery_len += 1
                if item not in query_word_set and item != '':
                    query_word_set.add(item)
        print(len(query_word_set))
        json_qnp['query_avdl'] = totQuery_len/totQuery

        
        json_qnp['passage_avdl'] = {}
        for queryID in query_pass.keys():
            
            passage_word_set[queryID] = set()
            totPass_len = 0
            totPass = 0
            for pass_id in query_pass[queryID].keys():
                totPass+=1
                pass_val = pass_dict[pass_id]
                for item in pass_val.split(' '):
                    if item != '':
                        totPass_len += 1
                        if item not in passage_word_set[queryID]:
                            passage_word_set[queryID].add(item)

            json_qnp['passage_avdl'][queryID] = totPass_len/totPass
        print(len(passage_word_set))
        
        json.dump(json_qnp, writeFile)

    writeFile.close()
    del json_qnp

    with open(path+'query_invIndex.json', 'w') as writeFile:
        json_invIndex = {}
        json_invIndex = {}

        for item in query_word_set:
            if item == '':
                continue
            json_invIndex[item] ={}
            json_invIndex[item][item] = 0
       
        for query_id in query_dict.keys():
            for item in query_dict[query_id].split(' '):
                if item == '':
                    continue
                if query_id not in json_invIndex[item].keys():
                    json_invIndex[item][query_id] = 1
                    json_invIndex[item][item] += 1
                else:
                    json_invIndex[item][query_id] += 1
                    json_invIndex[item][item] += 1
        
        for item in json_invIndex.keys():
            idf = np.log10(totQuery/(len(json_invIndex[item].keys())-1))
            json_invIndex[item]['idf'] = idf
        
        json.dump(json_invIndex, writeFile)
    writeFile.close()
    del json_invIndex

    with open(path+'pass_invIndex.json', 'w') as writeFile:
        json_invIndex = {}
        for queryID in passage_word_set.keys():
            json_invIndex[queryID] = {}
            totPass = len(list(query_pass[queryID].keys()))

            for item in passage_word_set[queryID]:
                if item == '':
                    continue
                json_invIndex[queryID][item] ={}
                json_invIndex[queryID][item][item] = 0
        
            for pass_id in query_pass[queryID].keys():
                pass_val = pass_dict[pass_id]
                for item in pass_val.split(' '):
                    if item == '':
                        continue
                    if pass_id not in json_invIndex[queryID][item].keys():
                        json_invIndex[queryID][item][pass_id] = 1
                        json_invIndex[queryID][item][item] += 1
                    else:
                        json_invIndex[queryID][item][pass_id] += 1
                        json_invIndex[queryID][item][item] += 1

            for item in json_invIndex[queryID].keys():
                idf = np.log10(totPass/(len(json_invIndex[queryID][item].keys())-1))
                json_invIndex[queryID][item]['idf'] = idf
        
        json.dump(json_invIndex, writeFile)
    writeFile.close()
    del json_invIndex

if __name__=='__main__':

    #generate the inverted index and parse the data, raw and preprocessed, to json files
    can_lst = '/Users/leekaho/Desktop/part2/validation_data.tsv'
    path = '/Users/leekaho/Desktop/part2/'
    query_dict, pass_dict, query_pass = fileToLst(can_lst)
    print('gen inv index')
    GenInvIndex(query_dict, pass_dict, query_pass, path)

    
