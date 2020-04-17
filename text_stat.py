#import pandas as pd
import numpy as np
import os
import re
import threading
import sys
import json
import time
import matplotlib.pyplot as plt 
from scipy import stats, special
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

def fileToList(filePath):
    f = open(filePath, "r", encoding="ISO-8859-1")

    f_lines = f.readlines()
    i = 0
    file_list = []
    passage = 0
    print('Preprocessing and stemming')
    for line in tqdm.tqdm(f_lines):
        temp = line.strip().lower()
        temp = re.sub("[^\sa-zA-Z0-9]+", ' ', temp)
        temp = stemming(temp)
        temp = temp.split(' ')
        
        file_list.append(temp)
        i+=1
    f.close()
    return file_list


def GenStat(file_list):
    word_set = set()

    for line in file_list:
        for item in line:
            if item not in word_set:
                word_set.add(item)
    
    csv_catogory =  np.array(list(word_set))
    print(csv_catogory.shape, csv_catogory)
    #csv_catogory = np.hstack((np.array(['passage_id']),csv_catogory))
    print(csv_catogory)
    i = 0

    '''
    The following un-commended part is to generate the inverted index
    for the dataset/passage_collection_new.txt
    '''
    # with open('passage_collection_stat.json', 'w') as writeFile:
    #     json_toStore = {}
    #     json_tot_toStore = {}
    #     #temp = np.hstack((np.array(['the']),temp))
    #     #temp = np.hstack((np.array(['the']),csv_catogory[:20]))
    #     for item in csv_catogory:   
    #         if item == '':
    #             continue
    #         json_toStore[item] = {}
    #         json_tot_toStore[item] = 0      
    #     for passage in range(len(file_list)):
    #         print(passage+1, end=' ', flush=True)
    #         for entry in file_list[passage]:
    #             if entry == '':
    #                 continue
    #             json_tot_toStore[entry] += 1
    #             if passage in json_toStore[entry].keys():
    #                 json_toStore[entry][passage] += 1
    #             else:
    #                 json_toStore[entry][passage] = 1
    #     json.dump(json_toStore, writeFile)
    # writeFile.close()

    json_tot_toStore = {}
    for item in csv_catogory:   
        if item == '':
            continue
        
        json_tot_toStore[item] = 0
    print('Generating frequency count for each word')
    for passage in tqdm.tqdm(range(len(file_list))):
        for entry in file_list[passage]:
            if entry == '':
                continue
            json_tot_toStore[entry] += 1
    print('\n')
    

    json_tot_toStore = {k: v for k, v in sorted(json_tot_toStore.items(), key=lambda item: item[1], reverse=True)}

    with open('passage_collection_tot_stat.json', 'w') as writeFile:
        json.dump(json_tot_toStore, writeFile)
    writeFile.close()
    print('\n')

def zipfs_law(tot_stat_file):

    with open(tot_stat_file, "r") as read_file:
        data = json.load(read_file)
    read_file.close()

    freq = []
    tot = 0
    for v in data.values():
        freq.append(v)
        tot += v
    freq = np.array(freq)
    print(freq/tot)
    rank100 = np.arange(100)+1
    rank_all = np.arange(freq.shape[0])+1

    '''
    Do the linear regression for the log(frequency)-log(rank) plot with fixed slope value of 1
    '''
    truncate = 5000
    x = np.vstack((np.ones(freq.shape[0]), np.log(rank_all)))
    x = x.T
    # x = x[:truncate, :]
    y = np.atleast_2d(np.log(freq))
    y = y.T
    # y = y[:truncate, :]
   
    w_opt = np.linalg.inv(x.T @ x ) @ x.T @ y
    print(w_opt)
    a = 1.07
    print(stats.kstest(freq/tot,'zipf', args=[a]))
    log_k = np.mean(np.log(rank_all)+np.log(freq))
    print(log_k)
    mse = np.log(freq) - np.log(rank_all) + log_k
    mse = np.sum(np.power(mse, 2))
    print(mse)

    y = np.log(freq)
    y_bar = np.mean(y)
    f = np.log(rank_all)*(-1)+log_k
    ss_tot = np.sum((y-y_bar)*(y-y_bar))
    ss_res = np.sum((y-f)*(y-f))
    r_square = 1 - (ss_res/ss_tot)

    fig1, ax1 = plt.subplots()
    ax1.plot(np.log(rank_all), np.log(freq), '.', label='Experimental Data')
    ax1.plot(np.log(rank_all), np.log(rank_all)*(-1)+log_k, label='Theoretical Line')
    ax1.set_title("log(k)={:.2f}    R^2={:.2f}".format(log_k, r_square) )
    ax1.set_xlabel('log(rank)')
    ax1.set_ylabel('log(frequency)')
    ax1.legend()
    plt.show()

    fig2, ax2 = plt.subplots()
    ax2.bar(rank_all[:150], freq[:150]/tot, label = 'Probality of Words')
    ax2.plot(rank_all[:150], stats.zipf.pmf(rank_all[:150], a), '.-', color = 'r', label="Zipf's Law with alpha = 1")
    # ax2.plot(rank_all[:50], 1/(rank_all[:50]**(a))*special.zetac(a), 'r')
    ax2.set_xlabel('Rank\n(By decreasing order)')
    ax2.set_ylabel('Probability\n(of occurence)')
    ax2.legend()
    plt.show()


if __name__=='__main__':

    try:
        path = int(input("Type 1 for generating the text statistic; Type 2 for showing the terms statistic with Zipf's Law    "))

    except:
        path = 1
    
    if path == 1:
        passage_collection_path = "dataset/passage_collection_new.txt"
        file_list = fileToList(passage_collection_path)
        GenStat(file_list)
    elif path == 2:
        tot_stat_file = 'passage_collection_tot_stat.json'
        zipfs_law(tot_stat_file)

    
    
    