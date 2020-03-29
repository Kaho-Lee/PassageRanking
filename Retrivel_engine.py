import json
import os
import numpy as np

class Information_Rerivel:
    def __init__(self):
        pass

    def retrivel(self):
        pass

    def saveToText(self, queryID, reranked_candidate):
        with open('{}.txt'.format(self.name), 'a') as writeFile:
            rank = 1
            for key, value in zip(reranked_candidate.keys(), reranked_candidate.values()):
                writeFile.write('<{} A1 {} {} {} {}>\n'.format(queryID, key, rank, value, self.name))
                rank += 1
        writeFile.close()



class BM25(Information_Rerivel):
    '''
    needed param.
    dl: count in run time
    avdl: count in preprocess
    N count in class init, raw data section 
    n_i = len(pass[item].keys())-2, run time
    r_i and R set to 0
    '''
    def __init__(self, pass_inv_path, pass_raw_path, query_inv_path, src_path):
        super().__init__()
        self.passage_inv_stored = pass_inv_path

        with open(src_path+self.passage_inv_stored, 'r') as readFile:
            self.pass_inv_data = json.load(readFile)
        readFile.close()

        self.query_inv_stored = query_inv_path

        with open(src_path+self.query_inv_stored, 'r') as readFile:
            self.query_inv_data = json.load(readFile)
        readFile.close()

        self.passage_raw_stored = pass_raw_path

        with open(src_path+self.passage_raw_stored, 'r') as readFile:
            temp = json.load(readFile)
            self.pass_raw_data = temp['passage']
            self.avdl = temp['passage_avdl']
            self.query_pass = temp['query_pass']
        readFile.close()
        del temp

        self.name = 'BM25'
        if(os.path.exists('{}.txt'.format(self.name))):
            os.remove('{}.txt'.format(self.name))

        '''
        parameters for K calculation
        '''
        self.k1 = 1.2
        self.k2 = 100 #default, tunable
        self.b = 0.75 #default,tunable

    def retrivel(self, queryID, queryVal, show=True):
        '''
        use Binary Independence Model
        '''

        candidate_ranked_pass = {}
        r_i = np.sum(np.array([self.query_pass[queryID][k] for k in list(self.query_pass[queryID])]))
        R = np.sum(np.array([self.query_pass[queryID][k] for k in list(self.query_pass[queryID])])) 
        #get from filter the self.query_pass by values, label
        N = len(list(self.query_pass[queryID].keys()))

        terms = set(queryVal.split(' '))
        print(queryID, terms)
        for item in terms:
            if (item != '') and (item in self.pass_inv_data[queryID].keys()) and (item in self.query_inv_data.keys()):
                candidate = self.pass_inv_data[queryID][item] #fetch a single inverted index by one term in query
                n_i = len(candidate.keys())-2           
                qf_i = self.query_inv_data[item][queryID]

                for key in candidate.keys(): #id of pass
                    if key == item  or key == 'idf':
                        continue

                    f_i = candidate[key]
                    dl = len(self.pass_raw_data[key].split(' '))
                    #dl = self.pass_raw_data[key]
                    K = self.k1*((1-self.b) + self.b*(dl/self.avdl[queryID]))
                    score = np.log( ((r_i + 0.5)/(R - r_i + 0.5)) / ((n_i -r_i +0.5)/(N - n_i-R+r_i+0.5)))
                    score = score * ((self.k1+1)*f_i / (K+f_i)) *  ((self.k2+1)*qf_i / (self.k2+qf_i))

                    if key not in candidate_ranked_pass.keys():
                        candidate_ranked_pass[key] = score
                    else:
                        candidate_ranked_pass[key] += score
                        
                    # if key == '8085708':
                        # print(item, key)
                        # print("n_i {} N-n_i {} K {} f_i {} score {}".format(n_i, N-n_i, K, f_i, score))
                        # print("{} {} {}".format(item, '{} score'.format(key), candidate_ranked_pass[key]))
                        # print('\n')     

        candidate_ranked_pass = {k: v for k, v in sorted(candidate_ranked_pass.items(), \
        key=lambda item: item[1], reverse=True)}

        if len(candidate_ranked_pass.keys()) <= 100:
            reranked_candidate = candidate_ranked_pass
        else:
            reranked_candidate = {k: candidate_ranked_pass[k] for k in list(candidate_ranked_pass)[:100]}

        if show:
            i=0
            for k in reranked_candidate.keys():
                i += 1
                print('{} {} {}'.format(i, k, self.pass_raw_data[k]))

        return reranked_candidate