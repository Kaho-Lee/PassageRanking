import numpy as np

def avg_Precision( results, labels):
    count = 0
    relavance = 0
    precision_lst = []

    #pid: score
    for pid in results.keys():
        count += 1
        if labels[pid] == 1.0:
            relavance += 1
            precision = relavance/count
            precision_lst.append(precision)
    precision_lst = np.array(precision_lst)

    if len(precision_lst) == 0:
        return 0
    else:
        return np.mean(precision_lst)

def NDCG(query_ID, results, labels):
    log_2 = np.arange(len(list(results.keys())))+2
    log_2 = np.log2(log_2)
    '''
    candidate_ranked_pass = {k: v for k, v in sorted(candidate_ranked_pass.items(), \
        key=lambda item: item[1], reverse=True)}
    '''
    labels = {k: v for k, v in sorted(labels.items(), \
        key=lambda item: item[1], reverse=True)}
    
    IDCG = np.array([labels[k] for k in list(labels)[:len(list(results.keys())) ]])
    #print(IDCG)
    DCG = np.array([labels[k] for k in list(results)])
    #print(DCG)
    
    DCG = np.sum(np.divide(DCG, log_2))
    IDCG = np.sum(np.divide(IDCG, log_2))
    
    return DCG/IDCG