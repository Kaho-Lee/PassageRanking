from utils import *
from DataGenerator import dataPipeLine


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
